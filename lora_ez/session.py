"""Multi-turn chat session with persistent history and auto-summarisation.

Memory layout of ``session.history``::

    [description anchor]?   <- immutable persona (always first, never lost)
    [system_prompt anchor]? <- current instruction (updated by /system-prompt)
    [Memory: <summary>]?    <- compressed FAR memory (older turns)
    <recent turns...>       <- raw recent window, kept verbatim

When the history exceeds ``max_tokens``, ONLY the old turns are compressed
into a summary; the persona anchors and the last few turns stay intact.

Use::

    # Programmatic multi-turn
    with model.chat_session("./chat.json", auto_save=True) as s:
        s > "Hello"
        s > "How are you?"
        print(s.history)

    # Interactive loop
    with model.chat_session(auto_save=False) as s:
        s.run()                # type messages at the prompt
"""

import json
from pathlib import Path

import torch

_KEEP_RECENT = 4  # raw messages kept verbatim at the tail when summarising


class _ChatSession:
    """Multi-turn chat session with persistent history and auto-summarisation.

    Not meant to be instantiated directly — use ``LoraModel.chat_session()``.
    """

    def __init__(self, model, save_path: str | None = None,
                 auto_save: bool = False, max_tokens: int = 4096,
                 system_prompt: str | None = None):
        """Multi-turn chat session with persistent history and auto-summarisation.

        Parameters
        ----------
        model : LoraModel
            The model instance that powers generation.
        save_path : str or None
            Path to a JSON file where history is loaded on enter
            and (if ``auto_save``) written on exit.  ``None`` means
            no persistence.
        auto_save : bool
            If ``True``, write history to ``save_path`` when the
            ``with`` block exits.
        max_tokens : int
            Upper bound on total tokenised history before automatic
            summarisation of the OLD turns kicks in (default 4096).
        system_prompt : str or None
            System-level instructions prepended to every turn.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.save_path = Path(save_path) if save_path else None
        self.auto_save = auto_save
        self.max_tokens = max_tokens
        self.history: list[dict] = []
        self._sync_anchors()

    # -- persona anchors -----------------------------------------------------

    def _effective_sp(self) -> str | None:
        """Session system prompt, falling back to the model's own."""
        return self.system_prompt if self.system_prompt is not None \
            else self.model.system_prompt

    def _anchors(self) -> list[dict]:
        """The immutable leading system messages (description, system prompt)."""
        msgs = []
        if self.model.description:
            msgs.append({"role": "system", "content": self.model.description})
        sp = self._effective_sp()
        if sp:
            msgs.append({"role": "system", "content": sp})
        return msgs

    def _sync_anchors(self):
        """Re-materialise persona anchors at the front of ``history``.

        Called on init / load / ``/system-prompt`` so the persona is always
        the first thing the model sees, even after summarisation.
        """
        anchors = self._anchors()
        body = self.history
        # drop stale leading anchors that are about to be re-added
        while body and body[0]["role"] == "system" and any(
                m["content"] == body[0]["content"] for m in anchors):
            body.pop(0)
        self.history = anchors + body

    def set_system_prompt(self, text: str):
        """Update the system prompt for the rest of the session."""
        self.system_prompt = text
        self._sync_anchors()

    # -- context manager -----------------------------------------------------

    def __enter__(self):
        if self.save_path and self.save_path.exists():
            self.history = json.loads(self.save_path.read_text())
        self._sync_anchors()
        return self

    def __exit__(self, *args):
        if self.auto_save and self.save_path:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_path.write_text(
                json.dumps(self.history, ensure_ascii=False, indent=2))
        self.history.clear()

    # -- chat -----------------------------------------------------------------

    def chat(self, prompt: str, **kwargs) -> str:
        """Send a message and get the assistant reply, auto-appending to history.

        Old turns are auto-summarised when the tokenised history exceeds
        ``max_tokens``; persona anchors and recent turns are kept verbatim.
        """
        self._maybe_summarize(prompt)
        self.history.append({"role": "user", "content": prompt})
        reply = self.model.chat(prompt, history=self.history[:-1],
                                system_prompt=self.system_prompt, **kwargs)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def __gt__(self, prompt: str) -> str:
        """``s > "prompt"`` is shorthand for ``s.chat("prompt")``."""
        return self.chat(prompt)

    def run(self):
        """Interactive REPL — type messages at a prompt, ``/exit`` to quit."""
        print(f"Chat session started.  Type /exit to quit.\n"
              f"{'='*50}")
        while True:
            try:
                prompt = input("User: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt:
                continue
            if prompt.startswith("/"):
                from .commands import dispatch
                if dispatch(self, prompt):
                    break
                continue
            reply = self.chat(prompt)
            print(f"{self.model:t}: {reply}")

    # -- summarisation ---------------------------------------------------------

    def _maybe_summarize(self, next_prompt: str):
        """Compress OLD turns if the history would overflow ``max_tokens``."""
        test = self.model.tokenizer.apply_chat_template(
            self.history + [{"role": "user", "content": next_prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        if len(self.model.tokenizer(test)["input_ids"]) <= self.max_tokens:
            return
        self._compress_history()

    def _force_summarize(self) -> str | None:
        """Always summarise the old turns (used by ``/summarize``)."""
        return self._compress_history()

    def _compress_history(self) -> str | None:
        """Summarise everything except anchors and the recent tail.

        Returns the new memory text, or ``None`` if there was nothing to
        compress.
        """
        anchors = self._anchors()
        body = self.history[len(anchors):]
        if len(body) <= _KEEP_RECENT:
            return None
        old, recent = body[:-_KEEP_RECENT], body[-_KEEP_RECENT:]
        # instruct in the user's language so the summary follows suit
        summary_prompt = (
            "请用和对话相同的语言，简洁地总结上面的对话。"
            "保留关键事实、决定和用户的意图。省略寒暄与客套。")
        msgs = old + [{"role": "user", "content": summary_prompt}]
        fmt = self.model.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inp = self.model.tokenizer(fmt, return_tensors="pt").to(self.model.model.device)
        # inp["input_ids"]: (1, old_len)  — old turns as one batch
        with torch.no_grad():
            out = self.model.model.generate(
                **inp, max_new_tokens=min(self.max_tokens // 2, 300),
                temperature=0.3, do_sample=False,
                pad_token_id=self.model.tokenizer.pad_token_id)
        # out: (1, old_len + summary_tokens)  — input + short summary
        raw = self.model.tokenizer.decode(out[0], skip_special_tokens=True)
        summary = raw.rpartition("assistant\n")[-1].strip()
        memory = f"[Memory] {summary}"
        self.history = anchors + [{"role": "system", "content": memory}] + recent
        return memory
