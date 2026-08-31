"""Multi-turn chat session with persistent history and auto-summarisation.

Use::

    # Programmatic multi-turn
    with model.chat_session("./chat.json", auto_save=True) as s:
        s.chat("Hello")
        s.chat("How are you?")
        print(s.history)

    # Interactive loop
    with model.chat_session(auto_save=False) as s:
        s.run()                # type messages at the prompt
"""

import json
from pathlib import Path

import torch


class _ChatSession:
    """Multi-turn chat session with persistent history and auto-summarisation.

    Not meant to be instantiated directly — use ``LoraModel.chat_session()``.
    """

    def __init__(self, model, save_path: str | None = None,
                 auto_save: bool = False, max_tokens: int = 2000,
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
            summarisation kicks in (default 2000).
        system_prompt : str or None
            System-level instructions prepended to every turn.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.save_path = Path(save_path) if save_path else None
        self.auto_save = auto_save
        self.max_tokens = max_tokens
        self.history: list[dict] = []  # all conversation turns so far

    def __enter__(self):
        if self.save_path and self.save_path.exists():
            self.history = json.loads(self.save_path.read_text())
        return self

    def __exit__(self, *args):
        if self.auto_save and self.save_path:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_path.write_text(
                json.dumps(self.history, ensure_ascii=False, indent=2))
        self.history.clear()

    def chat(self, prompt: str, **kwargs) -> str:
        """Send a message and get the assistant reply, auto-appending to history.

        Automatically summarises when the tokenised history exceeds ``max_tokens``.
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

    def _maybe_summarize(self, next_prompt: str):
        """Compress history if it would overflow ``max_tokens``, using the model itself."""
        test = self.model.tokenizer.apply_chat_template(
            self.history + [{"role": "user", "content": next_prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        if len(self.model.tokenizer(test)["input_ids"]) <= self.max_tokens:
            return
        self._force_summarize()

    def _force_summarize(self):
        """Always summarise history (used by ``/summarize`` command)."""
        summary_prompt = (
            "Summarise the conversation above concisely. "
            "Keep key facts, decisions, and the user's intent. "
            "Drop small talk.")
        msgs = self.history + [{"role": "user", "content": summary_prompt}]
        fmt = self.model.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inp = self.model.tokenizer(fmt, return_tensors="pt").to(self.model.model.device)
        # inp["input_ids"]: (1, hist_len)  — full history as one batch
        with torch.no_grad():
            out = self.model.model.generate(
                **inp, max_new_tokens=min(self.max_tokens // 2, 300),
                temperature=0.3, do_sample=False,
                pad_token_id=self.model.tokenizer.pad_token_id)
        # out: (1, hist_len + summary_tokens)  — input + short summary
        raw = self.model.tokenizer.decode(out[0], skip_special_tokens=True)
        summary = raw.rpartition("assistant\n")[-1].strip()

        # preserve description (immutable) as its own system message,
        # then merge summary into the system_prompt (mutable)
        self.history = []
        if self.model.description:
            self.history.append({"role": "system", "content": self.model.description})
        if self.system_prompt:
            self.history.append({"role": "system",
                                 "content": f"{self.system_prompt}\n\n[Summary of previous turns: {summary}]"})
        else:
            self.history.append({"role": "system",
                                 "content": f"Previous conversation summary: {summary}"})
