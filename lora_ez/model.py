"""Core model wrapper — LoraModel for LoRA fine-tuning and inference.

Use::

    from lora_ez import LoraModel

    m = LoraModel("Qwen/Qwen2.5-0.5B-Instruct")

    # ----- fine-tune -----
    # data is a list of ShareGPT convos: [{"messages": [{"role":"user",...}, ...]}, ...]
    m.enable_lora(r=8, alpha=16)
    m.train(data, epochs=30)
    m.save()                       # -> ./lora-assistant/

    # ----- chat (single-turn) -----
    print(m.chat("你好"))

    # ----- multi-turn with memory -----
    with m.chat_session("./chat.json", auto_save=True) as s:
        s.chat("I'm back")
        s.chat("How are you?")
        s.run()                    # interactive REPL, /exit to quit
"""

from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


class LoraModel:
    """A causal LM with built-in LoRA training & chat, hiding ``from_pretrained``."""

    def __init__(self, model_id: str, name: str = "assistant", device: str = "mps",
                 system_prompt: str | None = None):
        self.model_id = model_id
        self.name = name
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype="auto")
        self.peft_model = None
        self.model = self.base
        self._lora_enabled = False

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.model.parameters())
        return f"LoraModel('{self.model_id}', {n/1e9:.1f}B params, lora={'yes' if self._has_lora() else 'no'})"

    def _has_lora(self) -> bool:
        return self.peft_model is not None

    # -- LoRA ---------------------------------------------------------------

    def enable_lora(self, r: int = 8, alpha: int = 16, dropout: float = 0.1):
        if self.peft_model is None:
            cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                            task_type="CAUSAL_LM")
            self.peft_model = get_peft_model(self.base, cfg)
        self.model = self.peft_model
        self._lora_enabled = True

    def disable_lora(self):
        self.model = self.base
        self._lora_enabled = False

    @property
    def lora_enabled(self):
        return self._lora_enabled

    def print_peft_parameters(self):
        self.peft_model.print_trainable_parameters()

    # -- Chat ----------------------------------------------------------------

    def _format(self, messages, add_gen: bool = True) -> str:
        """Render a conversation to a string in Qwen chat format.

        ``messages`` is a list of ``{"role": ..., "content": ...}``.
        A plain string is wrapped as a single user turn.
        ``add_generation_prompt`` controls whether ``<|im_start|>assistant\\n``
        is appended (inference=True, train=False).
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_gen,
        )

    def chat(self, prompt: str, history: list | None = None, max_tokens: int = 80,
             system_prompt: str | None = None) -> str:
        """Send a prompt (with optional conversation history) and return the reply.

        ``history`` is a list of ``{"role": ..., "content": ...}`` dicts from
        previous turns.  The model sees the full context and can refer back to it.
        ``system_prompt`` overrides ``self.system_prompt`` for this single turn.
        """
        self.model.eval()
        sp = system_prompt if system_prompt is not None else self.system_prompt
        messages = list(history or [])
        if sp and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": sp}] + messages
        messages.append({"role": "user", "content": prompt})
        fmt = self._format(messages)
        inp = self.tokenizer(fmt, return_tensors="pt").to(self.model.device)
        # inp["input_ids"]:  (1, seq_len)   — batch of 1, padded token IDs
        # inp["attention_mask"]: (1, seq_len) — 1=real token, 0=padding
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_tokens,
                                      temperature=0.7, do_sample=True,
                                      pad_token_id=self.tokenizer.pad_token_id)
        # out: (1, seq_len + new_tokens)  — input prefix + generated reply
        return self.tokenizer.decode(out[0], skip_special_tokens=True).rpartition("assistant\n")[-1].strip()

    # -- Chat session (context manager) ------------------------------------

    def chat_session(self, save_path: str | None = None, auto_save: bool = False,
                     max_tokens: int = 2000, system_prompt: str | None = None):
        """Return a ``ChatSession`` context manager for multi-turn conversations.

        Usage::

            with model.chat_session("./history.json", auto_save=True) as s:
                s.chat("Hello")
                s.chat("What do you think?")
                s.history   # all turns so far

        When the history grows beyond ``max_tokens`` tokens it is automatically
        summarised by the model and replaced with a compact system message.
        ``system_prompt`` overrides ``self.system_prompt`` for the session.
        """
        from .session import _ChatSession
        return _ChatSession(self, save_path, auto_save, max_tokens,
                           system_prompt=system_prompt if system_prompt is not None else self.system_prompt)

    # -- Training -----------------------------------------------------------

    def train(self, conversations: list[dict], output: bool = False, **kwargs):
        """Fine-tune with LoRA on ShareGPT-format conversations.

        Pipeline: render each convo via chat template -> tokenize with pad+trunc ->
        mask padding positions in labels -> Trainer.
        Auto-enables LoRA if not already active.

        Args:
            output: if set, save training logs/checkpoints to ``./lora-output-{name}``.
                    False (default) produces no output files.
        """
        if not self.lora_enabled:
            self.enable_lora()
        self.model.config.use_cache = False

        texts = [self._format(self._render(c), add_gen=False) for c in conversations]
        tok = self.tokenizer(texts, truncation=True, padding="max_length", max_length=256)
        dataset = [{"input_ids": inds, "attention_mask": ms,
               "labels": [-100 if m == 0 else i for i, m in zip(inds, ms)]}
              for inds, ms in zip(tok["input_ids"], tok["attention_mask"])]

        output_dir = f"./lora-output-{self.name}" if output else "./temp_output"
        defaults = {"epochs": 30, "lr": 3e-4, "per_device_train_batch_size": 4, "logging_steps": 5}
        merged = defaults | kwargs
        args = TrainingArguments(
            output_dir=output_dir, **merged,
            save_strategy="no" if output is None else "epoch",
            report_to="none",
        )
        Trainer(model=self.model, args=args, train_dataset=dataset, processing_class=self.tokenizer).train()

        if not output:
            import shutil
            shutil.rmtree("./temp_output", ignore_errors=True)

        self.model.config.use_cache = True

    def _render(self, conv: dict) -> str:
        return self.tokenizer.apply_chat_template(
            conv["messages"], tokenize=False, add_generation_prompt=False)

    # -- Persistence --------------------------------------------------------

    def save(self, path: str | None = None):
        path = f"lora-{self.name}" if path is None else path
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"The adapter is saved in `{path}`.")

    def load(self, path: str | None = None):
        path = f"lora-{self.name}" if path is None else path
        self.peft_model = PeftModel.from_pretrained(self.base, path)
        self.enable_lora()
