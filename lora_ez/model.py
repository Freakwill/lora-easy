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
        s> "I'm back"
        s> "How are you?"
        s.run()                    # interactive REPL, /exit to quit
"""

import torch
from pathlib import Path
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


class LoraModel:
    """A causal LM with built-in LoRA fine-tuning and chat.

    Parameters
    ----------
    id_ : str
        HuggingFace model identifier (e.g. ``"Qwen/Qwen2.5-0.5B-Instruct"``).
    name : str
        Human-readable label, used for default save paths and REPL display.
    device : str
        ``"mps"`` (Apple Silicon), ``"cuda"``, or ``"cpu"``.
    system_prompt : str or None
        Prepend a system instruction to every chat turn.  When ``None`` the
        model uses its built-in default (``"You are Qwen …"``).
    description : str or None
        Immutable identity/role (e.g. ``"you are a sassy house cat"``).
        Rendered as its own system message BEFORE ``system_prompt``.
        Saved with the adapter (``description.txt``) and restored on ``load()``.
        ``None`` (default) means no description is stored or loaded.
    save_path : str
        The path where the adapter is saved.
    cache_dir : str or None
        Custom model download cache.  ``None`` uses the default
        ``~/.cache/huggingface/hub/``.

    Key attributes
    --------------
    base
        The frozen pretrained model (never modified directly).
    peft_model
        The LoRA-wrapped model, or ``None`` before ``enable_lora()``.
    model
        The active model — ``base`` or ``peft_model`` depending on LoRA state.
    """

    def __init__(self, id_: str, name: str = "Assistant", device: str = "mps",
                 system_prompt: str | None = None, save_path: str | None = None,
                 cache_dir: str | None = None, description: str | None = None):
        self.id_ = id_
        self.name = name
        self.system_prompt = system_prompt
        self._description = description
        # tokenizer download (cached at cache_dir or ~/.cache/huggingface/hub/)
        self.tokenizer = AutoTokenizer.from_pretrained(id_, cache_dir=cache_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # model weights download (largest; first run downloads GBs)
        self.base = AutoModelForCausalLM.from_pretrained(id_, device_map=device, torch_dtype="auto", cache_dir=cache_dir)
        self.peft_model = None
        self.model = self.base
        self._lora_enabled = False
        self.save_path = save_path or f"{self:l}-lora"

    @property
    def description(self) -> str | None:
        """Immutable role/identity string (read-only; set via ``__init__`` or ``load()``)."""
        return self._description

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.model.parameters())
        return f"{self:t}: LoraModel('{self:i}', {n/1e9:.1f}B params, lora={'yes' if self._has_lora() else 'no'})"

    def __str__(self):
        return self.name.title()

    def __format__(self, spec=''):
        if spec == '':
            return self.name
        elif spec == 'l':
            return self.name.lower()
        elif spec == 't':
            return self.name
        elif spec == 'i':
            return self.id_
        else:
            raise ValueError('`spec` should be one of [empty] | l | t')

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
        ``description`` (if set) is always prepended as its own system message.
        """
        self.model.eval()
        sp = system_prompt if system_prompt is not None else self.system_prompt
        messages = list(history or [])
        if not messages or messages[0].get("role") != "system":
            # description first (immutable identity), then system_prompt (mutable)
            sys_msgs = []
            if self.description:
                sys_msgs.append({"role": "system", "content": self.description})
            if sp:
                sys_msgs.append({"role": "system", "content": sp})
            messages = sys_msgs + messages
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
                s> "Hello"
                s> "What do you think?"
                s.history   # all turns so far

        When the history grows beyond ``max_tokens`` tokens it is automatically
        summarised by the model and replaced with a compact system message.
        """
        from .session import _ChatSession
        return _ChatSession(self, save_path, auto_save, max_tokens,
                           system_prompt=system_prompt if system_prompt is not None else self.system_prompt)

    # -- Training -----------------------------------------------------------

    def train(self, conversations: list[dict], save_checkpoints: bool = False,
              max_length: int = 256, **kwargs):
        """Fine-tune with LoRA on ShareGPT-format conversations.

        Pipeline: render each convo via chat template -> tokenize with pad+trunc ->
        mask padding positions in labels -> Trainer.
        Auto-enables LoRA if not already active.

        Args:
            save_checkpoints: if set, save a checkpoint per epoch to
                    ``./lora-output-{name}``.  False (default) produces no files.
            max_length: tokenizer truncation/padding length.  Raise it when
                    conversations are long (multi-turn or long replies) —
                    truncation cuts from the END, which would clip the target
                    assistant reply.  Default 256.
        """
        if not self.lora_enabled:
            self.enable_lora()
        self.model.config.use_cache = False

        texts = [self._format(self._render(c), add_gen=False) for c in conversations]
        tok = self.tokenizer(texts, truncation=True, padding="max_length",
                             max_length=max_length)
        dataset = [{"input_ids": inds, "attention_mask": ms,
               "labels": [-100 if m == 0 else i for i, m in zip(inds, ms)]}
              for inds, ms in zip(tok["input_ids"], tok["attention_mask"])]

        # friendly defaults; translate to TrainingArguments names below
        defaults = {"output_dir": f"./lora-output-{self:l}" if save_checkpoints else "./temp_output",
                    "epochs": 5, "lr": 1e-5,
                    "per_device_train_batch_size": 4, "logging_steps": 5}
        kwargs = defaults | kwargs
        args = TrainingArguments(
            num_train_epochs=kwargs.pop("epochs"),
            learning_rate=kwargs.pop("lr"),
            **kwargs,
            save_strategy="no" if not save_checkpoints else "epoch",
            report_to="none",
        )
        Trainer(model=self.model, args=args, train_dataset=dataset, processing_class=self.tokenizer).train()

        if not save_checkpoints:
            import shutil
            shutil.rmtree("./temp_output", ignore_errors=True)

        self.model.config.use_cache = True

    def _render(self, conv: dict) -> str:
        return self.tokenizer.apply_chat_template(
            conv["messages"], tokenize=False, add_generation_prompt=False)

    # -- Persistence --------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str):
        """Create a LoraModel from a YAML config file.

        Expected keys: ``id_``, ``name``, ``system_prompt``,
        ``peft_path`` (optional — auto-loads the adapter).
        """
        import yaml
        from pathlib import Path
        cfg = yaml.safe_load(Path(path).read_text())
        if "id_" in cfg:
            id_ = cfg["id_"]
        elif "id" in cfg:
            id_ = cfg["id"]
        else:
            raise KeyError("Not provide the key `id_` | `id`.")
        m = cls(id_=id_, name=cfg.get("name", "Assistant"),
                 system_prompt=cfg.get("system_prompt"),
                 description=cfg.get("description"))
        if peft := cfg.get("peft_path"):
            m.load(peft)
        return m

    def save(self, path: str | None = None):
        # save the adapter
        path = path or self.save_path
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        if self.description is not None:
            (Path(path) / "description.txt").write_text(self.description)

    def load(self, path: str | None = None):
        # load the adapter from a LOCAL directory
        p = Path(path or self.save_path)
        if not p.is_dir():
            raise FileNotFoundError(
                f"""Adapter directory not found: '{p}'. 
If it lives elsewhere pass the path explicitly, e.g. m.load('/path/to/ivanka-lora').""")
        self.peft_model = PeftModel.from_pretrained(self.base, str(p),
                                                    is_trainable=True)
        # restore description if it was saved alongside the adapter
        desc_file = p / "description.txt"
        if desc_file.exists():
            self._description = desc_file.read_text()
