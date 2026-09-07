# lora-easy

<p align="center">
  <img src="assets/banner.svg" alt="lora-easy — LoRA fine-tuning" width="700">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/peft-0.19%2B-orange" alt="peft">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="license">
</p>

A tiny, object-oriented wrapper around 🤗 **PEFT** for LoRA fine-tuning of causal
language models. One `LoraModel` class hides the `from_pretrained` boilerplate,
and `Agent` adds web / file search capabilities on top.

> This project is primarily developed by [Hermes Agent](https://hermes-agent.nousresearch.com),
> an AI coding agent by Nous Research, through iterative human-machine pairing.

## Key concepts

- **LoRA (Low-Rank Adaptation)** — instead of updating all of a model's weights,
  LoRA freezes the base model and trains two small low-rank matrices (`A` and `B`)
  injected into the attention layers. Typically **~0.1% of parameters** are
  trainable, so a checkpoint is a few MB instead of GB.
- **Base vs. adapter** — the large pretrained weights never change. The tiny
  adapter carries the new personality you trained. `enable_lora()` /
  `disable_lora()` just switch which one `self.model` points at, so toggling never
  loses your trained weights.
- **Chat template** — training and inference must format text the same way.
  Training data is rendered with `add_generation_prompt=False`; inference uses
  `True` so the model knows to start generating.
- **Label masking** — `labels` mirror `input_ids`, but padding positions are set to
  `-100` so they are ignored in the loss.
- **System prompt** — set `system_prompt=` on `LoraModel` or `Agent(description=…)`
  to give the model a persona. Change it at runtime with `/system-prompt` in an
  interactive session.
- **Slash commands** — register your own with `@command("/name")` and use them
  during `chat_session().run()`.

## Requirements

```
torch>=2.0
peft>=0.19
transformers>=4.45
```

```bash
pip install lora-easy
```

Runs on CUDA, Apple Silicon (MPS), or CPU.

## Quick start

### LoraModel

```python
from lora_ez import LoraModel

m = LoraModel("Qwen/Qwen2.5-0.5B-Instruct", name="cat",
              system_prompt="you are a sassy house cat")

# ----- fine-tune -----
m.enable_lora(r=8, alpha=16)
m.train(data, epochs=30)
m.save()                   # -> ./lora-cat/

# ----- single-turn chat -----
print(m.chat("hello!"))

# ----- multi-turn with memory -----
with m.chat_session("./chat.json", auto_save=True) as s:
    s > "I'm back"            # shorthand for s.chat("I'm back")
    s > "how are you?"
    s.run()                # interactive REPL, /exit to quit
```

### Agent (web + file search)

```python
from lora_ez import Agent

m = LoraModel("Qwen/Qwen2.5-0.5B-Instruct")
a = Agent(m, description="you are a data analyst",
          web_enabled=True, file_enabled=True)

a.chat("what Python packages are installed?")
a.web_fetch("https://example.com")
a.disable_web()
```

### API

**LoraModel**

| Method | What it does |
|--------|--------------|
| `LoraModel(model_id, name, system_prompt, device)` | Load base model + tokenizer |
| `enable_lora(r, alpha, dropout)` | Attach a LoRA adapter |
| `disable_lora()` | Point back to the frozen base model |
| `train(conversations, **kwargs)` | Fine-tune on ShareGPT-format data |
| `chat(prompt, history, system_prompt)` | Generate a reply |
| `chat_session(save_path, auto_save, system_prompt)` | Multi-turn session context manager |
| `save(path)` / `load(path)` | Persist / restore the adapter |

**Agent** — wraps a `LoraModel` with tools

| Method | What it does |
|--------|--------------|
| `Agent(model, description, web_enabled, …)` | Wrap a model with search tools |
| `chat(prompt)` | Auto-injects web / file context, then delegates to model |
| `enable_web()` / `disable_web()` | Toggle web search |
| `enable_files()` / `disable_files()` | Toggle local file search |
| `web_fetch(url)` | Fetch a URL, respecting allowlists / blocklists |
| `file_read(path)` | Read a file inside allowed directories |
| `chat_session(…)` | Multi-turn session (delegated to the model) |

**Slash commands** — build your own with the `@command` decorator

```python
from lora_ez import command

@command("/greet")
def greet(session, *args):
    return f"Hello, {' '.join(args)}!" if args else "Hello!"
```

Built-in: `/exit`, `/help`, `/system-prompt`.

## Demo

Two demo folders show end-to-end LoRA persona training:

- [`demo/`](demo/) trains `Qwen2.5-0.5B-Instruct` to talk like a sassy
  house cat using [`cat-chat.json`](demo/cat-chat.json) (YAML-driven via
  [`model.yml`](demo/model.yml)).
- [`demo2/`](demo2/) trains `Qwen2.5-1.5B-Instruct` as a high-EQ girlfriend
  persona "Ivanka" ([`ivanka-chat.json`](demo2/ivanka-chat.json)), showing
  multi-turn dialogues and long-reply handling via `max_length`.

```bash
cd demo && python3 run.py            # cat persona
cd demo2 && python3 run.py           # Ivanka persona (train ~1-2 min on MPS)
cd demo2 && python3 test-session.py  # interactive chat with Ivanka
```

## Links

- [🤗 PEFT documentation](https://huggingface.co/docs/peft)
- [LoRA paper (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [Qwen2.5 models](https://huggingface.co/Qwen)

## License

MIT
