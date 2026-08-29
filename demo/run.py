#!/usr/bin/env python3
"""Demo: train a model to talk like a cat, compare before/after.

What this script does
---------------------
1. Loads a tiny ShareGPT-format dataset (``cat-chat.json``) of cat-persona
   dialogues.
2. Builds a ``LoraModel`` on top of ``Qwen/Qwen2.5-0.5B-Instruct``.
   On first run this downloads the base model weights (≈1 GB) into
   ``~/.cache/huggingface/hub/`` (see ``cache_dir=`` to relocate).
3. Asks a few test prompts BEFORE training to show the base model's
   generic assistant answers.
4. Fine-tunes with LoRA (default ``r=8, alpha=16``, 30 epochs, lr 3e-4).
   Only ~0.1% of parameters are trained, so it finishes in minutes on MPS.
5. Re-asks the same prompts AFTER training to show the cat persona.

Run
---
    cd demo
    python3 run.py

"""

import json
from pathlib import Path

from lora_ez import LoraModel

# -- basic configuration --------------

data_path = Path(__file__).parent / "cat-chat.json"  # ShareGPT-format training data
id_ = "Qwen/Qwen2.5-0.5B-Instruct"                   # base model id
name = "cat"                                         # display name / save-path prefix

# -- train ----------------------------

# data & test prompts
data = json.loads(data_path.read_text())

test_prompts = [
    "你觉得今天的晚饭吃什么好？",
    "你为什么总是半夜跑酷？",
    "过来让我抱一下。"
]

# model
m = LoraModel(id_=id_, name=name)

print("\n=== BEFORE fine-tuning ===")
for p in test_prompts:
    print(f"  User:  {p}")
    print(f"  {name}: {m.chat(p)}\n")

m.train(data, epochs=30)

print("\n=== AFTER fine-tuning ===")
for p in test_prompts:
    print(f"  User:  {p}")
    print(f"  {name}: {m.chat(p)}\n")

# uncomment ``m.save()`` to persist it to ``./cat-lora/``.
# m.save()
