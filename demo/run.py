#!/usr/bin/env python3
"""Demo: train a model to talk like a cat, compare before/after.

What this script does
---------------------
1. Loads ``cat-chat.json`` — ShareGPT-format dialogues of a sassy cat persona.
2. Builds a ``LoraModel`` from ``model.yml`` (id/name/description/system_prompt)
   on ``Qwen/Qwen2.5-0.5B-Instruct``.  First run downloads the base weights.
3. Asks test prompts BEFORE training (generic assistant answers).
4. LoRA fine-tunes on the persona data (epochs/max_length from the YAML).
5. Re-asks the same prompts AFTER training to show the cat persona,
   then saves the adapter to ``./cat-lora/``.

Run
---
    cd demo
    python3 run.py
"""

import json
from pathlib import Path

import yaml

from lora_ez import LoraModel

HERE = Path(__file__).parent
CONFIG = HERE / "model.yml"

# -- basic configuration --------------

cfg = yaml.safe_load(CONFIG.read_text())
id_ = cfg["id"]
name = cfg["name"]
description = cfg.get("description")
system_prompt = cfg.get("system_prompt")
data_path = HERE / cfg.get("data_path", "cat-chat.json")
test_prompts = cfg.get("test_prompts", [])
epochs = cfg.get("epochs", 10)
max_length = cfg.get("max_length", 256)
lr = cfg.get("lr", 3e-4)

# -- data ------------------------------

data = json.loads(data_path.read_text())

# -- run -------------------------------

print(f"[1/4] Loading base model {id_} ...")
m = LoraModel(id_=id_, name=name, description=description,
              system_prompt=system_prompt)
print(f"      model ready: {m}")

print(f"[2/4] Testing {len(test_prompts)} prompts BEFORE fine-tuning ...\n")
print("=== BEFORE fine-tuning ===")
for p in test_prompts:
    print(f"  User:  {p}")
    print(f"  {name}: {m.chat(p)}\n")

print(f"[3/4] Fine-tuning with LoRA on {len(data)} conversations, {epochs} epochs ...")
m.train(data, epochs=epochs, lr=lr, max_length=max_length)
print("      training done")

print(f"[4/4] Testing the same prompts AFTER fine-tuning ...\n")
print("=== AFTER fine-tuning ===")
for p in test_prompts:
    print(f"  User:  {p}")
    print(f"  {name}: {m.chat(p)}\n")

# save the trained adapter + persona description to ./cat-lora/
m.save()
print(f"adapter saved to {HERE / 'cat-lora'}")
