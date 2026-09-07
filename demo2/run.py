#!/usr/bin/env python3
"""Demo2: train a high-EQ girlfriend persona "Ivanka", compare before/after.

What this script does
---------------------
1. Loads ``ivanka-chat.json`` — ShareGPT-format dialogues written for a
   high-EQ, highly-educated, attractive girlfriend persona.
2. Builds a ``LoraModel`` on ``Qwen/Qwen2.5-0.5B-Instruct`` with a fixed
   ``description`` (the persona) and a mutable ``system_prompt``.
   First run downloads ~1 GB of base weights into the HF cache.
3. Asks test prompts BEFORE training (generic assistant answers).
4. LoRA fine-tunes on the persona data (30 epochs, lr 3e-4, ~0.1% params).
5. Re-asks the same prompts AFTER training to show the persona.

Run
---
    cd demo2
    python3 run.py

Uncomment ``m.save()`` at the bottom to persist the adapter to ``./ivanka-lora/``.
"""

import json
from pathlib import Path

import yaml

from lora_ez import LoraModel

HERE = Path(__file__).parent
CONFIG = HERE / "ivanka.yml"

# -- basic configuration --------------

cfg = yaml.safe_load(CONFIG.read_text())
id_ = cfg["id"]
name = cfg["name"]
description = cfg.get("description")
system_prompt = cfg.get("system_prompt")
data_path = HERE / cfg.get("data_path", "ivanka-chat.json")
test_prompts = cfg.get("test_prompts", [])

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

print(f"[3/4] Fine-tuning with LoRA on {len(data)} conversations, {cfg.get('epochs', 30)} epochs ...")
m.train(data, epochs=cfg.get("epochs", 30), max_length=cfg.get("max_length", 256))
print("      training done")

print(f"[4/4] Testing the same prompts AFTER fine-tuning ...\n")
print("=== AFTER fine-tuning ===")
for p in test_prompts:
    print(f"  User:  {p}")
    print(f"  {name}: {m.chat(p)}\n")

# save the trained adapter + persona description to ./ivanka-lora/
m.save()
