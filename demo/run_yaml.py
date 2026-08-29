#!/usr/bin/env python3
"""Demo: train a model from a YAML config (model.yml)."""

import json
from pathlib import Path

import yaml

from lora_ez import LoraModel

HERE = Path(__file__).parent
CONFIG = HERE / "model.yml"

cfg = yaml.safe_load(CONFIG.read_text())

name = cfg.get("name", "assistant")
data_path = HERE / cfg.get("data_path", "cat-chat.json")
test_prompts = cfg.get("test_prompts", [])

# -- train ----------------------------

data = json.loads(data_path.read_text())

m = LoraModel.from_yaml(CONFIG)

print("\n=== BEFORE fine-tuning ===")
for p in test_prompts:
    print(f"  input:  {p}")
    print(f"  output: {m.chat(p)}\n")

save_checkpoints = cfg.get("save_checkpoints", False)
m.train(data, epochs=30, save_checkpoints=save_checkpoints)
# m.save()

print("=== AFTER fine-tuning ===")
for p in test_prompts:
    print(f"  input:  {p}")
    print(f"  output: {m.chat(p)}\n")
