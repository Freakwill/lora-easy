#!/usr/bin/env python3
"""Demo: train a model from a YAML config (model.yml)."""

# make the library importable when running from the demo/ folder
# import sys
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from pathlib import Path

import yaml

from lora_ez import LoraModel

HERE = Path(__file__).parent
cfg = yaml.safe_load((HERE / "model.yml").read_text())

model_id = cfg["model_id"]
name = cfg.get("name", "assistant")
data_path = HERE / cfg.get("data_path", "cat-chat.json")
save_path = HERE / cfg.get("save_path", f"{name}-lora")
output = cfg.get("output", False)
test_prompts = cfg.get("test_prompts", [])

# -- train ----------------------------

data = json.loads(data_path.read_text())

m = LoraModel(model_id=model_id, name=name)

print("\n=== BEFORE fine-tuning ===")
for p in test_prompts:
    print(f"  input:  {p}")
    print(f"  output: {m.chat(p)}\n")

m.train(data, epochs=30, output=output)
if output:
    m.save(save_path)

print("=== AFTER fine-tuning ===")
for p in test_prompts:
    print(f"  input:  {p}")
    print(f"  output: {m.chat(p)}\n")
