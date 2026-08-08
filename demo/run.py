#!/usr/bin/env python3
"""Demo: train a model to talk like a cat, compare before/after."""

# make the library importable when running from the demo/ folder
# import sys
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from pathlib import Path

from lora_ez import LoraModel


data_path = Path(__file__).parent / "cat-chat.json"

test_prompts = [
    "你觉得今天的晚饭吃什么好？",
    "你为什么总是半夜跑酷？",
    "过来让我抱一下。"
]

model_id = "Qwen/Qwen2.5-0.5B-Instruct"


# -- train ----------------------------

data = json.loads(data_path.read_text())

m = LoraModel(model_id=model_id, name='cat')

print("\n=== BEFORE fine-tuning ===")
for p in TEST_PROMPTS:
    print(f"  input:  {p}")
    print(f"  output: {m.chat(p)}\n")

m.enable_lora()
m.train(data, epochs=30)
if save:
    m.save()

print("\n=== AFTER fine-tuning ===")
for p in TEST_PROMPTS:
    print(f"  input:  {p}")
    print(f"  output: {m.chat(p)}\n")


