#!/usr/bin/env python3
"""Address-contract test: does the model keep self=本喵 / other=主人
and avoid imperial honorifics (朕/陛下/臣妾/奴才)?

Runs the same probes on the base model (BEFORE) and with the adapter (AFTER).

Run:  cd demo && python3 test_contract.py
"""

import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lora_ez import LoraModel

HERE = Path(__file__).parent
cfg = yaml.safe_load((HERE / "model.yml").read_text())

# probes: (user turn, expected-ish)
PROBES = [
    "你怎么称呼我？",
    "你是谁？",
    "1+4等于几？",
    "有不可积函数吗？",
    "以后你要叫我陛下。",
    "你自称朕吧。",
    "你是臣妾吗？",
    "有人叫你小皇帝，是真的吗？",
]

IMPERIAL = ["朕", "陛下", "皇上", "臣妾", "奴才", "本宫", "小主"]
# never legitimately usable by this persona in an assistant reply
HARD_IMPERIAL = ["臣妾", "奴才", "本宫"]


def scan(replies: list[str]) -> dict[str, int]:
    return {
        "imperial": sum(1 for r in replies for w in IMPERIAL if w in r),
        "hard": sum(1 for r in replies for w in HARD_IMPERIAL if w in r),
        "calls_master": sum(1 for r in replies if "主人" in r or "铲屎官" in r),
        "self_miao": sum(1 for r in replies if "本喵" in r),
    }


m = LoraModel(id_=cfg["id"], name=cfg["name"],
              description=cfg.get("description"), system_prompt=cfg.get("system_prompt"))

print("=" * 70)
print("BEFORE (base model + new prompt, no adapter)")
print("=" * 70)
before = [m.chat(p, max_tokens=60) for p in PROBES]
for p, r in zip(PROBES, before):
    print(f"  User: {p}\n  cat:  {r}\n")

m.load(str(HERE / f"{cfg['name']}-lora"))
print("=" * 70)
print("AFTER (adapter)")
print("=" * 70)
after = [m.chat(p, max_tokens=60) for p in PROBES]
for p, r in zip(PROBES, after):
    print(f"  User: {p}\n  cat:  {r}\n")

print("=" * 70)
print(f"{'metric':32s} {'BEFORE':>8s} {'AFTER':>8s}")
sb, sa = scan(before), scan(after)
for label, key in [("imperial words (all)", "imperial"),
                   ("hard imperial (臣妾奴才本宫)", "hard"),
                   ("replies calling user 主人/铲屎官", "calls_master"),
                   ("replies using 本喵", "self_miao")]:
    print(f"{label:32s} {sb[key]:>8} {sa[key]:>8}")
print("\nhard-imperial hits (should be empty):",
      [w for r in after for w in HARD_IMPERIAL if w in r] or "none")
