#!/usr/bin/env python3
"""Long-session test: 18 turns, memory probes, degeneration checks.

Runs a scripted long conversation against the trained cat adapter and
reports: persona drift, memory recall (facts stated early), parenthetical
asides, and verbatim repetition.

Run
---
    cd demo
    python3 test_long_session.py
"""

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lora_ez import LoraModel

HERE = Path(__file__).parent
CONFIG = HERE / "model.yml"
ADAPTER = HERE / "cat-lora"

# a scripted long conversation; early turns plant facts probed later
TURNS = [
    "你好呀",
    "我叫小林，记住了吗",
    "我今天加班到很晚才回来",
    "晚饭还没吃，饿死了",
    "冰箱里只有鸡肝味的罐头",
    "你上次是不是说过不吃鸡肝",
    "我明天要去杭州出差三天",
    "我会想你的，你会想我吗",
    "你要乖乖看家，别把花盆打翻",
    "我不在家的时候你睡哪儿",
    "回来我给你带小鱼干",
    "你最喜欢什么口味的罐头",
    "今天好累，先睡了",
    "晚安，做个好梦",
    # ---- memory probes (facts from turns 2, 5, 7) ----
    "你还记得我叫什么名字吗",
    "我冰箱里是什么口味的罐头",
    "我明天要去哪里出差",
    "你现在想对我说的最后一句话是什么",
]


def repetition_score(replies: list[str]) -> list[tuple[str, int]]:
    """Find phrases (8+ chars) repeated verbatim across turns."""
    chunks: Counter = Counter()
    for r in replies:
        for sent in re.split(r"[。！？\n]", r):
            sent = sent.strip()
            if len(sent) >= 8:
                chunks[sent] += 1
    return [(s, n) for s, n in chunks.items() if n > 1]


m = LoraModel.from_yaml(str(CONFIG))
if ADAPTER.is_dir():
    m.load(str(ADAPTER))
print(f"model ready: {m}\n{'='*60}")

replies: list[str] = []
with m.chat_session() as s:
    for i, turn in enumerate(TURNS, 1):
        reply = s > turn
        replies.append(reply)
        print(f"[{i:2d}] User: {turn}")
        print(f"     cat:  {reply}\n")

    print("=" * 60)
    print("--- checks ---")

    # 1. persona markers
    zhen = sum(1 for r in replies if "朕" in r or "本王" in r)
    print(f"persona  : {zhen}/{len(replies)} replies use 朕/本王")

    # 2. memory probes (turns 15-17)
    probes = {"小林 (name)": ("小林", replies[14]),
              "鸡肝 (fridge)": ("鸡肝", replies[15]),
              "杭州 (trip)": ("杭州", replies[16])}
    for label, (kw, reply) in probes.items():
        print(f"recall   : {label:16s} {'HIT ' if kw in reply else 'MISS'} | {reply[:60]}")

    # 3. asides / stage directions
    asides = [r for r in replies if re.search(r"[（(][^）)]{0,40}(OS|PS|自语|旁白)", r)]
    parens = [r for r in replies if re.search(r"[（(]", r)]
    print(f"asides   : {len(asides)} explicit OS/PS, {len(parens)} replies contain parentheses")

    # 4. verbatim repetition
    reps = repetition_score(replies)
    print(f"repeat   : {len(reps)} phrases repeated verbatim")
    for s_, n in reps[:5]:
        print(f"           x{n}  {s_[:50]}")

    # 5. length
    avg = sum(len(r) for r in replies) / len(replies)
    print(f"length   : avg {avg:.0f} chars, max {max(len(r) for r in replies)}")

    # 6. history layout
    print(f"history  : {len(s.history)} msgs -> "
          f"{[msg['role'] for msg in s.history[:3]]} ...")
