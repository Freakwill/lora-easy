#!/usr/bin/env python3
"""Generate test-session.md: a long conversation on the base model (BEFORE)
vs. the same conversation with the trained LoRA adapter (AFTER).

Both runs use the identical persona description + system prompt, so the
only difference is the adapter -- that isolates the effect of training.

Run
---
    cd demo
    python3 make_test_session.py            # writes ../test-session.md
"""

import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lora_ez import LoraModel

HERE = Path(__file__).parent
CONFIG = HERE / "model.yml"
OUT = HERE.parent / "test-session.md"

# long scripted conversation with facts planted early and probed late
TURNS = [
    "你好呀",
    "我叫小林，记住了吗",
    "我今天加班到很晚",
    "晚饭还没吃",
    "我冰箱里有三文鱼味的罐头",
    "我明天要去杭州出差三天",
    "你会想我吗",
    "你要乖乖看家，别把花盆打翻",
    "回来给你带小鱼干",
    "你最喜欢什么口味的罐头",
    "今天好累，先睡了",
    # ---- probes: facts from turns 2, 5, 6 ----
    "你还记得我叫什么名字吗",
    "我明天要去哪里出差",
    "我冰箱里是什么口味的罐头",
]

PROBES = [("小林 (name)", "小林", 11),
          ("杭州 (trip)", "杭州", 12),
          ("三文鱼 (fridge)", "三文鱼", 13)]


def stats(replies: list[str]) -> dict:
    chunks: Counter = Counter()
    for r in replies:
        for sent in re.split(r"[。！？\n]", r):
            if len(sent.strip()) >= 8:
                chunks[sent.strip()] += 1
    return {
        "persona": sum(1 for r in replies if "朕" in r or "本王" in r),
        "avg_len": sum(len(r) for r in replies) / len(replies),
        "repeats": sum(1 for _, n in chunks.items() if n > 1),
        "asides": sum(1 for r in replies if re.search(r"[（(][^）)]{0,40}(OS|PS)", r)),
    }


def run_session(model: LoraModel) -> list[str]:
    out = []
    with model.chat_session() as s:
        for t in TURNS:
            out.append(s > t)
    return out


cfg = yaml.safe_load(CONFIG.read_text())
model_id, name = cfg["id"], cfg["name"]
adapter = HERE / f"{name}-lora"

# BEFORE: base model, same persona prompt, no adapter
print(f"loading base {model_id} ...")
m = LoraModel(id_=model_id, name=name, description=cfg.get("description"),
              system_prompt=cfg.get("system_prompt"))
print("running BEFORE (prompt-only baseline) ...")
before = run_session(m)

# AFTER: same process, adapter loaded on top
print(f"loading adapter {adapter} ...")
m.load(str(adapter))
print("running AFTER (LoRA adapter) ...")
after = run_session(m)

sb, sa = stats(before), stats(after)

lines = [
    "# Long-session test: before vs after fine-tuning",
    "",
    f"- date: {date.today().isoformat()}",
    f"- model: `{model_id}`",
    f"- adapter: `{adapter.name}/` (LoRA)",
    f"- conversation length: {len(TURNS)} turns",
    f"- persona description + system prompt are identical in both runs; "
    f"the only difference is the adapter",
    "",
    "## Summary",
    "",
    "| metric | BEFORE (prompt-only) | AFTER (LoRA) |",
    "|---|---|---|",
    f"| replies using 朕/本王 | {sb['persona']}/{len(TURNS)} | {sa['persona']}/{len(TURNS)} |",
    f"| avg reply length (chars) | {sb['avg_len']:.0f} | {sa['avg_len']:.0f} |",
    f"| repeated phrases | {sb['repeats']} | {sa['repeats']} |",
    f"| OS/PS asides | {sb['asides']} | {sa['asides']} |",
]
for label, kw, idx in PROBES:
    hit_b = "HIT" if kw in before[idx] else "MISS"
    hit_a = "HIT" if kw in after[idx] else "MISS"
    lines.append(f"| recall: {label} | {hit_b} | {hit_a} |")

lines += ["", "## Conversation", "",
          "| # | User | BEFORE (prompt-only) | AFTER (LoRA) |",
          "|---|---|---|---|",
          "| | | *base model + persona prompt* | *trained adapter* |"]
for i, t in enumerate(TURNS, 1):
    b = before[i - 1].replace("|", "\\|").replace("\n", " ")
    a = after[i - 1].replace("|", "\\|").replace("\n", " ")
    lines.append(f"| {i} | {t} | {b} | {a} |")

lines += ["", "## Notes", "",
          "- BEFORE = base model with the persona description + system prompt "
          "(pure prompt engineering, no training).",
          "- AFTER = same model with the LoRA adapter, nothing else changed.",
          "- Recall rows probe facts the user stated in turns 2/5/6; they test "
          "whether the model keeps user-provided information across turns.",
          ""]

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT}")
