#!/usr/bin/env python3
"""Test: multi-turn chat session with slash commands (Ivanka persona).

Loads ``ivanka.yml`` (auto-loads the ``ivanka-lora`` adapter if present),
then exercises: multi-turn memory, slash commands, runtime system prompt.

Run
---
    cd demo2
    python3 test_session.py
"""

from pathlib import Path

from lora_ez import LoraModel, command

HERE = Path(__file__).parent
CONFIG = HERE / "ivanka.yml"
ADAPTER = HERE / "ivanka-lora"


# register a custom slash command for testing
@command("/time")
def _time(session, *args):
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# load config (id/name/description/system_prompt) -> model
m = LoraModel.from_yaml(str(CONFIG))
print(f"model ready: {m}")

# restore the trained persona if present
if ADAPTER.exists():
    m.load(str(ADAPTER))
    print(f"adapter loaded from {ADAPTER}")

with m.chat_session() as s:
    print("\n--- programmatic chat (3 turns, memory check) ---")
    for p in ["我今天被老板批评了，好难过", "你觉得我该考研还是直接工作？",
              "你刚才说的那些，我会好好考虑的"]:
        print(f"User: {p}")
        print(f"{m.name}: {s > p}\n")

    print("--- slash command test ---")
    from lora_ez.commands import dispatch
    dispatch(s, "/help")
    dispatch(s, "/time")
    dispatch(s, "/system-prompt 以后用简短温柔的语气回答")
    print()

    print("--- chat after /system-prompt ---")
    print(f"{m.name}: {s > '今天有点累'}\n")

    print(f"history: {len(s.history)} turns")
