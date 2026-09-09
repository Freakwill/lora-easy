#!/usr/bin/env python3
"""Test: multi-turn chat session with slash commands (cat persona).

Loads ``model.yml`` (auto-loads the ``cat-lora`` adapter if present),
then exercises: multi-turn memory, slash commands, runtime system prompt.

Run
---
    cd demo
    python3 test_session.py
"""

from pathlib import Path

from lora_ez import LoraModel, command

HERE = Path(__file__).parent
CONFIG = HERE / "model.yml"
ADAPTER = HERE / "cat-lora"


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
    for p in ["打个招呼吧", "你今天开心吗", "还记得你刚才说了什么吗"]:
        print(f"User: {p}")
        print(f"{m.name}: {s > p}\n")

    print("--- slash command test ---")
    from lora_ez.commands import dispatch
    dispatch(s, "/help")
    dispatch(s, "/time")
    dispatch(s, "/system-prompt you are a sassy house cat")
    print()

    print("--- chat after /system-prompt ---")
    print(f"{m.name}: {s > '你是谁'}\n")

    print(f"history: {len(s.history)} turns")
