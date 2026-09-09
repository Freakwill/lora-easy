#!/usr/bin/env python3
"""Run: interactive REPL chat session (cat persona).

Loads ``model.yml`` and auto-loads the trained ``cat-lora`` adapter if
present, then opens an interactive multi-turn chat with slash commands
(``/help``, ``/system-prompt``, ``/summarize``, ``/export``, ``/exit``).

Run
---
    cd demo
    python3 run.py            # train first (optional, recommended)
    python3 run_session.py    # then chat with the cat
"""

from pathlib import Path

from lora_ez import LoraModel

HERE = Path(__file__).parent
CONFIG = HERE / "model.yml"
ADAPTER = HERE / "cat-lora"

# load config (id/name/description/system_prompt) -> model
m = LoraModel.from_yaml(str(CONFIG))
print(f"model ready: {m}")

# restore the trained persona if present
if ADAPTER.exists():
    m.load(str(ADAPTER))
    print(f"adapter loaded from {ADAPTER}")

# custom slash command for this demo
from lora_ez import command

@command("/mood")
def _mood(session, *args):
    """Ask the cat how it is feeling right now."""
    return "本王今天心情尚可。想讨好朕？去开个罐头。"

# interactive multi-turn chat
with m.chat_session() as s:
    print(f"\n💬 {m.name} is here. Type /exit to quit.\n{'='*50}")
    s.run()
