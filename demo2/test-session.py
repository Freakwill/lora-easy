#!/usr/bin/env python3
"""Demo2: interactive chat session with Ivanka (needs a trained adapter).

What this script does
---------------------
1. Loads the persona config (``ivanka.yml``) via ``LoraModel.from_yaml``.
2. If ``./ivanka-lora/`` exists it auto-loads the trained adapter
   (``description.txt`` is restored together with the weights).
3. Opens a ``chat_session`` REPL: multi-turn with memory, slash commands
   (``/help``, ``/system-prompt``, ``/summarize``, ``/export``, ``/exit``),
   and a ``__gt__`` shorthand inside sessions.

Run
---
    cd demo2
    python3 run.py                      # train first (optional, recommended)
    python3 test-session.py             # then chat with Ivanka
"""

from pathlib import Path

from lora_ez import LoraModel

HERE = Path(__file__).parent
CONFIG = HERE / "ivanka.yml"
ADAPTER = HERE / "ivanka-lora"

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
    """Ask Ivanka how she is feeling right now."""
    return "我呀，只要你在，心情就是晴朗的～你呢？"

# interactive multi-turn chat
with m.chat_session(save_path=str(HERE / "ivanka-history.json"),
                    auto_save=False) as s:
    print(f"\n💬 伊万卡 is here. Type /exit to quit.\n{'='*50}")
    s.run()
