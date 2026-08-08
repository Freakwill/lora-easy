#!/usr/bin/env python3
"""Test: Agent.from_yaml() and a quick chat."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lora_ez import Agent

a = Agent.from_yaml("agent.yml")
print(a)
print(a.chat("who are you?"))
