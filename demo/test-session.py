#!/usr/bin/env python3
"""Test: multi-turn chat session with slash commands."""

from lora_ez import LoraModel, command


# register a custom slash command for testing
@command("/time")
def _time(session, *args):
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


m = LoraModel(id_="Qwen/Qwen2.5-0.5B-Instruct", name="cat")

with m.chat_session("./session-test.json", auto_save=False) as s:
    print("--- programmatic chat (3 turns) ---")
    print("User: 打个招呼吧")
    print("Cat:", s> "打个招呼吧")
    print("User: 你今天开心吗")
    print("Cat:", s> "你今天开心吗")
    print("User: 还记得你刚才说了什么吗")
    print("Cat:", s> "还记得你刚才说了什么吗")
    print()

    print("--- slash command test ---")
    from lora_ez.commands import dispatch
    dispatch(s, "/help")
    dispatch(s, "/time")
    dispatch(s, "/system_prompt you are a talking cat")
    print()

    print("--- chat after /system ---")
    print(s> "你是谁")
    print()

    print("history:", len(s.history), "turns")
