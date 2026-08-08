"""Slash-command registry for interactive chat sessions.

Commands are registered via ``@register(\"/name\")`` and receive
the ``_ChatSession`` instance as the first argument, followed by
any remaining words from the prompt as positional arguments::

    @register(\"/greet\")
    def greet(session, *args):
        return f\"Hello, {' '.join(args)}!\" if args else \"Hello!\"

Built-in commands: ``/exit``, ``/help``, ``/system-prompt``, ``/train``.
"""

_COMMANDS: dict[str, callable] = {}


def register(cmd: str):
    """Decorator — bind *cmd* to the decorated handler.

    The handler signature must be ``(session, *args) -> str | None``.
    Returning ``None`` means \"no output\" (the REPL stays silent).
    """
    def decorator(fn):
        _COMMANDS[cmd] = fn
        return fn
    return decorator


def dispatch(session, prompt: str) -> bool:
    """Route a slash-command *prompt* to the right handler.

    Returns ``True`` if the session should exit.
    """
    cmd, *args = prompt.split()

    if cmd == "/exit":
        return True
    handler = _COMMANDS.get(cmd)
    if handler is None:
        print(f"Unknown command: {cmd}.  Type /help to list available commands.")
    else:
        result = handler(session, *args)
        if result is not None:
            print(f"System: {result}")
    return False


# -- built-in ----------------------------------------------------------------

@register("/help")
def _help(session, *args):
    """List every registered slash command."""
    names = ", ".join(sorted(_COMMANDS))
    return f"Available commands: /exit, {names}"


@register("/system-prompt")
def _system_prompt(session, *args):
    """View or change the system prompt.  ``/system_prompt`` alone shows current."""
    if not args:
        current = session.system_prompt or "(default system prompt)"
        return f"Use current system prompt ---  {current}"
    session.system_prompt = " ".join(args)
    return f"System prompt updated."


@register("/train")
def _train(session, *args):
    """Live-train the model with conversations from a JSON file.

    Usage: ``/train <path/to/conversations.json>``
    """
    import json
    from pathlib import Path
    path = " ".join(args)
    if not path:
        return "Usage: /train <path/to/conversations.json>"
    try:
        data = json.loads(Path(path).read_text())
    except Exception as e:
        return f"Failed to load {path}: {e}"
    session.model.train(data)
    return f"Training complete on {len(data)} conversations from {path}."


@register("/summarize")
def _summarize(session, *args):
    """Manually summarise the conversation history."""
    if len(session.history) < 2:
        return "Not enough history to summarize."
    session._force_summarize()
    return "History summarized."


@register("/export")
def _export(session, *args):
    """Save the conversation history as JSON.

    Usage: ``/export [path]`` (default: ``./chat-export.json``).
    """
    import json
    from pathlib import Path
    path = Path(" ".join(args) or "chat-export.json")
    path.write_text(json.dumps(session.history, ensure_ascii=False, indent=2))
    return f"History exported to {path} ({len(session.history)} turns)."
