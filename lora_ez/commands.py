"""Slash-command registry for interactive chat sessions.

Commands are registered via ``@register(\"/name\")`` and receive
the ``_ChatSession`` instance as the first argument, followed by a
string of remaining arguments from the prompt::

    @register(\"/greet\")
    def greet(session, args):
        return f\"Hello, {args}!\" if args else \"Hello!\"

Built-in commands: ``/exit``, ``/help``.
"""

_COMMANDS: dict[str, callable] = {}


def register(cmd: str):
    """Decorator — bind *cmd* to the decorated handler.

    The handler signature must be ``(session, args: str) -> str | None``.
    Returning ``None`` means "no output" (the REPL stays silent).
    """
    def decorator(fn):
        _COMMANDS[cmd] = fn
        return fn
    return decorator


def dispatch(session, prompt: str) -> bool:
    """Route a slash-command *prompt* to the right handler.

    Returns ``True`` if the session should exit.
    """
    parts = prompt.strip().split(maxsplit=1)
    cmd = parts[0]
    args = parts[1] if len(parts) > 1 else ""

    if cmd == "/exit":
        return True
    handler = _COMMANDS.get(cmd)
    if handler is None:
        print(f"Unknown command: {cmd}.  Type /help to list available commands.")
    else:
        result = handler(session, args)
        if result is not None:
            print(result)
    return False


# -- built-in ----------------------------------------------------------------

@register("/help")
def _help(session, args):
    """List every registered slash command."""
    names = ", ".join(sorted(_COMMANDS))
    return f"Available commands: /exit, {names}\nAll commands accept a first argument (a session object) and an optional arguments string."
