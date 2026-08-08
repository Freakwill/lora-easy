"""Decorator — auto-load files (json / yaml) as the first positional argument."""

import json
from pathlib import Path


def from_file(fn):
    """Decorate ``fn(data, **kwargs)`` so it can be called with a file path.

    ``fn``'s first positional argument is replaced by the parsed content
    of the file given by the caller.  ``.json`` → ``json.load()``,
    ``.yml`` / ``.yaml`` → ``yaml.safe_load()``, otherwise raw text.

    Example::

        @from_file
        def train(data, epochs=30):
            # data = parsed contents of the JSON / YAML file
            ...

        train("data/cat_chat.json", epochs=15)
    """

    def _fn(path, **kwargs):
        p = Path(path)
        suffix = p.suffix.lower()
        content = p.read_text()
        if suffix in (".yml", ".yaml"):
            import yaml
            return fn(yaml.safe_load(content), **kwargs)
        if suffix == ".json":
            return fn(json.loads(content), **kwargs)
        if suffix in (".txt", ".md"):
            return fn(content, **kwargs)
        return fn(content.splitlines(), **kwargs)
    return _fn


def method(deco):
    """Adapt a decorator to also work on instance methods.

    When decorating a method the wrapper automatically skips ``self``
    (the first positional argument), so the inner decorator only sees
    the file path and keyword arguments.

    Example::

        @method
        @from_file
        def train(self, data, epochs=30):
            ...                              # ^ self is skipped by @method

        agent.train("cat_chat.json", epochs=15)  # works
        train("cat_chat.json", epochs=15)        # also works (plain function)
    """

    def _wrap(fn):
        wrapped = deco(fn)

        def _call(*args, **kwargs):
            if args and not isinstance(args[0], (str, Path)):
                return wrapped(*args[1:], **kwargs)
            return wrapped(*args, **kwargs)
        return _call
    return _wrap
