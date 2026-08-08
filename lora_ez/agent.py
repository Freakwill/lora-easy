"""Agent — wraps a LoraModel with web search and file search capabilities.

Use::

    from lora_ez import LoraModel, Agent

    m = LoraModel("Qwen/Qwen2.5-0.5B-Instruct")
    a = Agent(m, description="you are data analyst",
              web_enabled=True, file_enabled=True)

    a.chat("what version of Python?")    # auto-injects tool context
    a.web_fetch("https://example.com")   # fetch a URL
    a.file_read("README.md")             # read local file
    a.disable_web()                       # turn tools off
"""

import fnmatch
import re
from pathlib import Path


class Agent:
    """A conversational agent with web search and local file search.

    Parameters
    ----------
    model : LoraModel
        A pre-loaded model.  ``chat()`` delegates to it.
    name : str or None
        Display name.  Defaults to ``model.name``.
    description : str or None
        The agent's persona — forwarded as ``system_prompt`` on every turn.
        Defaults to ``model.system_prompt``.
    """

    def __init__(
        self,
        model,
        *,
        name: str | None = None,
        description: str | None = None,
        # web-search knobs (off by default)
        web_enabled: bool = False,
        web_allowlist: list[str] | None = None,
        web_blocklist: list[str] | None = None,
        # file-search knobs (off by default)
        file_enabled: bool = False,
        file_dirs: list[str] | None = None,
        file_include: list[str] | None = None,
        file_exclude: list[str] | None = None,
    ):
        self.model = model
        self.name = name or model.name
        self.description = description or model.system_prompt

        self.web_enabled = web_enabled
        self.web_allowlist = web_allowlist or []
        self.web_blocklist = web_blocklist or []
        self.file_enabled = file_enabled
        self.file_dirs = file_dirs or ["."]
        self.file_include = file_include or ["*"]
        self.file_exclude = file_exclude or [
            ".git", "__pycache__", "*.pyc", ".DS_Store", ".*",
        ]

    def __repr__(self) -> str:
        return f"Agent(name='{self.name}', web={self.web_enabled}, files={self.file_enabled})"

    @classmethod
    def from_yaml(cls, path: str):
        """Create an Agent from a YAML config file.

        Delegates to ``LoraModel.from_yaml()`` for the model;
        additional keys control web/file search.
        """
        import yaml
        from .model import LoraModel

        cfg = yaml.safe_load(Path(path).read_text())
        model = LoraModel.from_yaml(path)
        return cls(
            model,
            name=cfg.get("name", model.name),
            description=cfg.get("description", model.system_prompt),
            web_enabled=cfg.get("web_enabled", False),
            web_allowlist=cfg.get("web_allowlist"),
            web_blocklist=cfg.get("web_blocklist"),
            file_enabled=cfg.get("file_enabled", False),
            file_dirs=cfg.get("file_dirs"),
            file_include=cfg.get("file_include"),
            file_exclude=cfg.get("file_exclude"),
        )

    # -- tool toggles --------------------------------------------------------

    def enable_web(self):
        self.web_enabled = True

    def disable_web(self):
        self.web_enabled = False

    def enable_files(self):
        self.file_enabled = True

    def disable_files(self):
        self.file_enabled = False

    # -- chat (injects tool context) -----------------------------------------

    def chat(self, prompt: str, history=None, max_tokens: int = 80,
             system_prompt: str | None = None, **kwargs) -> str:
        context = self._gather_context(prompt)
        if context:
            prompt = (
                f"{prompt}\n\n"
                f"[Relevant context from tools:\n{context}\n]"
            )
        return self.model.chat(
            prompt, history=history, max_tokens=max_tokens,
            system_prompt=system_prompt or self.description, **kwargs,
        )

    # -- context gathering ---------------------------------------------------

    def _gather_context(self, prompt: str) -> str:
        parts = []
        if self.web_enabled:
            parts.append(self._web_search(prompt))
        if self.file_enabled:
            parts.append(self._file_search(prompt))
        return "\n".join(p for p in parts if p)

    # -- web helpers ---------------------------------------------------------

    def _web_search(self, query: str) -> str:
        try:
            from firecrawl import FirecrawlApp
            app = FirecrawlApp()
            data = app.search(query)
            if callable(getattr(data, "data", None)):
                items = list(data.data())[:5]
            elif isinstance(data, dict):
                items = data.get("data", [])[:5]
            else:
                items = getattr(data, "data", [])[:5] if hasattr(data, "data") else []
            if not items:
                return ""
            lines = ["Web results:"]
            for r in items:
                url = r.get("url", "")
                if not self._url_allowed(url):
                    continue
                title = r.get("title", "")
                desc = r.get("description", "")[:200]
                lines.append(f"  - {title}: {desc}  ({url})")
            return "\n".join(lines) if len(lines) > 1 else ""
        except ImportError:
            pass
        except Exception:
            pass
        return ""

    def web_fetch(self, url: str, timeout: int = 15) -> str:
        if not self._url_allowed(url):
            return f"[blocked: {url}]"
        try:
            import requests
            resp = requests.get(url, timeout=timeout,
                                headers={"User-Agent": "lora-easy-agent/1.0"})
            resp.raise_for_status()
            body = re.sub(r"<[^>]+>", " ", resp.text)
            body = re.sub(r"\s+", " ", body).strip()
            return body[:4000]
        except ImportError:
            return "[requests not installed]"
        except Exception as e:
            return f"[fetch error: {e}]"

    def _url_allowed(self, url: str) -> bool:
        if self.web_blocklist and any(
            fnmatch.fnmatch(url, pat) or pat in url for pat in self.web_blocklist
        ):
            return False
        if self.web_allowlist and not any(
            fnmatch.fnmatch(url, pat) or pat in url for pat in self.web_allowlist
        ):
            return False
        return True

    # -- file helpers --------------------------------------------------------

    def _file_search(self, query: str) -> str:
        terms = query.lower().split()
        results: list[str] = []
        for d in self.file_dirs:
            base = Path(d).resolve()
            if not base.is_dir():
                continue
            for pat in self.file_include:
                for f in base.rglob(pat):
                    if not f.is_file():
                        continue
                    if any(f.match(ex.lstrip("/")) for ex in self.file_exclude):
                        continue
                    rel = f.relative_to(base) if f.is_relative_to(base) else f
                    name = str(rel).lower()
                    if not terms or any(t in name for t in terms):
                        results.append(str(rel))
        if not results:
            return ""
        return "Files found:\n" + "\n".join(f"  - {r}" for r in results[:20])

    def file_read(self, path: str, limit: int = 100) -> str:
        p = Path(path)
        if not any((Path(d).resolve() / p).is_file() for d in self.file_dirs):
            return f"[access denied: {path}]"
        try:
            return "\n".join(p.read_text(errors="replace").splitlines()[:limit])
        except Exception as e:
            return f"[read error: {e}]"

    # -- chat session --------------------------------------------------------

    def chat_session(self, *args, **kwargs):
        from .session import _ChatSession
        return _ChatSession(
            self.model, *args,
            system_prompt=kwargs.pop("system_prompt", self.description),
            **kwargs,
        )
