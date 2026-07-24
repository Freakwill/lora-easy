"""Agent — a LoraModel subclass with description, web search, and file search.

Use::

    from lora_ez import Agent

    a = Agent("Qwen/Qwen2.5-0.5B-Instruct",
              description="you are data analyst",
              web_enabled=True,
              web_allowlist=["*docs.python*"],
              file_enabled=True,
              file_dirs=["."])

    # ----- chat (auto-injects tool context) -----
    print(a.chat("what version of Python is installed?"))

    # ----- fetch a URL -----
    print(a.web_fetch("https://docs.python.org/3/"))

    # ----- search local files -----
    print(a.file_read("README.md"))

    # ----- turn tools off -----
    a.disable_web()
    # pure-chat agent
"""

import fnmatch
import re
from pathlib import Path

from .model import LoraModel


class Agent(LoraModel):
    """A conversational agent with web search and local file search capabilities.

    Parameters
    ----------
    description : str
        The agent's personality / role.  Set automatically as
        ``system_prompt``, so the model sees it at every turn.
    """

    _URL_RE = re.compile(r"^https?://")

    def __init__(
        self,
        model_id: str,
        description: str,
        name: str = "agent",
        device: str = "mps",
        *,
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
        super().__init__(model_id, name=name, device=device,
                         system_prompt=description)
        self.description = description

        self.web_enabled = web_enabled
        self.web_allowlist = web_allowlist or []
        self.web_blocklist = web_blocklist or []
        self.file_enabled = file_enabled
        self.file_dirs = file_dirs or ["."]
        self.file_include = file_include or ["*"]
        self.file_exclude = file_exclude or [".git", "__pycache__", "*.pyc",
                                              ".DS_Store", ".*"]

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
        return super().chat(prompt, history=history, max_tokens=max_tokens,
                           system_prompt=system_prompt, **kwargs)

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
        """Search the web using firecrawl (if available) or requests.

        Only URLs matching ``web_allowlist`` (if set) are fetched;
        URLs matching ``web_blocklist`` are always skipped.
        """
        try:
            from firecrawl import FirecrawlApp
            app = FirecrawlApp()
            data = app.search(query)  # returns object with .data attribute
            if callable(getattr(data, "data", None)):
                items = list(data.data())[:5]  # pydantic model
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
        """Fetch a single URL and return its text content.

        Respects ``web_allowlist`` and ``web_blocklist``.
        """
        if not self._url_allowed(url):
            return f"[blocked: {url}]"
        try:
            import requests
            resp = requests.get(url, timeout=timeout,
                                headers={"User-Agent": "lora-easy-agent/1.0"})
            resp.raise_for_status()
            # crude text extraction: strip tags
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
        """Search local files matching include/exclude patterns.

        Only looks inside ``file_dirs``, honouring ``file_include``
        and ``file_exclude`` glob patterns.
        """
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
        return "Files found:\n" + "\n".join(
            f"  - {r}" for r in results[:20]
        )

    def file_read(self, path: str, limit: int = 100) -> str:
        """Read ``limit`` lines from a file inside ``file_dirs``."""
        p = Path(path)
        if not any((Path(d).resolve() / p).is_file() for d in self.file_dirs):
            return f"[access denied: {path}]"
        try:
            lines = p.read_text(errors="replace").splitlines()
            return "\n".join(lines[:limit])
        except Exception as e:
            return f"[read error: {e}]"

    # -- chat_session override -----------------------------------------------

    def chat_session(self, *args, **kwargs):
        """Same as ``LoraModel.chat_session``, but auto-forwards
        the agent's ``system_prompt`` (the description).
        """
        from .session import _ChatSession
        return _ChatSession(self, *args,
                           system_prompt=kwargs.pop("system_prompt", self.description),
                           **kwargs)
