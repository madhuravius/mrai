"""Static HTML renders of logged answers: one site for every workspace.

Layout under <mrai home>/site/, deliberately flat (Safari restricts local pages
from loading files outside their own folder):
  app.js, style.css   copied from this package
  index.js            search index over every workspace's topic logs
  <ws>--<slug>.html   one thin page per topic; <ws> is the state dir's name

Each logged answer writes its own page and rewrites index.js; no other page
changes. Pages work over file:// with no server.
"""

from __future__ import annotations

import json
import os
from contextlib import suppress
from functools import cache
from importlib.resources import files
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape
from markdown_it import MarkdownIt
from markupsafe import Markup

from ..logbook import MARKER_LINE, Logbook, _atomic_write_text, slugify, topic_title
from ..settings import mrai_home, workspace_key

HLJS = {
    "base": "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1",
    "light_sri": "sha512-0aPQyyeZrWj9sCA46UlmWgKOP0mUipLQ6OZXu8l4IcAmD2u31EPEy9VcIMvl7SoAaKe8bLXZhYoMaE/in+gcgA==",
    "dark_sri": "sha512-pU8Ny4jS7Uq58y/K4YLD+jF/74zC3R5SDco/Ln143vyLEmGQY7MV8p+Z5q7big/mNimhvQsfQpprGvUa2QzihQ==",
    "js_sri": "sha512-EBLzUL8XLl+va/zAsmXwS7Z2B1F9HUHkZwyS/VKwh3S7T/U0nF4BaU29EP/ZSf6zgiIxYAnKLu6bJ8dqpmX5uw==",
}
ASSETS = ("app.js", "style.css")
INDEX_FILE = "index.js"

_env = Environment(loader=PackageLoader("mrai", "render"), autoescape=select_autoescape(["j2"]))


def _markdown() -> MarkdownIt:
    """Raw HTML in the (possibly steered) model output is escaped, never passed through."""
    return MarkdownIt("gfm-like", {"linkify": True, "html": False})


@cache
def _asset(name: str) -> str:
    return files("mrai.render").joinpath(name).read_text(encoding="utf-8")


def csp() -> str:
    """Local files and the pinned CDN only; no inline script, no eval, no remote images."""
    local = "'self' file:"
    return "; ".join(
        [
            "default-src 'none'",
            f"script-src {local} {HLJS['base']}/highlight.min.js",
            f"style-src {local} {HLJS['base']}/styles/",
            "img-src data:",
            "base-uri 'none'",
            "form-action 'none'",
        ]
    )


def site_dir() -> Path:
    return mrai_home() / "site"


def workspace_dirs(current: Path | None = None) -> list[Path]:
    """Every state dir under <mrai home>/workspaces, plus `current` if it lives elsewhere."""
    root = mrai_home() / "workspaces"
    dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    if (
        current is not None
        and current.is_dir()
        and current.resolve() not in {d.resolve() for d in dirs}
    ):
        dirs.append(current)
    return dirs


def host_path(path: Path, mrai_dir: Path | None = None) -> Path:
    """Map a container path to its host location when running in the ./ai container."""
    if host := os.getenv("MRAI_HOST_HOME"):
        with suppress(ValueError):
            return Path(host) / path.relative_to(mrai_home().resolve())
    if (host := os.getenv("MRAI_HOST_DIR")) and mrai_dir is not None:
        with suppress(ValueError):
            return Path(host) / path.relative_to(mrai_dir.resolve())
    return path


def _sync_assets(site: Path) -> None:
    for name in ASSETS:
        dest = site / name
        text = _asset(name)
        if not dest.is_file() or dest.read_text(encoding="utf-8") != text:
            _atomic_write_text(dest, text)


def page_path(mrai_dir: Path, topic: str) -> Path:
    return site_dir() / f"{mrai_dir.name}--{slugify(topic)}.html"


def write_page(mrai_dir: Path, topic: str, markdown: str, title: str | None = None) -> Path:
    """Write the topic's page for one answer, plus the shared assets it loads.

    `title` is the topic's display title (as the nav and search show it); the
    page's file name always comes from `topic`.
    """
    html = _env.get_template("page.html.j2").render(
        title=title or topic,
        body=Markup(_markdown().render(MARKER_LINE.sub("", markdown))),
        csp=csp(),
        hljs=HLJS,
    )
    path = page_path(mrai_dir, topic)
    _atomic_write_text(path, html)
    _sync_assets(site_dir())
    return path


def index_js(entries: list[dict[str, str]]) -> str:
    # `<` escaped so the data stays inert even if this file is ever inlined
    data = json.dumps(entries, ensure_ascii=False).replace("<", "\\u003c")
    return f"window.MRAI_INDEX = {data};\n"


def write_index(current: Path | None = None, memory_name: str = "memories.md") -> Path:
    """Rewrite site/index.js from every workspace's topic logs, newest first."""
    md = _markdown()
    entries: list[dict[str, str]] = []
    for ws in workspace_dirs(current):
        for e in Logbook(ws).search_index(exclude=ws / memory_name):
            e["ws"] = ws.name
            e["html"] = md.render(e["text"])
            e["file"] = str(host_path(Path(e["file"]).resolve(), current))
            entries.append(e)
    entries.sort(key=lambda e: e["ts"], reverse=True)
    path = site_dir() / INDEX_FILE
    _atomic_write_text(path, index_js(entries))
    return path


def latest_answer(topic_md: str) -> str:
    """The newest answer in a topic log, as `Logbook.append` laid it out."""
    i = topic_md.rfind("**Q:** ")
    if i < 0:
        return topic_md
    _, _, answer = topic_md[i:].partition("\n\n")
    return answer.strip()


def record_workspace(mrai_dir: Path, cwd: Path) -> None:
    """Only when the state dir is provably cwd's (its key hashes the path), not a shared --dir."""
    if Path(os.getenv("MRAI_HOST_DIR") or mrai_dir).name == workspace_key(cwd):
        Logbook(mrai_dir).record_workspace(cwd)


def reindex(current: Path | None = None, memory_name: str = "memories.md") -> list[Path]:
    """Rebuild the whole site from every workspace (e.g. after upgrading mrai)."""
    if current is not None:
        record_workspace(current, Path.cwd())
    pages = []
    for ws in workspace_dirs(current):
        for path in sorted(ws.glob("*.md")):
            if path.name == memory_name:
                continue
            text = path.read_text(encoding="utf-8")
            pages.append(
                write_page(ws, path.stem, latest_answer(text), topic_title(text, path.stem))
            )
    write_index(current, memory_name)
    _remove_stale_pages(set(pages))
    return pages


def _remove_stale_pages(keep: set[Path]) -> None:
    """The site is entirely generated: drop pages no log produces any more,
    including the per-workspace subfolders of the older layout."""
    site = site_dir()
    for page in site.glob("*.html"):
        if page not in keep:
            page.unlink()
    for sub in (d for d in site.iterdir() if d.is_dir()):
        for page in sub.glob("*.html"):
            page.unlink()
        with suppress(OSError):
            sub.rmdir()
