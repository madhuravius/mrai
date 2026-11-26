from __future__ import annotations

import os
import re
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path


def slugify(text: str) -> str:
    """kebab-case slug, ASCII-only."""
    text = text.encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return (text or "untitled")[:80].rstrip("-")


# Where `append` starts an entry: its `## title`, the optional stamp, then `**Q:**`.
# Plain `## ` lines inside answers are the model's own subheadings, not entries.
_ENTRY_START = re.compile(r"^## [^\n]*\n\n(?:<!-- mrai ts=[^\n]*-->\n\n)?\*\*Q:\*\* ", re.M)
_STAMP = re.compile(r"^<!-- mrai ts=(\S+) cwd=(.*?) -->\n?", re.M)
WORKSPACE_FILE = "workspace"
STATS_MARKER = "<!-- mrai stats -->"  # run stats follow; not part of the answer
MARKER_LINE = re.compile(r"^<!-- mrai [^\n]*-->\n?", re.M)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


class Logbook:
    """Append-only markdown log stored under .mrai/*.md."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory

    def topics(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(p.stem for p in self.dir.glob("*.md"))

    def path_for(self, topic: str) -> Path:
        return self.dir / f"{slugify(topic)}.md"

    def append(
        self, topic: str, title: str, query: str, markdown: str, cwd: Path | None = None
    ) -> tuple[Path, bool]:
        """Write a log entry. Returns (path, created).

        Creates a new file for an unseen topic, otherwise appends a dated
        section at the bottom of the existing topic's file.
        """
        path = self.path_for(topic)
        created = not path.exists()
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        where = " ".join(str(cwd or "").replace("-->", "").split())
        stamp = f"<!-- mrai ts={ts} cwd={where} -->\n\n"
        section = f"## {title}\n\n{stamp}**Q:** {query}\n\n{markdown.strip()}\n\n"
        _atomic_write_text(
            path, f"# {title}\n\n{section}" if created else path.read_text() + section
        )
        return path, created

    def remember(self, memory_file: Path, topic: str, title: str) -> None:
        """Append a compact durable memory line to the long-term memory file.

        Creates the file with a header when it does not already exist. This is
        the CLI's own write (like the logbook) — the model reads this file, it
        never writes it. `--no-write` skips it.
        """
        line = f"- {date.today().isoformat()} · {slugify(topic)} · {' '.join(title.split())}\n"
        if memory_file.is_file():
            text = memory_file.read_text()
            if text and not text.endswith("\n"):
                text += "\n"
            _atomic_write_text(memory_file, text + line)
            return
        _atomic_write_text(
            memory_file,
            "# Long-term memory (CLI-appended; the model reads this, never writes)\n\n" + line,
        )

    def known_context(
        self,
        memory_file: Path,
        *,
        topic_limit: int = 50,
        memory_chars: int = 2000,
    ) -> str:
        """Compact, read-only 'known context' block for prompt injection.

        Summarizes the logbook index plus a hand-maintained memory file
        (MRAI_MEMORY_FILE). The CLI folds this into the agent's system prompt
        (or leaves it empty); the model never writes here — use of this block
        is zero-write on the agent side.
        """
        bits: list[str] = []
        topics = self.topics()
        if topics:
            shown = ", ".join(topics[:topic_limit])
            if len(topics) > topic_limit:
                shown += f" …and {len(topics) - topic_limit} more"
            bits.append(
                f"Previously logged topics for this workspace ({len(topics)}): {shown}. "
                "Reuse an existing topic slug when your answer relates."
            )
        if memory_file.is_file():
            text = memory_file.read_text(encoding="utf-8").strip()
            if len(text) > memory_chars:
                tail = text[-memory_chars:]
                text = "… [older notes truncated]\n" + tail[tail.find("\n") + 1 :]
            bits.append(f"Memory notes (newest last):\n{text}")
        return "\n\n".join(bits)

    def record_workspace(self, cwd: Path) -> None:
        """Remember which repo this state dir belongs to, for legacy entries without a stamp."""
        path = self.dir / WORKSPACE_FILE
        if not path.is_file() or path.read_text(encoding="utf-8").strip() != str(cwd):
            _atomic_write_text(path, f"{cwd}\n")

    def search_index(self, exclude: Path | None = None) -> list[dict[str, str]]:
        """One entry per logged answer across every topic file, newest first.

        Time and workspace come from each section's stamp; unstamped (older)
        sections fall back to the file's mtime and the recorded workspace.
        """
        ws_file = self.dir / WORKSPACE_FILE
        workspace = ws_file.read_text(encoding="utf-8").strip() if ws_file.is_file() else ""
        entries: list[dict[str, str]] = []
        for path in sorted(self.dir.glob("*.md")) if self.dir.is_dir() else []:
            if exclude is not None and path == exclude:
                continue
            text = path.read_text(encoding="utf-8")
            title = path.stem
            if text.startswith("# "):
                first, _, text = text.partition("\n")
                title = first[2:].strip() or title
            mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(
                timespec="seconds"
            )
            starts = [m.start() for m in _ENTRY_START.finditer(text)] or [0]
            for begin, end in zip(starts, [*starts[1:], len(text)], strict=True):
                heading, _, body = text[begin:end].partition("\n")
                heading = heading.removeprefix("## ")
                ts_i, cwd_i = mtime, workspace
                if m := _STAMP.search(body):
                    ts_i, cwd_i = m.group(1), m.group(2) or workspace
                    body = _STAMP.sub("", body)
                body = body.split(STATS_MARKER, 1)[0]
                body = re.sub(r"\n{3,}", "\n\n", body.strip())
                if not (heading.strip() or body):
                    continue
                entries.append(
                    {
                        "topic": path.stem,
                        "title": title,
                        "heading": heading.strip(),
                        "text": body,
                        "ts": ts_i,
                        "cwd": cwd_i,
                        "file": str(path),
                    }
                )
        entries.sort(key=lambda e: e["ts"], reverse=True)
        return entries
