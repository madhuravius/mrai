from __future__ import annotations

import os
import re
import tempfile
from datetime import date
from pathlib import Path


def slugify(text: str) -> str:
    """kebab-case slug, ASCII-only."""
    text = text.encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "untitled"


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

    def append(self, topic: str, title: str, query: str, markdown: str) -> tuple[Path, bool]:
        """Write a log entry. Returns (path, created).

        Creates a new file for an unseen topic, otherwise appends a dated
        section at the bottom of the existing topic's file.
        """
        path = self.path_for(topic)
        created = not path.exists()
        section = f"## {title}\n\n**Q:** {query}\n\n{markdown.strip()}\n\n"
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
                f"Logged topics under .mrai/*.md ({len(topics)}): {shown}. "
                "Reuse an existing topic slug when your answer relates; read files on demand."
            )
        if memory_file.is_file():
            text = memory_file.read_text(encoding="utf-8").strip()
            if len(text) > memory_chars:
                text = text[:memory_chars].rstrip() + "\n… [truncated]"
            bits.append(f"Memory notes ({memory_file}):\n{text}")
        return "\n\n".join(bits)
