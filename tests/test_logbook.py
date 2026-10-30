from pathlib import Path

from mrai.logbook import Logbook, slugify


def test_slugify():
    assert slugify("FastAPI WebSockets!") == "fastapi-websockets"
    assert slugify("  UPPER  _  case ") == "upper-case"
    assert slugify("???") == "untitled"


def test_append_creates_new(tmp_path: Path):
    lb = Logbook(tmp_path)
    path, created = lb.append("fastapi-websockets", "WS auth", "how?", "Use `x`.")
    assert created is True
    assert path == tmp_path / "fastapi-websockets.md"
    text = path.read_text()
    assert text.startswith("# WS auth")
    assert "**Q:** how?" in text
    assert "Use `x`." in text


def test_append_updates_existing(tmp_path: Path):
    lb = Logbook(tmp_path)
    path, _ = lb.append("fastapi-websockets", "First", "q1", "a1")
    _, created = lb.append("fastapi-websockets", "Second", "q2", "a2")
    assert created is False
    text = path.read_text()
    assert text.count("# Second") == 1
    assert text.index("a1") < text.index("a2")  # appended, not replaced


def test_topics_lists_existing(tmp_path: Path):
    lb = Logbook(tmp_path)
    assert lb.topics() == []
    lb.append("alpha-beta", "A", "q", "a")
    lb.append("grep", "G", "q", "a")
    assert lb.topics() == ["alpha-beta", "grep"]


def test_remember_creates_with_header_and_appends(tmp_path: Path):
    lb = Logbook(tmp_path)
    mem = tmp_path / "memories.md"
    lb.remember(mem, "greeting", "hello  world")
    first = mem.read_text()
    assert first.startswith("# Long-term memory")
    assert "· greeting · hello world" in first
    assert first.count("# Long-term memory") == 1
    lb.remember(mem, "auth", "auth flow")
    second = mem.read_text()
    assert second.count("# Long-term memory") == 1
    assert "· auth · auth flow" in second
    assert len(second) > len(first)  # appended, not replaced
