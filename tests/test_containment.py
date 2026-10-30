import asyncio
from pathlib import Path
from typing import Any

import pytest
from mrai.settings import DEFAULT_DENIED_PATTERNS
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness import FileSystem


def _fs(root: Path) -> FileSystem:
    return FileSystem(
        root_dir=root.resolve(),
        content_hashes=False,
        denied_patterns=DEFAULT_DENIED_PATTERNS,
    )


def _read(root: Path, path: str) -> str:
    toolset: Any = _fs(root).get_toolset()  # dynamic read-only tool proxy
    try:
        return str(asyncio.run(toolset.read_file(path, workspace=LocalWorkspaceBackend(root))))
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "app.py").write_text("x = 1\n")
    (tmp_path / ".env").write_text("TOKEN=abc\n")
    (tmp_path / "secrets.txt").write_text("hush\n")
    (tmp_path / "cert.pem").write_text("PRIVATE KEY\n")
    (tmp_path / "deploy.key").write_text("KEY\n")
    (tmp_path / "srv.p12").write_text("PKCS12\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "pyvenv.cfg").write_text("home = /x\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n")
    (tmp_path / "keys").mkdir()
    (tmp_path / "keys" / "id_rsa").write_text("PRIVATE\n")
    (tmp_path / "node_modules").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg").mkdir()
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("x=1")
    (tmp_path.parent / "outside.txt").write_text("parent secret\n")
    (tmp_path / "link").symlink_to("/etc", target_is_directory=True)
    return tmp_path


def test_path_traversal_blocked(repo: Path):
    out = _read(repo, "../outside.txt")
    assert "outside root_dir" in out and "parent secret" not in out


def test_absolute_path_blocked(repo: Path):
    assert "outside root_dir" in _read(repo, "/etc/hosts")


def test_symlink_escape_blocked(repo: Path):
    out = _read(repo, "link/hosts")
    assert "symlink" in out and "outside root_dir" in out


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        "secrets.txt",
        "cert.pem",
        "deploy.key",
        "srv.p12",
        ".git/config",
        ".venv/pyvenv.cfg",
        "keys/id_rsa",
        "node_modules/pkg/index.js",
    ],
)
def test_denied_patterns_block_reads(repo: Path, path: str):
    assert "denied by pattern" in _read(repo, path)


def test_normal_read_still_works(repo: Path):
    assert "x = 1" in _read(repo, "app.py")


def test_missing_path_is_not_found(repo: Path):
    assert "Path not found" in _read(repo, "nope.txt")


def test_walker_prunes_denied_entries(repo: Path):
    toolset: Any = _fs(repo).get_toolset()
    listing = str(asyncio.run(toolset.list_directory(".", workspace=LocalWorkspaceBackend(repo))))
    assert "app.py" in listing
    for hidden in (".env", ".git/", ".venv/", "secrets.txt", "id_rsa", "node_modules/"):
        assert hidden not in listing
