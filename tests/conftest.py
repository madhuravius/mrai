import os
import subprocess
from pathlib import Path

import pytest

KEY_ALIASES = (
    "OPENROUTER_API_KEY",
    "MRAI_OPENROUTER_API_KEY",
    "MRAI_API_KEY",
    "OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def keyless_env(monkeypatch):
    for key in KEY_ALIASES:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def isolated_mrai_home(tmp_path, monkeypatch):
    """Never read the developer's ~/.mrai (shared mrai.yml, workspaces) from tests."""
    monkeypatch.setenv("MRAI_HOME", str(tmp_path / "mrai-home"))
    monkeypatch.delenv("MRAI_DIR", raising=False)
    monkeypatch.delenv("MRAI_HOST_DIR", raising=False)


def _no_network(*args, **kwargs):
    raise AssertionError("network call should not happen")


@pytest.fixture
def cli_hermetic(tmp_path, monkeypatch):
    from mrai import router

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MRAI_DIR", str(tmp_path / ".mrai"))
    monkeypatch.setenv("MRAI_SANDBOXED", "1")
    monkeypatch.setattr(router.httpx, "post", _no_network)
    return tmp_path


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"},
    ).stdout


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    tmp_path.mkdir(exist_ok=True)
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "a.txt").write_text("one\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "-c", "user.email=a@b.c", "-c", "user.name=t", "commit", "-q", "-m", "init")
    return tmp_path
