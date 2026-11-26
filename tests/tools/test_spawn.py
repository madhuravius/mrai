import pytest
from mrai.tools.spawn import ALLOWED_COMMANDS, SandboxError, safe_run


def test_allows_git():
    assert safe_run(["git", "version"]).returncode == 0


@pytest.mark.parametrize("cmd", [["sh"], ["curl", "x"], ["ssh", "x"], ["chmod", "777", "/x"]])
def test_blocks_dangerous(cmd):
    with pytest.raises(SandboxError):
        safe_run(cmd)


def test_blocks_shell():
    with pytest.raises(SandboxError, match="shell=True"):
        safe_run(["echo"], shell=True)


def test_blocks_path():
    with pytest.raises(SandboxError, match="bare name"):
        safe_run(["/usr/bin/git"])


def test_blocks_empty():
    with pytest.raises(SandboxError, match="empty"):
        safe_run([])


def test_allowlist_is_closed():
    with pytest.raises(AttributeError):
        ALLOWED_COMMANDS.add("curl")  # type: ignore[attr-defined]
