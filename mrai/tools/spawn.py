import subprocess
from collections.abc import Sequence

ALLOWED_COMMANDS: frozenset[str] = frozenset({"git"})


class SandboxError(PermissionError):
    pass


def safe_run(
    cmd: Sequence[str],
    *,
    cwd=None,
    timeout=None,
    env=None,
    capture_output=True,
    text=True,
    check=False,
    **kwargs,
):
    if kwargs.get("shell"):
        raise SandboxError("shell=True is forbidden")
    if not cmd:
        raise SandboxError("empty command")
    exe = cmd[0]
    if "/" in exe or "\\" in exe:
        raise SandboxError(f"bare name required, not path: {exe!r}")
    if exe not in ALLOWED_COMMANDS:
        raise SandboxError(f"'{exe}' not allowed")
    return subprocess.run(
        cmd,
        cwd=cwd,
        timeout=timeout,
        env=env,
        capture_output=capture_output,
        text=text,
        check=check,
        **kwargs,
    )
