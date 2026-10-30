from __future__ import annotations

import os
import subprocess
from pathlib import Path


class GitTools:
    """Frozen-argv read-only git views (diff/status) exposed as agent tools.

    argv is built from literals; the model can only supply `path` (after `--`)
    or the `staged` boolean. No raw git passthrough, no shell, no write flags.
    """

    def __init__(self, root: Path, timeout: float = 10.0) -> None:
        self._root = root
        self._timeout = timeout
        self._env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")

    def _run(self, *argv: str) -> str:
        try:
            proc = subprocess.run(
                ["git", *argv],
                cwd=self._root,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env=self._env,
            )
            return (proc.stdout + proc.stderr).rstrip() or "(no output)"
        except Exception as exc:
            return f"git tool error: {type(exc).__name__}: {exc}"

    def git_diff(self, path: str = "", staged: bool = False) -> str:
        """Unified diff of working tree against HEAD (--cached if staged=True)."""
        argv = ["diff"] + (["--cached"] if staged else ["HEAD"])
        if path:
            argv += ["--", path]
        return self._run(*argv)

    def git_status(self) -> str:
        """Short status (porcelain), including untracked files."""
        return self._run("status", "--porcelain")
