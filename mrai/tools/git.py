from __future__ import annotations

import os
from pathlib import Path

from .spawn import safe_run


class GitTools:
    """Frozen-argv read-only git views (diff/status) exposed as agent tools.

    argv is built from literals; the model can only supply `path` (after `--`)
    or the `staged` boolean. No raw git passthrough, no shell, no write flags.
    """

    def __init__(self, root: Path, timeout: float = 10.0) -> None:
        self._root = root
        self._timeout = timeout
        self._env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "GIT_OPTIONAL_LOCKS": "0",
        }

    def _known_refs(self) -> set[str]:
        """Known ref names (branches, remotes, tags, HEAD) for validating the compare target."""
        refs = {"HEAD"}
        for r in self._run("for-each-ref", "--format=%(refname)").splitlines():
            r = r.strip()
            if r:
                refs |= {
                    r,
                    r.removeprefix("refs/heads/"),
                    r.removeprefix("refs/remotes/"),
                    r.removeprefix("refs/tags/"),
                }
        return refs

    def _run(self, *argv: str) -> str:
        try:
            proc = safe_run(
                ["git", *argv],
                cwd=self._root,
                timeout=self._timeout,
                env=self._env,
            )
            return (proc.stdout + proc.stderr).rstrip() or "(no output)"
        except Exception as exc:
            return f"git tool error: {type(exc).__name__}: {exc}"

    def git_diff(self, path: str = "", staged: bool = False) -> str:
        """Unified diff of working tree against HEAD (--cached if staged=True)."""
        argv = ["diff"] + (["--cached"] if staged else ["HEAD"])
        if path and (".." in Path(path).parts or Path(path).is_absolute()):
            return "git tool error: path must be relative and within the repo"
        if path:
            argv += ["--", path]
        return self._run(*argv)

    def git_status(self) -> str:
        """Short status (porcelain), including untracked files."""
        return self._run("status", "--porcelain")

    def git_compare(self, ref: str = "HEAD", path: str = "") -> str:
        """Unified diff of the working tree against another branch/ref (e.g. "origin/main"),
        optionally limited to `path`. Read-only: target is validated against known refs."""
        ref = ref.strip() or "HEAD"
        if ref not in self._known_refs():
            return f"git tool error: unknown ref {ref!r}"
        argv = ["diff", ref]
        if path and (".." in Path(path).parts or Path(path).is_absolute()):
            return "git tool error: path must be relative and within the repo"
        if path:
            argv += ["--", path]
        return self._run(*argv)
