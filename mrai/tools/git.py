from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from ..settings import DEFAULT_DENIED_PATTERNS
from .spawn import safe_run

# Repo-local config is attacker-controlled when you ask about a cloned repo:
# these keys name commands git would run on a plain `status`/`diff`. Each is
# blanked with `-c key=` (command-line config wins over every config file).
_FILTER_KEY = re.compile(r"^filter\.(.+)\.(clean|smudge|process)$")
_ALWAYS_OFF = ("core.fsmonitor=false", "core.hooksPath=/dev/null")
# --no-ext-diff: ignore diff.external / diff.<driver>.command.
# --no-textconv: ignore diff.<driver>.textconv.
_SAFE_DIFF = ("--no-ext-diff", "--no-textconv")


def _glob_regex(pattern: str) -> re.Pattern[str]:
    """`**/` = any leading dirs (or none), `*` / `?` = within one path segment."""
    rx = re.escape(pattern).replace(r"\*\*/", "(?:.*/)?").replace(r"\*\*", ".*")
    return re.compile(rx.replace(r"\*", "[^/]*").replace(r"\?", "[^/]") + r"\Z")


def _is_denied(path: str, patterns: Sequence[str]) -> bool:
    """True if `path` or any parent directory matches a denied glob."""
    p = PurePosixPath(path)
    candidates = [str(q) for q in (p, *p.parents) if q.parts]
    return any(_glob_regex(pat).match(c) for pat in patterns for c in candidates)


def _excludes(patterns: Sequence[str]) -> list[str]:
    """Exclude pathspecs so git itself never prints a denied file's contents.

    `pat/**` as well as `pat`: excluding a directory pattern alone would still
    show the files inside it.
    """
    return [f":(exclude,glob){p}{suffix}" for p in patterns for suffix in ("", "/**")]


class GitTools:
    """Frozen-argv read-only git views (diff/status) exposed as agent tools.

    argv is built from literals; the model can only supply `path` (after `--`),
    a validated ref, or the `staged` boolean. No raw git passthrough, no shell,
    no write flags. Diffs never include files matching `denied_patterns` (the
    same list the filesystem tools refuse), and repo-local config cannot make
    git run a command.
    """

    def __init__(
        self,
        root: Path,
        timeout: float = 10.0,
        denied_patterns: Sequence[str] = DEFAULT_DENIED_PATTERNS,
    ) -> None:
        self._root = root
        self._timeout = timeout
        self._denied = tuple(denied_patterns)
        self._env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "GIT_OPTIONAL_LOCKS": "0",
            # Only the repo's own config is read (and then neutralized below).
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }

    def _neutralizing_config(self) -> list[str]:
        """`-c` overrides that blank every command-bearing key the repo config sets.

        `git config --list` only reads config; it runs nothing.
        """
        names = self._exec("config", "--list", "--name-only", "--includes").splitlines()
        blanked = [f"{n}=" for n in names if _FILTER_KEY.match(n)]
        return [arg for kv in (*_ALWAYS_OFF, *blanked) for arg in ("-c", kv)]

    def _exec(self, *argv: str) -> str:
        proc = safe_run(
            ["git", "--no-pager", *argv], cwd=self._root, timeout=self._timeout, env=self._env
        )
        return (proc.stdout + proc.stderr).rstrip()

    def _run(self, *argv: str) -> str:
        try:
            return self._exec(*self._neutralizing_config(), *argv) or "(no output)"
        except Exception as exc:
            return f"git tool error: {type(exc).__name__}: {exc}"

    def _diff(self, *argv: str, path: str = "") -> str:
        if path and (".." in Path(path).parts or Path(path).is_absolute()):
            return "git tool error: path must be relative and within the repo"
        if path.startswith(":"):
            return "git tool error: pathspec magic (a leading ':') is not allowed"
        if path and _is_denied(path, self._denied):
            return f"git tool error: {path!r} matches a denied pattern (secrets/build output)"
        scope = [path] if path else ["."]
        return self._run("diff", *_SAFE_DIFF, *argv, "--", *scope, *_excludes(self._denied))

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

    def git_diff(self, path: str = "", staged: bool = False) -> str:
        """Unified diff of working tree against HEAD (--cached if staged=True)."""
        return self._diff("--cached" if staged else "HEAD", path=path)

    def git_status(self) -> str:
        """Short status (porcelain), including untracked files."""
        return self._run("status", "--porcelain")

    def git_compare(self, ref: str = "HEAD", path: str = "") -> str:
        """Unified diff of the working tree against another branch/ref (e.g. "origin/main"),
        optionally limited to `path`. Read-only: target is validated against known refs."""
        ref = ref.strip() or "HEAD"
        if ref not in self._known_refs():
            return f"git tool error: unknown ref {ref!r}"
        return self._diff(ref, path=path)
