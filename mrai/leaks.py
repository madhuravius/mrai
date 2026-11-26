"""Spot credentials in text the model wants to send off this machine.

A prompt-injected file or page can ask the model to "fetch" a URL that carries
repo data in its query string or payload. This catches the cheap cases (your own
API key, well-known token formats, also once URL-decoded); it does not catch
deliberately encoded data, which is why the human still approves every fetch.
"""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Iterable

_PATTERNS: dict[str, re.Pattern[str]] = {
    name: re.compile(rx)
    for name, rx in {
        "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        "AWS access key": r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
        "GitHub token": r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})",
        "API key (sk-…)": r"\bsk-[A-Za-z0-9_-]{20,}",
        "Slack token": r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
        "Google API key": r"\bAIza[0-9A-Za-z_-]{35}",
        "JWT": r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
    }.items()
}


def find_secret(text: str | None, known: Iterable[str | None] = ()) -> str | None:
    """Name the first credential found in `text` (raw or URL-decoded), else None.

    `known` are exact secret values to look for (e.g. the configured API key).
    """
    if not text:
        return None
    variants = {text, urllib.parse.unquote_plus(text)}
    for value in known:
        if value and len(value) >= 8 and any(value in v for v in variants):
            return "your configured API key"
    for name, rx in _PATTERNS.items():
        if any(rx.search(v) for v in variants):
            return name
    return None
