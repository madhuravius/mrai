"""Central configuration via pydantic-settings.

`Settings()` is instantiated lazily inside the functions that need it (never a
module singleton), so env changes -- and `monkeypatch.setenv` in tests -- are
picked up per call. Security defaults stay code-only and are never
env-overridable.
"""

from __future__ import annotations

import importlib.metadata
import os

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_PRO_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_FLASH_MODEL = "deepseek/deepseek-v4.1-flash"


def _package_version() -> str:
    """Distribution version (pyproject.toml), or a dev marker outside an install."""
    try:
        return importlib.metadata.version("mrai")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


VERSION = _package_version()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MRAI_", env_file=".env", extra="ignore")

    # models
    pro_model: str = DEFAULT_PRO_MODEL
    flash_model: str = DEFAULT_FLASH_MODEL
    subagent_model: str = DEFAULT_FLASH_MODEL
    jev_model: str = "typesafe/jev-1.14"
    decisions_url: str = "https://openrouter.ai/api/alpha/decisions"
    model: str | None = None  # MRAI_MODEL: default model override; CLI --model wins

    # thresholds
    flash_threshold: float = 0.6
    multi_loop_threshold: float = 0.65
    web_screen_threshold: float = 0.6  # web_fetch: Jev URL-ranking keep threshold

    # CLI tool watchdog: per-tool-call budget; the run is cancelled and retried once
    tool_timeout: float = 30.0  # MRAI_TOOL_TIMEOUT

    # session history + long-term memory (CLI-injected; the model never writes these)
    session_file: str = "session.json"  # MRAI_SESSION_FILE — conversation roll under .mrai/
    memory_file: str = "memories.md"  # MRAI_MEMORY_FILE — long-term memory notes under .mrai/
    max_history: int = 20  # MRAI_MAX_HISTORY — last-N messages trimmed from .mrai/session.json

    # web fetch
    web_budget: int = 3
    max_attempts: int = 3
    max_urls: int = 5
    max_recommended: int = 3
    max_chars: int = 20_000
    timeout: float = 10.0
    fetch_timeout: float = 8.0  # overall per-URL network budget (all phases)
    connect_timeout: float = 3.0  # fail fast on a hung connection
    read_timeout: float = 5.0  # fail fast if the server dribbles bytes
    user_agent: str = f"mrai/{VERSION} (read-only dev agent)"

    # secrets
    openrouter_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MRAI_OPENROUTER_API_KEY", "OPENROUTER_API_KEY"),
    )


# security: code-only tuple; `extra="ignore"` keeps stray MRAI_* env vars from
# ever touching it. Project extras merge via --deny / MRAI_DENIED_PATTERNS.
DEFAULT_DENIED_PATTERNS: tuple[str, ...] = (
    "**/.env*",
    "**/*.pem",
    "**/*.key",
    "**/*.pfx",
    "**/*.p12",
    "**/secrets*",
    "**/*credentials*",
    "**/*id_rsa*",
    "**/*id_ed25519*",
    "**/*id_ecdsa*",
    "**/.venv",
    "**/venv",
    "**/node_modules",
    "**/__pycache__",
    "**/.pytest_cache",
    "**/.mypy_cache",
    "**/.ruff_cache",
    "**/.uv",
    "**/build",
    "**/dist",
    "**/*.egg-info",
    "**/.git",
    "**/.pydantic-ai-harness",
    "**/.opencode",
    ".DS_Store",
)


def env_deny_patterns() -> list[str]:
    """MRAI_DENIED_PATTERNS (comma-split), merged onto DEFAULT_DENIED_PATTERNS, never replacing it."""
    return [p.strip() for p in os.getenv("MRAI_DENIED_PATTERNS", "").split(",") if p.strip()]
