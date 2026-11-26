"""Central configuration via pydantic-settings.

`Settings()` is instantiated lazily inside the functions that need it (never a
module singleton), so env changes -- and `monkeypatch.setenv` in tests -- are
picked up per call. Security defaults stay code-only and are never
env-overridable.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import re
from pathlib import Path

from dotenv import dotenv_values
from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

UPSTREAM_RETRY_MARKERS = ("status_code: 504", "status_code: 529", "status_code: 429")
DEFAULT_PRO_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_FLASH_MODEL = "deepseek/deepseek-v4.1-flash"


def _package_version() -> str:
    """Distribution version (pyproject.toml), or a dev marker outside an install."""
    try:
        return importlib.metadata.version("mrai")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


VERSION = _package_version()


def mrai_home() -> Path:
    return Path(os.getenv("MRAI_HOME") or Path.home() / ".mrai")


def workspace_key(root: Path) -> str:
    """`<basename-slug>-<sha256(path)[:8]>`; mirrored in ./ai, keep in sync."""
    path = str(root)
    name = re.sub(rb"[^a-z0-9]+", b"-", Path(path).name.encode().lower()).decode()
    name = name.strip("-")[:40].rstrip("-") or "root"
    return f"{name}-{hashlib.sha256(path.encode()).hexdigest()[:8]}"


def default_mrai_dir(root: Path) -> Path:
    return mrai_home() / "workspaces" / workspace_key(root)


def _env(name: str) -> str | None:
    return os.getenv(name) or dotenv_values(".env").get(name)


class _ProfileYamlSource(YamlConfigSettingsSource):
    """mrai.yml, with the active entry of `profiles:` laid over the top-level values."""

    def _read_files(self, *args, **kwargs) -> dict:
        data = dict(super()._read_files(*args, **kwargs) or {})
        profiles = data.pop("profiles", None) or {}
        name = os.getenv("MRAI_PROFILE") or data.get("profile")
        if not name:
            return data
        if name not in profiles:
            defined = ", ".join(profiles) or "none"
            raise ValueError(f"unknown mrai profile {name!r} (mrai.yml defines: {defined})")
        return {**data, **profiles[name], "profile": name}


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
    pro_threshold: float = 0.7  # min Jev confidence in "complex" to use pro
    multi_loop_threshold: float = 0.65
    web_screen_threshold: float = 0.6  # web_fetch: Jev URL-ranking keep threshold

    # per-tool-call budget: a slow tool becomes a retry prompt to the model (2x is the hard backstop)
    tool_timeout: float = 30.0  # MRAI_TOOL_TIMEOUT
    subagent_timeout: float = 120.0  # MRAI_SUBAGENT_TIMEOUT
    max_requests: int = 100  # MRAI_MAX_REQUESTS

    # session history + long-term memory (CLI-injected; the model never writes these)
    session_file: str = "session.json"  # MRAI_SESSION_FILE — conversation roll
    memory_file: str = "memories.md"  # MRAI_MEMORY_FILE — long-term memory notes
    routing_log: str = "routing.jsonl"  # MRAI_ROUTING_LOG
    max_history: int = 20  # MRAI_MAX_HISTORY — last-N messages kept in the roll

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
    # empty = any public host
    web_allowed_domains: list[str] = Field(default_factory=list)

    profile: str | None = (
        None  # MRAI_PROFILE / --profile: picks an entry of `profiles:` in mrai.yml
    )
    provider: str = "openrouter"  # "openrouter" | "openai" (compatible gateway) | "anthropic"
    base_url: str | None = None  # gateway URL; for anthropic, overrides the default endpoint
    route_enabled: bool = True  # MRAI_ROUTE_ENABLED: False skips Jev routing
    api_key_env: str | None = None  # env var holding the model key, e.g. ANTHROPIC_API_KEY
    jev_api_key_env: str | None = None  # env var holding Jev's OpenRouter key; default: model key

    # secrets
    openrouter_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "MRAI_API_KEY", "MRAI_OPENROUTER_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"
        ),
    )

    @model_validator(mode="after")
    def _key_from_named_env(self) -> Settings:
        if self.api_key_env:
            self.openrouter_api_key = _env(self.api_key_env)
        return self

    @property
    def jev_api_key(self) -> str | None:
        """Jev (routing, URL screening, picking) is OpenRouter-only, whatever the model provider."""
        return _env(self.jev_api_key_env) if self.jev_api_key_env else self.openrouter_api_key

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _ProfileYamlSource(
                settings_cls, yaml_file=[mrai_home() / "mrai.yml", Path(".mrai/mrai.yml")]
            ),
            file_secret_settings,
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
    "**/.mrai/renders",
    ".DS_Store",
)


def env_deny_patterns() -> list[str]:
    """MRAI_DENIED_PATTERNS (comma-split), merged onto DEFAULT_DENIED_PATTERNS, never replacing it."""
    return [p.strip() for p in os.getenv("MRAI_DENIED_PATTERNS", "").split(",") if p.strip()]
