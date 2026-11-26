from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai_harness import FileSystem, SubAgent, SubAgents

from .prompt import EXPLORER_PROMPT, RESEARCH_PROMPT, WRITER_PROMPT
from .retry import model_client
from .settings import DEFAULT_DENIED_PATTERNS, Settings
from .tools import GitTools
from .webfetch import WebFetchTool

__all__ = [
    "Answer",
    "MemoryDeps",
    "build_researcher",
    "build_writers",
]

OutputT = TypeVar("OutputT")


@dataclass
class MemoryDeps:
    """Per-run deps: injected long-term context. Read-only; owned by the CLI.

    Carries the memory block computed from the per-workspace state dir
    (logbook index + MRAI_MEMORY_FILE). The agent reads it via `@agent.instructions`; it has no
    write path back to disk.
    """

    known_context: str = ""


class Answer(BaseModel):
    topic: str = Field(
        description="kebab-case slug, max 5 words; reuse a logged topic from known context when related"
    )
    title: str = Field(description="short human-readable title for the log entry")
    markdown: str = Field(description="the terse, developer-oriented markdown answer")


def _read_only(root: Path, extra: tuple[str, ...] = ()) -> list:
    """cwd-rooted, zero-write capability set (path traversal/symlink escapes refused)."""
    return [
        LocalWorkspace(root, read_only=True),
        FileSystem(
            root_dir=root.resolve(),
            read_only=True,
            content_hashes=False,
            denied_patterns=[*DEFAULT_DENIED_PATTERNS, *extra],
        ),
    ]


def _model(s: Settings, key: str | None, name: str):
    if s.provider == "anthropic":  # the SDK retries 429/5xx itself
        return AnthropicModel(
            name,
            provider=AnthropicProvider(api_key=key, base_url=s.base_url),
            settings=AnthropicModelSettings(
                anthropic_cache=True,
                anthropic_cache_instructions=True,
                anthropic_cache_tool_definitions=True,
            ),
        )
    client = model_client()  # retries a failed request, not the whole run
    if s.base_url:  # any OpenAI-compatible gateway (bifrost, …)
        return OpenAIChatModel(
            name, provider=OpenAIProvider(api_key=key, base_url=s.base_url, http_client=client)
        )
    return OpenRouterModel(
        name,
        provider=OpenRouterProvider(api_key=key, http_client=client),
        # OpenRouter then reports each request's actual cost (provider_details["cost"])
        settings=OpenRouterModelSettings(openrouter_usage={"include": True}),
    )


def _make(
    root: Path,
    model_name: str,
    *,
    output_type: type[OutputT],
    instructions: str,
    tools_on: bool,
    web_fetch: bool,
    read_only: bool = True,
    subagents: bool = False,
    subagent_model: str | None = None,
    web_fetch_budget: int | None = None,
    deny: tuple[str, ...] = (),
    settings: Settings | None = None,
    memory: str | None = None,
) -> tuple[Agent[MemoryDeps, OutputT], WebFetchTool | None]:
    """Construct a read-only agent (cwd-rooted, zero writes).

    `tools_on` gates the git/web read tools; `web_fetch` gates the (stateful,
    human-approved) WebFetchTool; `read_only` gates the filesystem read
    capabilities. Writers are built with all three off so they never re-read or
    re-fetch — they synthesize purely from the researcher's transcript.
    """
    s = settings or Settings()
    key = s.openrouter_api_key
    caps: list = _read_only(root, deny) if read_only else []
    if subagents:
        caps.append(
            SubAgents(
                agents=[
                    SubAgent(
                        Agent(
                            model=_model(s, key, subagent_model or s.subagent_model),
                            instructions=EXPLORER_PROMPT,
                            capabilities=_read_only(root, deny),
                            retries=3,
                            tool_timeout=s.tool_timeout,
                        ),
                        name="explorer",
                        description="Fast read-only codebase explorer: search, grep, read.",
                    )
                ]
            )
        )
    tools: list = []
    webfetch: WebFetchTool | None = None
    if web_fetch:
        webfetch = WebFetchTool(budget=web_fetch_budget, settings=s)
        tools.append(webfetch.web_fetch)
    if tools_on:
        git_tools = GitTools(root, denied_patterns=[*DEFAULT_DENIED_PATTERNS, *deny])
        tools += [git_tools.git_diff, git_tools.git_status, git_tools.git_compare]
    # A slow tool becomes a retry prompt to the model rather than a cancelled run;
    # delegate_task shares the agent-wide budget, so subagents get theirs.
    # Some Claude models reject the forced tool call that tool-based output relies on.
    native = s.provider == "anthropic" and output_type is not str
    agent = Agent(
        model=_model(s, key, model_name),
        output_type=NativeOutput(output_type) if native else output_type,
        instructions=instructions,
        capabilities=caps,
        tools=tools,
        deps_type=MemoryDeps,
        retries=3,
        tool_timeout=s.subagent_timeout if subagents else s.tool_timeout,
    )
    if memory:

        @agent.instructions
        def _known_context(ctx: RunContext[MemoryDeps]) -> str | None:
            d = ctx.deps
            return d.known_context if d and d.known_context else None

    return agent, webfetch


def build_researcher(
    root: Path,
    model_name: str,
    *,
    subagents: bool = False,
    subagent_model: str | None = None,
    web_fetch: bool = True,
    web_fetch_budget: int | None = None,
    deny: tuple[str, ...] = (),
    settings: Settings | None = None,
    memory: str | None = None,
) -> tuple[Agent[MemoryDeps, str], WebFetchTool | None]:
    """Phase 1: the one agent that owns every tool call (git/read/web) exactly
    once, streaming a single visible thought process, returning an evidence
    transcript (a `str` output). `web_fetch=False` honors `--no-web`."""
    return _make(
        root,
        model_name,
        output_type=str,
        instructions=RESEARCH_PROMPT,
        tools_on=True,
        web_fetch=web_fetch,
        read_only=True,
        subagents=subagents,
        subagent_model=subagent_model,
        web_fetch_budget=web_fetch_budget,
        deny=deny,
        settings=settings,
        memory=memory,
    )


def build_writers(
    root: Path,
    model_name: str,
    angles: Sequence[str],
    *,
    deny: tuple[str, ...] = (),
    settings: Settings | None = None,
    memory: str | None = None,
) -> list[Agent[MemoryDeps, Answer]]:
    """Phase 2: tool-less writers — no web, no git, no filesystem view — each
    synthesizes ONE Answer from the researcher's transcript under its angle.
    `web_fetch_budget` is accepted for signature parity with the shared CLI
    kwargs but is never used (writers fetch nothing)."""
    return [
        _make(
            root,
            model_name,
            output_type=Answer,
            instructions=WRITER_PROMPT,
            tools_on=False,
            web_fetch=False,
            read_only=False,  # no filesystem view at all: synthesize, never read
            subagents=False,
            subagent_model=None,
            deny=deny,
            settings=settings,
            memory=memory,
        )[0]
        for angle in angles
    ]
