from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai_harness import FileSystem, SubAgent, SubAgents

from .prompt import EXPLORER_PROMPT, SYSTEM_PROMPT
from .settings import DEFAULT_DENIED_PATTERNS, Settings
from .tools import GitTools
from .webfetch import WebFetchTool

__all__ = ["Answer", "AnswerCandidates", "MemoryDeps", "build_agent"]


@dataclass
class MemoryDeps:
    """Per-run deps: injected long-term context. Read-only; owned by the CLI.

    Carries the memory block computed from `.mrai/` (logbook index +
    MRAI_MEMORY_FILE). The agent reads it via `@agent.instructions`; it has no
    write path back to disk.
    """

    known_context: str = ""


class Answer(BaseModel):
    topic: str = Field(
        description="kebab-case slug, max 5 words; reuse an existing .mrai topic when related"
    )
    title: str = Field(description="short human-readable title for the log entry")
    markdown: str = Field(description="the terse, developer-oriented markdown answer")


class AnswerCandidates(BaseModel):
    """Two or three complete candidate answers; Jev picks the winner after the run."""

    candidates: list[Answer] = Field(min_length=2, max_length=3)


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
    if s.base_url:  # any OpenAI-compatible gateway (bifrost, …)
        return OpenAIChatModel(name, provider=OpenAIProvider(api_key=key, base_url=s.base_url))
    return OpenRouterModel(name, provider=OpenRouterProvider(api_key=key))


def build_agent(
    root: Path,
    model_name: str,
    subagents: bool = False,
    subagent_model: str | None = None,
    web_fetch: bool = True,
    web_fetch_budget: int | None = None,
    deny: tuple[str, ...] = (),
    settings: Settings | None = None,
    memory: str | None = None,
) -> tuple[Agent[MemoryDeps, AnswerCandidates], WebFetchTool | None]:
    """Read-only agent (cwd-rooted, zero writes) with optional explorer + web_fetch.

    `memory` is an optional long-term context block (logbook index + memory
    file) computed by the CLI. It rides on `deps` through an `@agent.instructions`
    function, which pydantic-ai re-applies on every request outside of message
    history — so it survives `--continue` history trimming and is never written
    by the model.
    """
    s = settings or Settings()
    key = s.openrouter_api_key
    caps = _read_only(root, deny)
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
    git_tools = GitTools(root)
    tools += [git_tools.git_diff, git_tools.git_status]
    agent = Agent(
        model=_model(s, key, model_name),
        output_type=AnswerCandidates,
        instructions=SYSTEM_PROMPT,
        capabilities=caps,
        tools=tools,
        deps_type=MemoryDeps,
        retries=3,
    )
    if memory:

        @agent.instructions
        def _known_context(ctx: RunContext[MemoryDeps]) -> str | None:
            d = ctx.deps
            return d.known_context if d and d.known_context else None

    return agent, webfetch
