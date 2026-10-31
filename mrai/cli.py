from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from collections import Counter
from collections.abc import AsyncIterable, Sequence
from contextlib import suppress
from decimal import Decimal
from pathlib import Path
from time import monotonic
from typing import Any, cast

import typer
from markdown_it import MarkdownIt
from pydantic_ai import (
    AgentRunError,
    AgentRunResult,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    RunContext,
    ThinkingPartDelta,
    UnexpectedModelBehavior,
)
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ToolCallPart
from pydantic_ai.usage import RunUsage, UsageLimits
from rich.console import Console, Group
from rich.live import Live as _Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.text import Text

from .agent import Answer, MemoryDeps, build_researcher, build_writers
from .logbook import Logbook, _atomic_write_text, slugify
from .router import Route, resolve_route, select_final_answer
from .settings import UPSTREAM_RETRY_MARKERS, VERSION, Settings, env_deny_patterns
from .webfetch import WebFetchTool

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


class ToolCallTimeout(RuntimeError):
    """Raised by the CLI watchdog when a tool call exceeds the per-call budget."""


# Tools that show an interactive human prompt while they run (e.g. web_fetch's
# approval flow). While one is in flight we suspend the Live refresh and the
# watchdog, so the prompt isn't redrawn-over and a slow human decision doesn't
# trip the tool timeout.
_INTERACTIVE_TOOLS = frozenset({"web_fetch"})

_LIVE_WINDOW = 20  # transcript lines shown in the live overlay


class ToolRunView:
    """Spinner state + watchdog + transcript for the run's in-flight tool calls.

    The Live bottom line renders the transcript (last 20 lines) plus a spinner
    with per-tool elapsed timers. Everything shown inside `Live` lives here —
    nothing is printed to the console while Live is active, so Rich never has
    to fight external writes. The watchdog fires once any tool exceeds
    `timeout`. `slow_overrides` maps tool names to their own (longer) budget —
    used for `delegate_task`, which drives a whole nested agent loop.
    """

    def __init__(self, timeout: float, slow_overrides: dict[str, float] | None = None) -> None:
        self.timeout = timeout
        self.slow_overrides = slow_overrides or {}
        self.starts: dict[str, float] = {}
        self.thinking = False
        self.interactive = False
        self.fired = asyncio.Event()
        self.fired_names: set[str] = set()
        self._spinner = Spinner("dots", text="thinking…")
        self.history: list[Text] = []
        self._thought = ""
        self._shown = 0

    def _budget(self, name: str) -> float:
        return self.slow_overrides.get(name, self.timeout)

    def add_tool_call(self, name: str, args: str) -> None:
        self._flush_thought()
        self.thinking = False
        self.history.append(
            Text.assemble(("↳ ", "cyan"), (name, "cyan"), (f" {args}" if args else "", "dim"))
        )

    def add_tool_result(self, name: str, content: str) -> None:
        self._flush_thought()
        self.history.append(Text(f"  └ {name} → {content}", style="dim"))

    def add_thought(self, delta: str) -> None:
        self.thinking = True
        self._thought += delta

    def _flush_thought(self) -> None:
        if self._thought:
            for line in self._thought.splitlines() or [self._thought]:
                self.history.append(Text(line, style="dim italic"))
            self._thought = ""

    def new_lines(self) -> list[Text]:
        """Transcript lines added since the last read (for plain-terminal reprints)."""
        tail = self.history[self._shown :]
        self._shown = len(self.history)
        return tail

    def snapshot(self) -> Group:
        now = monotonic()
        parts = []
        for name, start in sorted(self.starts.items(), key=lambda kv: kv[1]):
            secs = now - start
            flag = " [TIMEOUT]" if secs >= self._budget(name) else ""
            parts.append(f"{name} {secs:.1f}s{flag}")
        if self.thinking:
            parts.append("thinking…")
        self._spinner.text = " | ".join(parts) or "idle"

        blocks: list = [*self.history[self._shown : self._shown + _LIVE_WINDOW]]
        if not self._shown:
            blocks = [*self.history[-_LIVE_WINDOW:]]
        if self._thought:
            blocks.append(Text(self._thought, style="dim italic"))
        blocks.append(self._spinner)
        return Group(*blocks)

    def timed_out(self) -> list[str]:
        now = monotonic()
        return [name for name, start in self.starts.items() if now - start >= self._budget(name)]

    def fire(self, name: str) -> None:
        self.fired_names.add(name)
        self.fired.set()


def _event_kind(view: ToolRunView, event) -> str | None:
    """Apply one stream event to the view; return its kind for rendering.

    Both the terminal (Live) and headless (plain) branches share this so the
    three event types are interpreted in exactly one place.
    """
    if isinstance(event, FunctionToolCallEvent):
        view.starts[event.part.tool_name] = monotonic()
        view.thinking = False
        view.add_tool_call(event.part.tool_name, _fmt_args(event.part.args))
        if event.part.tool_name in _INTERACTIVE_TOOLS:
            view.interactive = True
        return "call"
    if isinstance(event, FunctionToolResultEvent):
        view.starts.pop(event.part.tool_name, None)
        if event.part.tool_name in _INTERACTIVE_TOOLS:
            view.interactive = False
        view.add_tool_result(event.part.tool_name, _one_line(str(event.part.content)))
        return "result"
    if (
        isinstance(event, PartDeltaEvent)
        and isinstance(event.delta, ThinkingPartDelta)
        and event.delta.content_delta
    ):
        return "thought"
    return None


async def _event_handler(
    _ctx: RunContext, stream: AsyncIterable, view: ToolRunView, live: bool = True
) -> None:
    """Stream thinking text and tool activity to the terminal.

    A single loop consumes events via `_event_kind`; only the *rendering*
    differs: a transient Live bottom line (transcript + spinner) on a real
    terminal, or plain line-by-line printing headless (`live=False`, pipes/CI)
    — each with its own watchdog. `redirect_stdout=False` keeps human-facing
    tool prompts (e.g. web_fetch's approval) on the real terminal; while an
    interactive tool is in flight the refresh loop and watchdog are suspended
    so the prompt isn't redrawn-over.
    """

    async def _print_plain() -> None:
        async for event in stream:
            kind = _event_kind(view, event)
            if kind in ("call", "result"):
                for line in view.new_lines():
                    console.print(line)
            elif kind == "thought":
                console.print(event.delta.content_delta, markup=False, style="dim italic", end="")

    async def _watch() -> None:
        while not view.fired.is_set():
            if not view.interactive:
                for name in view.timed_out():
                    view.fire(name)
                    return
            await asyncio.sleep(0.25)

    if not live or not console.is_terminal or os.getenv("CI"):
        watcher = asyncio.create_task(_watch())
        try:
            await _print_plain()
        finally:
            watcher.cancel()
        return

    try:
        with _Live(
            view.snapshot(),
            console=console,
            screen=False,
            refresh_per_second=8,
            transient=True,
            redirect_stdout=False,
        ) as lc:

            def _suspend_for_prompt() -> None:
                """Flush the thinking, exit the overlay, and print the compiled
                transcript so a human-facing tool prompt lands on a stable
                terminal instead of fighting the live layer."""
                with suppress(Exception):
                    lc.stop()
                for line in view.new_lines():
                    console.print(line)

            async def _tick() -> None:
                while True:
                    if not view.interactive:
                        lc.update(view.snapshot(), refresh=True)
                        for name in view.timed_out():
                            view.fire(name)
                            return
                    await asyncio.sleep(0.25)

            watcher = asyncio.create_task(_tick())
            try:
                async for event in stream:
                    kind = _event_kind(view, event)
                    if kind == "call" and view.interactive:
                        _suspend_for_prompt()
                    elif kind == "result" and not lc.is_started:
                        lc.start()
                    elif kind == "thought":
                        view.add_thought(event.delta.content_delta)
                    if not view.interactive:
                        lc.update(view.snapshot(), refresh=True)
            finally:
                watcher.cancel()
    finally:
        # transient=True cleared the live layer; print whatever was never
        # replayed to the terminal plainly (new_lines is idempotent).
        for line in view.new_lines():
            console.print(line)


async def _run_with_watchdog(
    agent,
    query: str,
    history,
    deps: MemoryDeps,
    view: ToolRunView,
    usage: RunUsage,
    live: bool = True,
    watchdog: bool = True,
    limits: UsageLimits | None = None,
):
    """Race `agent.run` against the tool-call watchdog; cancel + raise on timeout.

    `usage` is mutated in place by `agent.run`, so a timed-out attempt's tokens
    survive into the retry (and the footer) rather than being discarded.
    The event-stream handler is wrapped here so the stream is the only signal
    path (the `handler` callable pydantic-ai invokes takes exactly two args).
    `live=False` routes through the plain/headless renderer (used by the
    parallel run, where per-agent Live overlays would fight each other).
    `watchdog=False` skips the fired-wait race for tool-less writers (the timer
    can never fire, so there is no race to guard).
    `limits` caps model requests via pydantic-ai's UsageLimits.
    """

    async def handler(ctx: RunContext, stream: AsyncIterable) -> None:
        await _event_handler(ctx, stream, view, live=live)

    run_task = asyncio.create_task(
        agent.run(
            query,
            message_history=history,
            deps=deps,
            usage=usage,
            usage_limits=limits,
            event_stream_handler=handler,
        )
    )
    if not watchdog:
        return await run_task
    timeout_task = asyncio.create_task(view.fired.wait())
    done, pending = await asyncio.wait(
        {run_task, timeout_task}, return_when=asyncio.FIRST_COMPLETED
    )
    if run_task not in done:
        run_task.cancel()
        for p in pending:
            p.cancel()
        await asyncio.gather(run_task, *pending, return_exceptions=True)
        raise ToolCallTimeout(", ".join(sorted(view.fired_names)))
    timeout_task.cancel()
    return run_task.result()


# The three "angles" run as parallel agents, each producing ONE candidate
# answer; Jev picks the winner afterwards. Distinct angles yield distinct,
# comparable candidates.
_ANGLES = (
    "code-provenance: trace where things come from and cite exact files/lines",
    "architecture: explain the design, its roles, and the tradeoffs involved",
    "pragmatic: give the shortest directly-actionable answer",
)


def _merge_usage(*usages: RunUsage) -> RunUsage:
    """Sum RunUsage accumulators across parallel agents.

    `cost` is Decimal, so sum it outside the int/float loop.
    """
    merged: dict[str, int | float] = {}  # no Decimal here
    cost = Decimal(0)
    for u in usages:
        if u is None:
            continue
        for field, value in u.__dict__.items():
            if field.startswith("_") or not isinstance(value, (int, float)):
                continue
            merged[field] = merged.get(field, 0) + value
        if isinstance(u.cost, Decimal):
            cost += u.cost
    result = RunUsage(**cast(dict[str, Any], merged))
    if cost > 0:
        result.cost = cost
    return result


def _writer_prompt(query: str, angle: str) -> str:
    """The writer's user message: the question plus its assigned angle."""
    return f"Question: {query}\n\nAngle: {angle}\nWrite ONE final Answer."


async def _run_limited(
    agent,
    query: str,
    history,
    deps: MemoryDeps,
    timeout: float,
    usage: RunUsage,
    *,
    live: bool = True,
    watchdog: bool = True,
    limits: UsageLimits | None = None,
    slow_overrides: dict[str, float] | None = None,
):
    """Run one agent with a UsageLimits guard; on UsageLimitExceeded retry once
    with a doubled request_limit (transient 50-request exhaustion shouldn't end
    a legitimate deep-thinking run), then re-raise."""
    try:
        return await _run_with_watchdog(
            agent,
            query,
            history,
            deps,
            ToolRunView(timeout, slow_overrides),
            usage,
            live=live,
            watchdog=watchdog,
            limits=limits,
        )
    except UsageLimitExceeded:
        console.print("[yellow]hit request budget; retrying with double limit[/yellow]")
        base = limits.request_limit if limits and limits.request_limit else 100
        doubled = UsageLimits(request_limit=2 * base)
        return await _run_with_watchdog(
            agent,
            query,
            history,
            deps,
            ToolRunView(timeout, slow_overrides),
            usage,
            live=live,
            watchdog=watchdog,
            limits=doubled,
        )


async def _run_pipeline(
    researcher,
    writers: Sequence,
    query: str,
    history,
    deps: MemoryDeps,
    settings: Settings,
) -> tuple[AgentRunResult, list[AgentRunResult[Answer] | BaseException], RunUsage]:
    """Two-phase run.

    Phase 1 — the one visible, tool-owning researcher streams a single thought
    process and does every git/read/web step exactly once, returning an
    evidence transcript. Phase 2 — three tool-less writers fan out in parallel,
    each synthesizing ONE Answer from that transcript under its angle. Nothing
    is re-collected. Returns (research, writer_results, merged_usage).
    """
    timeout = settings.tool_timeout
    limits = UsageLimits(request_limit=settings.max_requests)
    slow = {"delegate_task": settings.subagent_timeout}
    r_usage = RunUsage()
    research = await _run_limited(
        researcher,
        query,
        history,
        deps,
        timeout,
        r_usage,
        live=True,
        limits=limits,
        slow_overrides=slow,
    )
    evidence = research.all_messages()  # the full tool workflow, verbatim

    w_usages = [RunUsage() for _ in writers]
    results = await asyncio.gather(
        *[
            _run_limited(
                w,
                _writer_prompt(query, angle),
                list(evidence),
                deps,
                timeout,
                u,
                live=False,  # no Live collision while fanning out
                watchdog=False,  # tool-less writers: no timer race
                limits=limits,
                slow_overrides=slow,
            )
            for w, angle, u in zip(writers, _ANGLES, w_usages, strict=False)
        ],
        return_exceptions=True,
    )
    return research, list(results), _merge_usage(r_usage, *w_usages)


def _write_render(topic: str, markdown: str) -> Path:
    md = MarkdownIt("gfm-like", {"linkify": True})
    body = md.render(markdown)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/styles/github.min.css">
<style>
body {{ font-family: system-ui; max-width: 72ch; margin: 2rem auto; line-height: 1.6; }}
pre {{ position: relative; }}
pre code.hljs {{ padding: 1rem; border-radius: 6px; overflow-x: auto; }}
.copy-btn {{
  position: absolute; top: 0.5rem; right: 0.5rem; padding: 0.25rem 0.5rem;
  font-size: 0.75rem; background: #e1e4e8; border: 1px solid #d1d5da;
  border-radius: 4px; cursor: pointer; opacity: 0; transition: opacity 0.15s;
}}
pre:hover .copy-btn, .copy-btn:focus {{ opacity: 1; }}
</style></head><body>{body}
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/highlight.min.js"></script>
<script>hljs.highlightAll();</script>
<script>
(function(){{
  document.querySelectorAll('pre').forEach(function(pre){{
    var btn = document.createElement('button');
    btn.className = 'copy-btn'; btn.textContent = 'Copy';
    btn.onclick = function(){{
      var code = pre.querySelector('code');
      navigator.clipboard.writeText((code||pre).textContent).then(function(){{
        btn.textContent = 'Copied!';
        setTimeout(function(){{ btn.textContent = 'Copy'; }}, 2000);
      }});
    }};
    pre.appendChild(btn);
  }});
}})();
</script></body></html>"""
    path = Path(".mrai/renders") / f"{slugify(topic)}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path=path, text=html)
    return path


def _enter_sandbox(no_web: bool) -> None:
    if sys.platform != "linux":
        console.print("[yellow]--sandbox skipped: Linux-only[/yellow]")
        return
    if os.getenv("MRAI_SANDBOXED"):
        return
    if shutil.which("bwrap") is None:
        raise typer.Exit(1)
    cwd, home = os.getcwd(), os.path.expanduser("~")
    cmd = [
        "bwrap",
        "--ro-bind",
        "/",
        "/",
        "--bind",
        cwd,
        cwd,
        "--tmpfs",
        "/tmp",
        "--unshare-pid",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--die-with-parent",
        "--new-session",
        "--setenv",
        "MRAI_SANDBOXED",
        "1",
    ]
    mrai_dir = os.getenv("MRAI_DIR") or f"{home}/.mrai"
    cmd += [
        "--bind",
        mrai_dir,
        mrai_dir,  # real data + writable
        "--tmpfs",
        f"{home}/.cache",  # just needs writable
    ]
    if no_web:
        cmd += ["--unshare-net"]
    argv = [a for a in sys.argv if a != "--sandbox"]  # strip to avoid re-entry
    os.execvpe("bwrap", cmd + argv, {**os.environ, "MRAI_SANDBOXED": "1"})


@app.command()
def main(
    ctx: typer.Context,
    query: str | None = typer.Argument(None, help="the prompt to act on (read-only)"),
    version: bool = typer.Option(False, "--version", help="print the version and exit"),
    flash: bool = typer.Option(False, "--flash", "-f", help="force the flash-tier model"),
    pro: bool = typer.Option(False, "--pro", "-p", help="force the pro-tier model"),
    subagents: bool = typer.Option(False, "--subagents", "-s", help="enable the explorer subagent"),
    sandbox: bool = typer.Option(
        True, "--sandbox", "-b", help="enable sandboxing and run natively"
    ),
    no_subagents: bool = typer.Option(
        False, "--no-subagents", help="disable the explorer subagent"
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="OpenRouter model id"),
    mrai_dir: Path = typer.Option(Path.home() / ".mrai", "--dir", help="markdown log directory"),
    no_write: bool = typer.Option(False, "--no-write", help="print answer only, do not log"),
    no_web: bool = typer.Option(False, "--no-web", help="disable the web_fetch tool"),
    max_web_fetches: int | None = typer.Option(
        None, "--max-web-fetches", min=0, help="approved fetches per run (default: MRAI_WEB_BUDGET)"
    ),
    deny: list[str] | None = typer.Option(
        None, "--deny", "-d", help="glob pattern to deny (repeatable)"
    ),
    no_memory: bool = typer.Option(
        False,
        "--no-memory",
        help="do not inject long-term memory (logbook index + MRAI_MEMORY_FILE)",
    ),
) -> None:
    """Answer a dev query against the repo, logging to .mrai/*.md. Jev routes model + subagents; web_fetch is Jev-screened and human-approved."""
    if version:
        typer.echo(f"mrai {VERSION}")
        raise typer.Exit(0)
    if sandbox:
        _enter_sandbox(no_web=no_web)
    if query is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)
    subs: bool | None = None
    if no_subagents:
        subs = False
    elif subagents:
        subs = True
    route = resolve_route(query, flash, pro, model, subs)
    if rows := route.jev_summary():
        console.print("[routing] [yellow]Jev scores[/yellow]:")
        for label, score, blurb, chosen in rows:
            mark = " → [green]chosen[/green]" if chosen else ""
            console.print(f"  [bold]{label:<16}[/bold] {score:.2f}  [dim]{blurb}[/dim]{mark}")
    else:
        console.print(f"[routing] {route.describe()}")
    if route.jev_error:
        console.print(f"[dim]warning: Jev unavailable ({route.jev_error}); routing fell back[/dim]")
    mrai_dir = Path(os.getenv("MRAI_DIR") or mrai_dir)
    settings = Settings()
    session_path = mrai_dir / settings.session_file
    memory_file = mrai_dir / settings.memory_file

    history: list[ModelMessage] = _load_history(session_path, settings.max_history)
    console.print(
        f"[dim]history: continuing {session_path} ({len(history)} messages)[/dim]"
        if history
        else f"[dim]history: no session yet at {session_path}; starting fresh[/dim]"
    )
    known = ""
    if not no_memory:
        known = Logbook(mrai_dir).known_context(memory_file)
        console.print(
            f"[dim]memory: known context from {memory_file} + logbook[/dim]"
            if known
            else "[dim]memory: nothing to inject (no logbook topics / memory file)[/dim]"
        )
    deps = MemoryDeps(known_context=known)

    common = {
        "deny": tuple((deny or []) + env_deny_patterns()),
        "settings": settings,
        "memory": known or None,
    }
    try:
        # Phase 1 owns the tools; phase 2 writers are tool-less by construction.
        researcher, webfetch = build_researcher(
            Path.cwd(),
            route.model,
            web_fetch=not no_web,
            **{
                **common,
                "subagents": route.subagents,
                "web_fetch_budget": max_web_fetches,
            },
        )
        writers = build_writers(Path.cwd(), route.model, _ANGLES, **common)
    except Exception as exc:
        console.print(f"[red]model setup failed:[/red] {exc}")
        raise typer.Exit(1) from None

    try:
        research, results, usage = asyncio.run(
            _run_pipeline(researcher, writers, query, history or None, deps, settings)
        )
    except ToolCallTimeout as exc:
        console.print(
            f"[yellow]tool call exceeded {settings.tool_timeout:.0f}s ({exc}); retrying run once[/yellow]"
        )
        try:
            research, results, usage = asyncio.run(
                _run_pipeline(researcher, writers, query, history or None, deps, settings)
            )
        except (ToolCallTimeout, UnexpectedModelBehavior, AgentRunError) as exc2:
            _log_failure(mrai_dir, query, f"retry failed: {_one_line(str(exc2))}", no_write)
            console.print("[yellow]retry also failed; giving up[/yellow]")
            raise typer.Exit(1) from None
    except UnexpectedModelBehavior as exc:
        _log_failure(
            mrai_dir,
            query,
            f"model gave up on a failing tool call: {_one_line(str(exc))}",
            no_write,
        )
        console.print(
            f"[yellow]model gave up on a failing tool call:[/yellow] {_one_line(str(exc))}"
        )
        raise typer.Exit(1) from None
    except AgentRunError as exc:
        if _is_retryable(exc):
            console.print(
                f"[yellow]upstream HTTP 504/529/429 ({_one_line(str(exc))}); retrying run once[/yellow]"
            )
            try:
                research, results, usage = asyncio.run(
                    _run_pipeline(researcher, writers, query, history or None, deps, settings)
                )
            except AgentRunError as exc2:
                _log_failure(
                    mrai_dir,
                    query,
                    f"upstream error, retry failed: {_one_line(str(exc2))}",
                    no_write,
                )
                console.print(f"[red]retry failed:[/red] {_one_line(str(exc2))}")
                raise typer.Exit(1) from None
        else:
            _log_failure(
                mrai_dir,
                query,
                f"run failed: {_one_line(str(exc))}",
                no_write,
            )
            console.print(f"[red]run failed:[/red] {_one_line(str(exc))}")
            raise typer.Exit(1) from None

    ok = [r for r in results if not isinstance(r, BaseException)]
    for i, r in enumerate(results, 1):
        if isinstance(r, BaseException):
            console.print(f"[yellow]candidate {i} failed:[/yellow] {_one_line(str(r))}")
    if not ok:
        console.print("[red]all candidates failed; nothing to select[/red]")
        raise typer.Exit(1) from None

    candidates: list[Answer] = [r.output for r in ok]
    idx = pick_cost = None
    scores: list[float] = []
    pick_err = "fewer than two candidates; nothing to choose between"
    if len(candidates) >= 2:
        idx, scores, pick_cost, pick_err = select_final_answer(
            query,
            [(c.topic, c.title, c.markdown) for c in candidates],
            settings.openrouter_api_key,
            settings,
        )
    if idx is None:
        console.print(
            f"[dim]warning: Jev final selection unavailable ({pick_err}); using first candidate[/dim]"
        )
        idx = 0
    else:
        console.print("[routing] [yellow]Jev scores[/yellow] (final answers):")
        for i, c in enumerate(candidates):
            mark = " → [green]chosen[/green]" if i == idx else ""
            console.print(
                f"  [bold]{i + 1} · {c.title}[/bold] {scores[i]:.2f}  [dim]{' '.join(c.markdown.split())[:96]}[/dim]{mark}"
            )
    ans = candidates[idx]
    if not no_write:
        _save_history(session_path, research.all_messages(), settings.max_history)
    console.print()
    console.print(Markdown(ans.markdown))
    console.print()
    _print_footer(
        research,
        route,
        webfetch,
        mrai_dir,
        ans.topic,
        ans.title,
        ans.markdown,
        query,
        no_write,
        memory_file,
        usage,
        pick_cost,
    )


def _trim_history(messages: Sequence[ModelMessage], max_history: int) -> list[ModelMessage]:
    """Copy + bound the roll to the last-N messages (never mutates the live list)."""
    messages = list(messages)
    if max_history > 0 and len(messages) > max_history:
        return messages[-max_history:]
    return messages


def _load_history(path: Path, max_history: int) -> list[ModelMessage]:
    """Read .mrai/session.json -> last-N ModelMessages; [] on missing/invalid data."""
    if not path.is_file():
        return []
    try:
        history = ModelMessagesTypeAdapter.validate_json(path.read_text())
    except Exception:
        return []
    return _trim_history(history, max_history)


def _save_history(path: Path, messages: Sequence[ModelMessage], max_history: int) -> None:
    """Persist a snapshot of the run's messages, trimmed to last-N."""
    _atomic_write_text(
        path, ModelMessagesTypeAdapter.dump_json(_trim_history(messages, max_history)).decode()
    )


def _fmt_args(args) -> str:
    try:
        s = json.dumps(args, default=str, ensure_ascii=False) if args else ""
    except Exception:
        s = str(args)
    return s if len(s) <= 200 else s[:197] + "..."


def _one_line(text: str) -> str:
    line = " ".join(text.split())
    return line if len(line) <= 200 else line[:197] + "..."


def count_tool_calls(result) -> Counter[str]:
    """Count model tool calls from the run's message history."""
    return Counter(
        part.tool_name
        for msg in result.all_messages()
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    )


def _print_footer(
    result,
    route: Route,
    webfetch: WebFetchTool | None,
    mrai_dir: Path,
    topic: str,
    title: str,
    markdown: str,
    query: str,
    no_write: bool,
    memory_file: Path,
    usage: RunUsage,
    pick_cost: float | None,
) -> None:
    _log_topic(mrai_dir, topic, title, markdown, query, no_write, memory_file)
    _print_stats(result, route, webfetch, usage, pick_cost)


def _log_topic(
    mrai_dir: Path,
    topic: str,
    title: str,
    markdown: str,
    query: str,
    no_write: bool,
    memory_file: Path,
) -> None:
    bits = [f"[bold]topic[/bold] {topic}"]
    if not no_write:
        lb = Logbook(mrai_dir)
        path, created = lb.append(topic, title, query, markdown)
        lb.remember(memory_file, topic, title)
        bits.append(f"[bold green]{'new' if created else 'appended'}[/bold green] {path}")
        html_path = _write_render(topic, markdown)
        console.print(f"[dim]render:[/dim] file://{html_path.resolve()}")
    console.print(" · ".join(bits))


def _log_failure(
    mrai_dir: Path,
    query: str,
    reason: str,
    no_write: bool,
) -> None:
    """Record a terminal failure as a stub log entry (no durable memory note)."""
    if no_write:
        return
    lb = Logbook(mrai_dir)
    path, _ = lb.append(
        query,
        "Run failed",
        query,
        f"> This run failed and produced no answer.\n\n**Reason:** {reason}\n",
    )
    console.print(f"[dim]logged failure: {path}[/dim]")


def _is_retryable(exc: BaseException) -> bool:
    """OpenRouter 504 upstream-idle-timeout / 529 overloaded / 429 rate-limit."""
    text = str(exc)
    if any(m in text for m in UPSTREAM_RETRY_MARKERS):
        return True
    for wrapped in (getattr(exc, "__cause__", None), getattr(exc, "__context__", None)):
        if wrapped is not None and any(m in str(wrapped) for m in UPSTREAM_RETRY_MARKERS):
            return True
    return False


def _print_stats(
    result, route: Route, webfetch: WebFetchTool | None, usage: RunUsage, pick_cost: float | None
) -> None:
    meta = []
    if usage.cost is None:
        meta.append("cost unavailable")
    else:
        meta.append(f"cost ${usage.cost:.6f}")
    jev_total = (route.jev_cost or 0.0) + (pick_cost or 0.0)
    if jev_total:
        meta.append(f"jev ${jev_total:.6f}")
    if webfetch is not None and webfetch.screen_cost:
        meta.append(f"screen ${webfetch.screen_cost:.6f}")
    if (requests := getattr(usage, "requests", None)) is not None:
        meta.append(f"{requests} requests")
    if meta:
        console.print("[dim]" + " · ".join(meta) + "[/dim]")

    if counts := count_tool_calls(result):
        detail = ", ".join(f"{name}: {n}" for name, n in counts.most_common())
        console.print(f"[dim]tool calls ({sum(counts.values())}):[/dim] {detail}")

    if webfetch is not None:
        if urls := webfetch.fetched_urls:
            console.print(f"[dim]web fetches ({len(urls)}):[/dim]")
            for u in urls:
                console.print(f"  [dim]•[/dim] {u}")
        else:
            console.print("[dim]web fetches: none[/dim]")


if __name__ == "__main__":
    app()
