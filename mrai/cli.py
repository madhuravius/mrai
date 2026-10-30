from __future__ import annotations

import asyncio
import json
import os
from collections import Counter
from collections.abc import AsyncIterable, Sequence
from contextlib import suppress
from pathlib import Path
from time import monotonic

import typer
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
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ToolCallPart
from pydantic_ai.usage import RunUsage
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.text import Text

from .agent import AnswerCandidates, MemoryDeps, build_agent
from .logbook import Logbook, _atomic_write_text
from .router import Route, resolve_route, select_final_answer
from .settings import VERSION, Settings, env_deny_patterns
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
    `timeout`.
    """

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self.starts: dict[str, float] = {}
        self.thinking = False
        self.interactive = False
        self.fired = asyncio.Event()
        self.fired_names: set[str] = set()
        self._spinner = Spinner("dots", text="thinking…")
        self.history: list[Text] = []
        self._thought = ""
        self._shown = 0

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
            flag = " [TIMEOUT]" if secs >= self.timeout else ""
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
        return [name for name, start in self.starts.items() if now - start >= self.timeout]

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


async def _event_handler(_ctx: RunContext, stream: AsyncIterable, view: ToolRunView) -> None:
    """Stream thinking text and tool activity to the terminal, live.

    A single loop consumes events via `_event_kind`; only the *rendering*
    differs: a transient Live bottom line (transcript + spinner) on a real
    terminal, or plain line-by-line printing headless (pipes/CI) — each with
    its own watchdog. `redirect_stdout=False` keeps human-facing tool prompts
    (e.g. web_fetch's approval) on the real terminal; while an interactive tool
    is in flight the refresh loop and watchdog are suspended so the prompt
    isn't redrawn-over.
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

    if not console.is_terminal or os.getenv("CI"):
        watcher = asyncio.create_task(_watch())
        try:
            await _print_plain()
        finally:
            watcher.cancel()
        return

    try:
        with Live(
            view.snapshot(),
            console=console,
            screen=False,
            refresh_per_second=8,
            transient=True,
            redirect_stdout=False,
        ) as live:

            def _suspend_for_prompt() -> None:
                """Flush the thinking, exit the overlay, and print the compiled
                transcript so a human-facing tool prompt lands on a stable
                terminal instead of fighting the live layer."""
                with suppress(Exception):
                    live.stop()
                for line in view.new_lines():
                    console.print(line)

            async def _tick() -> None:
                while True:
                    if not view.interactive:
                        live.update(view.snapshot(), refresh=True)
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
                    elif kind == "result" and not live.is_started:
                        live.start()
                    elif kind == "thought":
                        view.add_thought(event.delta.content_delta)
                    if not view.interactive:
                        live.update(view.snapshot(), refresh=True)
            finally:
                watcher.cancel()
    finally:
        # transient=True cleared the live layer; print whatever was never
        # replayed to the terminal plainly (new_lines is idempotent).
        for line in view.new_lines():
            console.print(line)


async def _run_with_watchdog(
    agent, query: str, history, deps: MemoryDeps, view: ToolRunView, usage: RunUsage
):
    """Race `agent.run` against the tool-call watchdog; cancel + raise on timeout.

    `usage` is mutated in place by `agent.run`, so a timed-out attempt's tokens
    survive into the retry (and the footer) rather than being discarded.
    The event-stream handler is wrapped here so the stream is the only signal
    path (the `handler` callable pydantic-ai invokes takes exactly two args).
    """

    async def handler(ctx: RunContext, stream: AsyncIterable) -> None:
        await _event_handler(ctx, stream, view)

    run_task = asyncio.create_task(
        agent.run(
            query,
            message_history=history,
            deps=deps,
            usage=usage,
            event_stream_handler=handler,
        )
    )
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


@app.command()
def main(
    ctx: typer.Context,
    query: str | None = typer.Argument(None, help="the prompt to act on (read-only)"),
    version: bool = typer.Option(False, "--version", help="print the version and exit"),
    flash: bool = typer.Option(False, "--flash", "-f", help="force the flash-tier model"),
    pro: bool = typer.Option(False, "--pro", "-p", help="force the pro-tier model"),
    subagents: bool = typer.Option(False, "--subagents", "-s", help="enable the explorer subagent"),
    no_subagents: bool = typer.Option(
        False, "--no-subagents", help="disable the explorer subagent"
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="OpenRouter model id"),
    mrai_dir: Path = typer.Option(Path(".mrai"), "--dir", help="markdown log directory"),
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
    try:
        agent, webfetch = build_agent(
            Path.cwd(),
            route.model,
            route.subagents,
            web_fetch=not no_web,
            web_fetch_budget=max_web_fetches,
            deny=tuple((deny or []) + env_deny_patterns()),
            settings=settings,
            memory=known or None,
        )
    except Exception as exc:
        console.print(f"[red]model setup failed:[/red] {exc}")
        raise typer.Exit(1) from None

    deps = MemoryDeps(known_context=known)
    usage = RunUsage()

    def _run() -> AgentRunResult[AnswerCandidates]:
        return asyncio.run(
            _run_with_watchdog(
                agent, query, history or None, deps, ToolRunView(settings.tool_timeout), usage
            )
        )

    try:
        result = _run()
    except ToolCallTimeout as exc:
        console.print(
            f"[yellow]tool call exceeded {settings.tool_timeout:.0f}s ({exc}); retrying run once[/yellow]"
        )
        try:
            result = _run()
        except ToolCallTimeout:
            console.print("[yellow]retry also timed out; giving up[/yellow]")
            raise typer.Exit(1) from None
    except UnexpectedModelBehavior as exc:
        console.print(
            f"[yellow]model gave up on a failing tool call:[/yellow] {_one_line(str(exc))}"
        )
        raise typer.Exit(1) from None
    except AgentRunError as exc:
        console.print(f"[red]run failed:[/red] {_one_line(str(exc))}")
        raise typer.Exit(1) from None
    if not no_write:
        _save_history(session_path, result.all_messages(), settings.max_history)
    candidates = result.output.candidates
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
    console.print()
    console.print(Markdown(ans.markdown))
    console.print()
    _print_footer(
        result,
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


# --- session persistence (the CLI is the single write site; the agent never writes) ---


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
    console.print(" · ".join(bits))


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
