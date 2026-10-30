"""CLI tool-call watchdog: ToolRunView timers + event-stream timeout wiring.

Headless by design: pytest runs with a non-terminal console, so the event
handler exercises the plain (non-Live) path while still tracking watchdog
timestamps.
"""

import asyncio
from time import monotonic
from typing import cast

from mrai.agent import MemoryDeps
from mrai.cli import ToolCallTimeout, ToolRunView, _event_handler, _run_with_watchdog
from pydantic_ai import Agent, FunctionToolCallEvent, FunctionToolResultEvent, RunContext
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage


def _call_event(name: str = "read_file") -> FunctionToolCallEvent:
    return FunctionToolCallEvent(part=ToolCallPart(tool_call_id="1", tool_name=name, args={}))


def _result_event(name: str = "read_file") -> FunctionToolResultEvent:
    return FunctionToolResultEvent(part=ToolReturnPart(tool_name=name, content="ok"))


def test_tool_run_view_tracks_starts_and_timeouts():
    view = ToolRunView(timeout=1.0)
    view.starts["read_file"] = monotonic() - 5  # started 5s ago
    view.starts["web_fetch"] = monotonic()  # just started
    timed = view.timed_out()
    assert "read_file" in timed and "web_fetch" not in timed
    view.fire("read_file")
    assert view.fired_names == {"read_file"}
    assert view.fired.is_set()
    view.snapshot()  # updates the spinner's text for the Live render
    spinner = str(view._spinner.text)
    assert "read_file" in spinner and "TIMEOUT" in spinner and "web_fetch" in spinner


def test_tool_run_view_slow_override_gives_subagent_headroom():
    view = ToolRunView(timeout=1.0, slow_overrides={"delegate_task": 120.0})
    view.starts["read_file"] = monotonic() - 5  # past the 1s budget
    view.starts["delegate_task"] = monotonic() - 10  # 10s: past 1s, before 120s
    assert "read_file" in view.timed_out()
    assert "delegate_task" not in view.timed_out()  # judged against 120s
    # the spinner flag reflects the per-tool budget too
    view.snapshot()
    spinner = str(view._spinner.text)
    assert "delegate_task" in spinner and "read_file" in spinner
    assert spinner.count("[TIMEOUT]") == 1  # only read_file past its budget


def test_tool_run_view_builds_transcript():
    view = ToolRunView(timeout=60.0)
    view.add_thought("let me check the router")
    view.add_tool_call("read_file", '{"path": "router.py"}')
    view.add_thought("hmm")
    view.add_tool_result("read_file", "route = resolve_route(...)")
    text = "\n".join(line.plain for line in view.history)
    assert "let me check the router" in text  # thought flushed before the tool line
    assert "read_file" in text and "route = resolve_route(...)" in text
    # a trailing thought stays in-flight (in _thought, not yet flushed to history)
    view.add_thought("still thinking")
    assert view._thought == "still thinking"
    assert "still thinking" not in "\n".join(line.plain for line in view.history)


def test_event_handler_plain_path_tracks_tool_starts():
    view = ToolRunView(timeout=60.0)

    async def run() -> None:
        events = [_call_event(), _result_event(), _call_event("web_fetch")]

        async def stream():
            for ev in events:
                yield ev

        await _event_handler(cast(RunContext, None), stream(), view)

    asyncio.run(run())
    # read_file returned (popped); web_fetch is still in flight
    assert "read_file" not in view.starts
    assert "web_fetch" in view.starts
    assert view.fired_names == set()


def test_event_handler_marks_interactive_span():
    """web_fetch (a human-prompt tool) brackets the interactive flag in the stream."""
    view = ToolRunView(timeout=60.0)

    async def run(stop_after_call: bool) -> None:
        async def stream():
            yield _call_event("web_fetch")
            if not stop_after_call:
                yield _result_event("web_fetch")

        await _event_handler(cast(RunContext, None), stream(), view)

    asyncio.run(run(stop_after_call=True))  # stream ends while web_fetch is in flight
    assert view.interactive is True
    asyncio.run(run(stop_after_call=False))  # call + result -> span closed
    assert view.interactive is False


def test_headless_watchdog_fires_on_stale_start():
    """The non-terminal path runs its own watchdog, so timeouts apply headless."""

    async def run() -> None:
        agent = Agent(model=TestModel(), instructions="x")
        view = ToolRunView(timeout=0.01)
        view.starts["read_file"] = monotonic() - 5  # already past the budget
        try:
            await _run_with_watchdog(agent, "q", None, MemoryDeps(), view, RunUsage())
        except ToolCallTimeout as exc:
            assert "read_file" in str(exc)
        else:
            raise AssertionError("expected ToolCallTimeout")

    asyncio.run(run())


def test_watchdog_raises_timeout_when_fired():
    async def run() -> None:
        agent = Agent(model=TestModel(), instructions="x")
        view = ToolRunView(timeout=1.0)
        view.fire("web_fetch")  # pre-fire: the watchdog trip is already set
        try:
            await _run_with_watchdog(agent, "q", None, MemoryDeps(), view, RunUsage())
        except ToolCallTimeout as exc:
            assert "web_fetch" in str(exc)
        else:
            raise AssertionError("expected ToolCallTimeout")

    asyncio.run(run())


def test_watchdog_returns_result_when_fast():
    async def run() -> None:
        agent = Agent(model=TestModel(), instructions="x")
        res = await _run_with_watchdog(
            agent, "q", None, MemoryDeps(), ToolRunView(timeout=60.0), RunUsage()
        )
        assert res.output is not None

    asyncio.run(run())
