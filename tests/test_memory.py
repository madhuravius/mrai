"""Session persistence (._mrai/session.json) + long-term memory injection.

All offline: the system-prompt path is exercised with pydantic-ai's TestModel,
and the trim/load/save helpers run against real ModelMessage round-trips.
"""

import asyncio
from pathlib import Path
from typing import Any

from mrai.agent import MemoryDeps
from mrai.cli import _load_history, _save_history, _trim_history
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel


def _roll(messages: int) -> list[Any]:
    """Produce exactly `messages` real ModelMessages from a single session."""

    async def run() -> list[Any]:
        agent = Agent(model=TestModel(), instructions="x")
        history: list[Any] = []
        while len(history) < messages:
            res = await agent.run(f"q{len(history)}", message_history=history or None)
            history = list(res.all_messages())
        return history[:messages]

    return asyncio.run(run())


# --- trimming / persistence helpers -------------------------------------------


def test_trim_history_bounds_and_copies():
    hist = _roll(4)
    trimmed = _trim_history(hist, 2)
    assert len(trimmed) == 2
    assert trimmed == hist[-2:]  # newest first order preserved
    assert len(hist) == 4  # source list is never mutated


def test_trim_history_keeps_all_when_unbounded():
    hist = _roll(4)
    assert _trim_history(hist, 0) == hist  # 0 = no cap
    assert _trim_history(hist, 5) == hist


def test_trim_drops_turn_without_reply():
    hist = _roll(3)
    assert _trim_history(hist, 0) == hist[:2]


def test_session_roundtrip_via_file(tmp_path: Path):
    hist = _roll(4)
    path = tmp_path / "session.json"
    _save_history(path, hist, 2)
    assert path.is_file()
    loaded = _load_history(path, 2)
    assert loaded == hist[-2:]


def test_session_load_missing_or_corrupt(tmp_path: Path):
    assert _load_history(tmp_path / "nope.json", 20) == []
    bad = tmp_path / "session.json"
    bad.write_text("{not json")
    assert _load_history(bad, 20) == []


# --- memory injection via deps + @agent.instructions --------------------------


def _memory_agent(model: TestModel) -> Agent[MemoryDeps, str]:
    agent = Agent(model=model, instructions="You are mrai.", deps_type=MemoryDeps)

    @agent.instructions
    def _known_context(ctx: RunContext[MemoryDeps]) -> str | None:
        d = ctx.deps
        return d.known_context if d and d.known_context else None

    return agent


def _instruction_contents(model: TestModel) -> list[str | None]:
    params = model.last_model_request_parameters
    if params is None or params.instruction_parts is None:
        return []
    return [getattr(p, "content", None) for p in params.instruction_parts]


def test_memory_injected_via_deps_on_every_run():
    model = TestModel()
    agent = _memory_agent(model)

    async def run() -> None:
        # default: empty deps -> no memory instruction
        await agent.run("q1", deps=MemoryDeps())
        assert _instruction_contents(model) == ["You are mrai."]

        res = await agent.run("q2", deps=MemoryDeps(known_context="MEM: alpha"))
        assert "MEM: alpha" in _instruction_contents(model)

        # worst-case trim: history reduced to only the last response, so no
        # prior request (and thus no system-prompt part) survives. The memory
        # block must still be re-applied on the continued run.
        hist = [m for m in res.all_messages() if "Response" in type(m).__name__]
        await agent.run("q3", message_history=hist, deps=MemoryDeps(known_context="MEM: alpha"))
        assert "MEM: alpha" in _instruction_contents(model)

        # a changed memory block shows up on the very next run
        await agent.run(
            "q4",
            message_history=list(res.all_messages()),
            deps=MemoryDeps(known_context="MEM: beta"),
        )
        assert "MEM: beta" in _instruction_contents(model)

    asyncio.run(run())


def _tool_roll() -> list[Any]:

    async def run() -> list[Any]:
        agent = Agent(model=TestModel(), instructions="x")

        @agent.tool_plain
        def read_file(path: str) -> str:
            return "contents"

        history: list[Any] = []
        for q in ("first question", "second question"):
            res = await agent.run(q, message_history=history or None)
            history = list(res.all_messages())
        return history

    return asyncio.run(run())


def _prompt(msg: Any) -> str:
    return msg.parts[0].content


def _pairs_ok(msgs: list[Any]) -> bool:
    from pydantic_ai.messages import ModelRequest, ModelResponse

    kinds = [type(m) for m in msgs]
    return kinds == [ModelRequest, ModelResponse] * (len(msgs) // 2)


def test_trim_never_cuts_mid_tool_exchange():
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart

    hist = _tool_roll()
    assert len(hist) > 4
    for n in range(0, len(hist) + 1):
        trimmed = _trim_history(hist, n)
        parts = [p for m in trimmed for p in m.parts]
        assert not any(isinstance(p, (ToolCallPart, ToolReturnPart)) for p in parts)
        assert _pairs_ok(trimmed)
    assert _prompt(_trim_history(hist, 2)[0]) == "second question"


def test_load_repairs_roll_saved_mid_exchange(tmp_path: Path):
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    hist = _tool_roll()
    path = tmp_path / "session.json"
    path.write_bytes(ModelMessagesTypeAdapter.dump_json(hist[2:]))
    loaded = _load_history(path, 20)
    assert _pairs_ok(loaded) and _prompt(loaded[0]) == "second question"
