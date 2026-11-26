"""CLI wiring tests.

Hermetic: no API key in env, cwd is a temp dir, Jev routing is offline. Tests
that need a run past model setup stub `_run_pipeline` via the `pipeline` fixture;
the real pipeline is exercised in test_cli_e2e.py.
"""

import asyncio
from types import SimpleNamespace

import pytest
from mrai import cli
from mrai.agent import Answer
from mrai.cli import app
from pydantic_ai import AgentRunError, UnexpectedModelBehavior
from pydantic_ai.exceptions import UsageLimitExceeded
from typer.testing import CliRunner


@pytest.fixture(autouse=True)
def wide_console(monkeypatch):
    """Rich wraps at the terminal width; keep each message on one line for asserts."""
    monkeypatch.setattr(cli.console, "width", 1000)


def _ai(*argv: str):
    return CliRunner().invoke(app, list(argv))


def _upstream_error(status: int = 504) -> AgentRunError:
    return AgentRunError(
        f"status_code: {status}, model_name: deepseek/deepseek-v4-pro, "
        "body: Upstream idle timeout exceeded"
    )


def _research():
    return SimpleNamespace(all_messages=lambda: [], new_messages=lambda: [])


def _answer(n: int):
    """A writer result whose Answer is topic tN / title TN / markdown mN."""
    ans = Answer(topic=f"t{n}", title=f"T{n}", markdown=f"m{n}")
    return SimpleNamespace(output=ans, all_messages=lambda: [], new_messages=lambda: [])


class _ScriptedPipeline:
    """Each pipeline call raises or returns the next queued outcome."""

    def __init__(self):
        self.outcomes: list = []
        self.histories: list = []

    def __call__(self, *outcomes):
        self.outcomes.extend(outcomes)

    async def run(self, researcher, writers, query, history, deps, settings):
        self.histories.append(history)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return _research(), outcome, cli.RunUsage()


@pytest.fixture
def pipeline(cli_hermetic, monkeypatch):
    """Skip model setup and replace the real pipeline with scripted outcomes."""
    script = _ScriptedPipeline()
    monkeypatch.setattr(cli, "build_researcher", lambda *a, **k: (object(), None))
    monkeypatch.setattr(
        cli, "build_writers", lambda root, model, angles, **k: [object()] * len(angles)
    )
    monkeypatch.setattr(cli, "_run_pipeline", script.run)
    return script


def test_missing_key_exits_gracefully(cli_hermetic):
    result = _ai("--no-memory", "--no-write", "hello world")
    assert result.exit_code == 1
    assert "model setup failed" in result.output


@pytest.mark.parametrize("argv", [[], ["--flash"]])
def test_no_query_shows_help(cli_hermetic, argv):
    result = _ai(*argv)
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "model setup failed" not in result.output


def test_version_flag_prints_version(cli_hermetic):
    result = _ai("--version")
    assert result.exit_code == 0
    assert result.output.strip().startswith("mrai ")


def test_sandbox_without_bwrap_says_why(cli_hermetic, monkeypatch):
    monkeypatch.delenv("MRAI_SANDBOXED")
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    result = _ai("hello")
    assert result.exit_code == 1
    assert "bubblewrap" in result.output and "--no-sandbox" in result.output


def test_no_sandbox_skips_bwrap(cli_hermetic, monkeypatch):
    monkeypatch.delenv("MRAI_SANDBOXED")
    monkeypatch.setattr(cli, "_enter_sandbox", lambda **k: pytest.fail("sandboxed"))
    assert _ai("--no-sandbox").exit_code == 0


def test_jev_picks_best_candidate(pipeline, monkeypatch):
    monkeypatch.setattr(cli, "select_final_answer", lambda *a, **k: (2, [0.5, 0.7, 0.9], 0.1, None))
    pipeline([_answer(1), _answer(2), _answer(3)])
    result = _ai("--no-write", "hello")
    assert result.exit_code == 0
    assert "3 · T3" in result.output and "0.90" in result.output and "→ chosen" in result.output
    assert "m3" in result.output


def test_without_jev_first_candidate_wins(pipeline, monkeypatch):
    monkeypatch.setattr(cli, "select_final_answer", lambda *a, **k: (None, [], None, "jev is down"))
    pipeline([_answer(1), _answer(2), _answer(3)])
    result = _ai("--no-write", "hello")
    assert result.exit_code == 0
    assert "Jev final selection unavailable" in result.output
    assert "m1" in result.output


def test_single_candidate_skips_jev(pipeline, monkeypatch):
    monkeypatch.setattr(cli, "select_final_answer", pytest.fail)
    pipeline([_answer(1)])
    result = _ai("--no-write", "hello")
    assert result.exit_code == 0
    assert "fewer than two candidates" in result.output
    assert "m1" in result.output


def test_all_candidates_failing_exits(pipeline):
    pipeline([UnexpectedModelBehavior("Tool 'file_info' exceeded max retries")] * 3)
    result = _ai("hello")
    assert result.exit_code == 1
    assert "all candidates failed" in result.output


def test_routing_decision_is_logged(pipeline, cli_hermetic, monkeypatch):
    import json

    monkeypatch.setattr(cli, "select_final_answer", lambda *a, **k: (0, [0.9], None, None))
    pipeline([_answer(1)])
    assert _ai("hello").exit_code == 0
    lines = (cli_hermetic / ".mrai" / "routing.jsonl").read_text().splitlines()
    rec = json.loads(lines[-1])
    assert rec["query"] == "hello" and rec["source"] == "fallback" and rec["tier"] == "pro"
    assert rec["jev_error"]
    assert (cli_hermetic / ".mrai" / "routing.jsonl").stat().st_mode & 0o777 == 0o600


def test_no_write_skips_routing_log(pipeline, cli_hermetic):
    pipeline([_answer(1)])
    assert _ai("--no-write", "hello").exit_code == 0
    assert not (cli_hermetic / ".mrai" / "routing.jsonl").exists()


def test_non_retryable_error_exits_without_retry(pipeline):
    pipeline(_upstream_error(400))
    result = _ai("--no-write", "hello")
    assert result.exit_code == 1
    assert "run failed" in result.output and "retrying" not in result.output


@pytest.mark.parametrize(
    "status,expected", [(504, True), (529, True), (429, True), (400, False), (500, False)]
)
def test_is_retryable(status, expected):
    assert cli._is_retryable(_upstream_error(status)) is expected


def test_is_retryable_sees_chained_cause():
    exc = AgentRunError("model request failed")
    exc.__cause__ = _upstream_error(504)
    assert cli._is_retryable(exc) is True


def test_log_failure_writes_stub_but_not_durable_memory(cli_hermetic):
    state = cli_hermetic / ".mrai"
    cli._log_failure(state, "why is the sky blue", "boom — tool timeout", no_write=False)
    assert "boom — tool timeout" in (state / "why-is-the-sky-blue.md").read_text()
    assert not (state / "memories.md").exists()


def test_log_failure_respects_no_write(cli_hermetic):
    cli._log_failure(cli_hermetic / ".mrai", "q", "boom", no_write=True)
    assert not (cli_hermetic / ".mrai").exists()


def _workspace_error() -> cli.UserError:
    return cli.UserError(
        "The message history continues in workspace `local:/elsewhere`, but none of "
        "the agent's workspace capabilities recognized it."
    )


def test_history_from_another_workspace_is_dropped(pipeline, monkeypatch):
    monkeypatch.setattr(cli, "_load_history", lambda *a, **k: ["stale"])
    pipeline(_workspace_error(), [_answer(1)])
    result = _ai("--no-write", "hello")
    assert pipeline.histories == [["stale"], None]
    assert "starting fresh" in result.output and result.exit_code == 0


def test_workspace_error_without_history_propagates(pipeline):
    pipeline(_workspace_error())
    result = _ai("hello")
    assert pipeline.histories == [None]
    assert isinstance(result.exception, cli.UserError)


def test_resolve_mrai_dir_precedence(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    default = cli.resolve_mrai_dir(None, repo)
    assert default.parent == tmp_path / "mrai-home" / "workspaces"
    assert default.name.startswith("repo-")
    monkeypatch.setenv("MRAI_DIR", str(tmp_path / "env"))
    assert cli.resolve_mrai_dir(None, repo) == tmp_path / "env"
    assert cli.resolve_mrai_dir(tmp_path / "flag", repo) == tmp_path / "flag"


def test_default_state_dir_differs_per_workspace(tmp_path):
    a = cli.resolve_mrai_dir(None, tmp_path / "x" / "api")
    b = cli.resolve_mrai_dir(None, tmp_path / "y" / "api")
    assert a != b
    assert a == cli.resolve_mrai_dir(None, tmp_path / "x" / "api")


def test_stats_markdown_is_a_labelled_table():
    stats = cli.RunStats(
        cost=None,
        jev=0.000074,
        requests=9,
        tools=cli.Counter({"read_file": 6, "git_diff": 2}),
        web=["https://x.example/a|b"],
    )
    md = cli._stats_markdown(stats)
    assert md.startswith(cli.STATS_MARKER)
    assert "| Run stats | |" in md
    assert "| Cost | unavailable |" in md
    assert "| Jev | $0.000074 |" in md
    assert "| Requests | 9 |" in md
    assert "| Tool calls (8) | `read_file` 6 · `git_diff` 2 |" in md
    assert "| Web fetches | https://x.example/a\\|b |" in md


def test_upstream_error_fails_without_rerunning_the_pipeline(pipeline, cli_hermetic):
    pipeline(_upstream_error(504), [_answer(1)])
    result = _ai("hello")
    assert result.exit_code == 1
    assert "run failed after retrying the request" in result.output
    assert len(pipeline.histories) == 1  # the transport retried; the run did not
    assert "run failed" in (cli_hermetic / ".mrai" / "hello.md").read_text()


def test_hung_tool_fails_without_rerunning(pipeline):
    pipeline(cli.ToolCallTimeout("delegate_task"), [_answer(1)])
    result = _ai("--no-write", "hello")
    assert result.exit_code == 1 and "hung past its time budget" in result.output
    assert len(pipeline.histories) == 1


def _route(tier: str, source: str) -> cli.Route:
    return cli.Route("m", 1.0, source, tier=tier)


@pytest.mark.parametrize(
    "tier,source,n",
    [
        ("pro", "jev", 3),
        ("pro", "flag", 3),
        ("pro", "fallback", 1),
        ("flash", "jev", 1),
        ("", "flag", 1),
    ],
)
def test_only_a_judged_complex_question_fans_out(tier, source, n):
    assert len(cli._angles(_route(tier, source))) == n


def test_trivial_question_is_answered_but_not_logged(pipeline, cli_hermetic):
    pipeline([_answer(1)])
    result = _ai("hi")
    assert result.exit_code == 0 and "m1" in result.output
    assert "not logged" in result.output
    assert not list((cli_hermetic / ".mrai").glob("t1.md"))


def test_is_trivial_needs_both_short_and_tool_free():
    assert cli._is_trivial("hi there", cli.RunStats(cost=None))
    assert not cli._is_trivial("hi", cli.RunStats(cost=None, tools=cli.Counter({"read_file": 1})))
    assert not cli._is_trivial("where is slugify defined", cli.RunStats(cost=None))


def test_history_saves_the_answer_not_the_brief(tmp_path):
    path = tmp_path / "session.json"
    cli._append_turn(path, [], "q1", "answer one", 20)
    cli._append_turn(path, cli._load_history(path, 20), "q2", "answer two", 20)
    texts = [getattr(p, "content", None) for m in cli._load_history(path, 20) for p in m.parts]
    assert texts == ["q1", "answer one", "q2", "answer two"]


def test_writers_get_only_the_brief(monkeypatch):
    calls = []

    async def fake(agent, prompt, history, deps, view, usage, **kw):
        calls.append((agent, prompt, history))
        if agent == "researcher":
            return SimpleNamespace(output="Question: q\nslugify at logbook.py:10")
        return _answer(1)

    monkeypatch.setattr(cli, "_run_with_watchdog", fake)
    settings = cli.Settings(openrouter_api_key=None)
    writers = [("w1", "angle one"), ("w2", "angle two")]
    _, results, _ = asyncio.run(
        cli._run_pipeline("researcher", writers, "q", ["old turn"], cli.MemoryDeps(), settings)
    )
    assert calls[0] == ("researcher", "q", ["old turn"])
    for _agent, prompt, history in calls[1:]:
        assert history is None
        assert "slugify at logbook.py:10" in prompt
    assert {c[0] for c in calls[1:]} == {"w1", "w2"} and len(results) == 2


def test_request_limit_writes_from_partial_evidence(monkeypatch):
    prompts = []

    async def fake(agent, prompt, history, deps, view, usage, **kw):
        if agent == "researcher":
            raise UsageLimitExceeded("request limit")
        prompts.append(prompt)
        return _answer(1)

    monkeypatch.setattr(cli, "_run_with_watchdog", fake)
    research, results, _ = asyncio.run(
        cli._run_pipeline(
            "researcher",
            [("w", "a")],
            "q",
            None,
            cli.MemoryDeps(),
            cli.Settings(openrouter_api_key=None),
        )
    )
    assert isinstance(research, cli._PartialResearch)
    assert "evidence is partial" in prompts[0] and len(results) == 1


def test_partial_evidence_keeps_calls_results_and_notes():
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
    )

    msgs = [
        ModelResponse(
            parts=[TextPart(content="looking"), ToolCallPart("read_file", {"path": "a.py"})]
        ),
        ModelRequest(parts=[ToolReturnPart("read_file", "x" * 5000)]),
    ]
    text = cli._partial_evidence(msgs, per_part=100)
    assert "looking" in text and '→ read_file({"path": "a.py"})' in text
    assert "x" * 100 + " …" in text and "x" * 101 not in text


def test_reported_cost_sums_this_runs_responses_only():
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

    def resp(cost):
        return ModelResponse(parts=[TextPart("x")], provider_details={"cost": cost})

    research = SimpleNamespace(
        new_messages=lambda: [ModelRequest(parts=[UserPromptPart("q")]), resp(0.002)]
    )
    writer = SimpleNamespace(new_messages=lambda: [resp(0.0005)])
    assert cli._reported_cost([research, writer]) == cli.Decimal("0.0025")
    assert cli._reported_cost([SimpleNamespace(new_messages=lambda: [])]) is None


def test_partial_research_excludes_replayed_history():
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    h1, h2, new = (ModelRequest(parts=[UserPromptPart(t)]) for t in ("h1", "h2", "new"))
    r = cli._PartialResearch([h1, h2, new], replayed=2)
    assert r.new_messages() == [new] and r.all_messages() == [h1, h2, new]


def test_cost_note_is_shown():
    stats = cli.RunStats(cost=0.01, cost_note="excl. explorer subagent")
    assert "| Cost | $0.010000 (excl. explorer subagent) |" in cli._stats_markdown(stats)


def test_openrouter_models_request_usage_accounting():
    from mrai import agent as agent_mod

    model = agent_mod._model(cli.Settings(openrouter_api_key="k"), "k", "a/b")
    assert model.settings == {"openrouter_usage": {"include": True}}


def test_profile_flag_selects_the_profile(pipeline, monkeypatch):
    seen = []
    monkeypatch.setattr(
        cli,
        "resolve_route",
        lambda *a, **k: (
            seen.append(cli.os.environ.get("MRAI_PROFILE")) or cli.Route("m", 1.0, "flag")
        ),
    )
    pipeline([_answer(1)])
    _ai("--no-write", "--profile", "work", "hello")
    assert seen == ["work"]
