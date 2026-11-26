"""CLI wiring tests.

Hermetic by construction: no key in env (env vars removed) and no `.env`
(chdir to a temp dir), and Jev routing is monkeypatched offline — so the
invocation never touches the network and always fails at model setup.
"""

import asyncio

import pytest
from mrai import cli
from mrai.agent import Answer
from mrai.cli import app
from pydantic_ai import AgentRunError, UnexpectedModelBehavior
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits
from typer.testing import CliRunner


def _raise(*args, **kwargs):
    raise AssertionError("should not be reached")


def _invoke_missing_key(cli_hermetic, *argv):
    return CliRunner().invoke(app, [*argv, "hello world"])


def _agent_run_error(status: int = 504) -> AgentRunError:
    return AgentRunError(
        f"status_code: {status}, model_name: deepseek/deepseek-v4-pro, "
        "body: Upstream idle timeout exceeded"
    )


def test_missing_key_exits_gracefully(cli_hermetic):
    result = _invoke_missing_key(cli_hermetic)
    assert result.exit_code == 1
    assert "model setup failed" in result.output


def test_no_query_shows_help(cli_hermetic):
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "model setup failed" not in result.output


def test_flags_without_query_show_help(cli_hermetic):
    result = CliRunner().invoke(app, ["--flash"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "model setup failed" not in result.output


def test_version_flag_prints_version(cli_hermetic):
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip().startswith("mrai ")
    assert "Usage:" not in result.output


def test_opt_out_flags_parse_and_still_exit_gracefully(cli_hermetic):
    result = _invoke_missing_key(cli_hermetic, "--no-memory", "--no-write")
    assert result.exit_code == 1
    assert "model setup failed" in result.output


def test_machine_error_exits_gracefully(cli_hermetic, monkeypatch):
    monkeypatch.setattr(cli, "build_researcher", lambda *a, **k: (object(), None))
    monkeypatch.setattr(cli, "build_writers", lambda *a, **k: [object()] * 3)

    async def _stub(researcher, writers, query, history, deps, settings):
        return (
            object(),
            [UnexpectedModelBehavior("Tool 'file_info' exceeded max retries count of 1")] * 3,
            cli.RunUsage(),
        )

    monkeypatch.setattr(cli, "_run_pipeline", _stub)
    result = CliRunner().invoke(app, ["hello world"])
    assert result.exit_code == 1
    assert "all candidates failed" in result.output


def _fake_research():
    class _R:
        def all_messages(self):
            return []

    return _R()


def _fake_answer_result(ans: Answer):
    class _R:
        output = ans

        def all_messages(self):
            return []

    return _R()


def _stub_pipeline(*results):
    """Patch main's pipeline inputs: `results` are the writer outcomes."""

    def _install(monkeypatch):
        monkeypatch.setattr(cli, "build_researcher", lambda *a, **k: (object(), None))
        monkeypatch.setattr(cli, "build_writers", lambda *a, **k: [object()] * 3)

        async def _run_pipeline(researcher, writers, query, history, deps, settings):
            return _fake_research(), list(results), cli.RunUsage()

        monkeypatch.setattr(cli, "_run_pipeline", _run_pipeline)

    return _install


def test_jev_pick_success_path(cli_hermetic, monkeypatch):
    monkeypatch.setattr(cli, "select_final_answer", lambda *a, **k: (2, [0.5, 0.7, 0.9], 0.1, None))
    _stub_pipeline(
        _fake_answer_result(Answer(topic="t1", title="T1", markdown="m1")),
        _fake_answer_result(Answer(topic="t2", title="T2", markdown="m2")),
        _fake_answer_result(Answer(topic="t3", title="T3", markdown="m3")),
    )(monkeypatch)
    result = CliRunner().invoke(app, ["--no-write", "hello"])
    assert result.exit_code == 0
    assert "Jev scores" in result.output
    assert "0.90  " in result.output  # candidate 3's noul is displayed
    assert "3 · T3" in result.output and "→ chosen" in result.output
    assert "m3" in result.output  # winner's markdown is rendered


def test_jev_pick_falls_back_without_jev(cli_hermetic, monkeypatch):
    monkeypatch.setattr(cli, "select_final_answer", lambda *a, **k: (None, [], None, "jev is down"))
    _stub_pipeline(
        _fake_answer_result(Answer(topic="t1", title="T1", markdown="m1")),
        _fake_answer_result(Answer(topic="t2", title="T2", markdown="m2")),
        _fake_answer_result(Answer(topic="t3", title="T3", markdown="m3")),
    )(monkeypatch)
    result = CliRunner().invoke(app, ["--no-write", "hello"])
    assert result.exit_code == 0
    assert "Jev final selection unavailable" in result.output
    assert "Jev scores" not in result.output
    assert "m1" in result.output  # falls back to the first candidate


def test_jev_pick_single_candidate_skips_jev(cli_hermetic, monkeypatch):
    monkeypatch.setattr(cli, "select_final_answer", _raise)  # must not be called
    _stub_pipeline(_fake_answer_result(Answer(topic="t1", title="T1", markdown="m1")))(monkeypatch)
    result = CliRunner().invoke(app, ["--no-write", "hello"])
    assert result.exit_code == 0
    assert "fewer than two candidates" in result.output
    assert "Jev scores" not in result.output
    assert "m1" in result.output


@pytest.mark.parametrize(
    "status,expected", [(504, True), (529, True), (429, True), (400, False), (500, False)]
)
def test_is_retryable(status, expected):
    assert cli._is_retryable(_agent_run_error(status)) is expected


def test_is_retryable_sees_chained_cause():
    exc = AgentRunError("model request failed")
    exc.__cause__ = _agent_run_error(504)
    assert cli._is_retryable(exc) is True


def test_run_limited_retries_on_usage_limit_exceeded(monkeypatch):
    calls = {"n": 0}
    limits_seen = []

    async def _fake(
        agent, query, history, deps, view, usage, live=True, watchdog=True, limits=None
    ):
        calls["n"] += 1
        limits_seen.append(limits.request_limit if limits else None)
        if calls["n"] == 1:
            raise UsageLimitExceeded("request limit exceeded")
        return "done"

    monkeypatch.setattr(cli, "_run_with_watchdog", _fake)
    monkeypatch.setattr(cli, "console", type("C", (), {"print": lambda *a, **k: None})())

    usage = cli.RunUsage()
    out = asyncio.run(
        cli._run_limited(
            object(),
            "q",
            None,
            cli.MemoryDeps(),
            30.0,
            usage,
            limits=UsageLimits(request_limit=100),
        )
    )
    assert out == "done"
    assert calls["n"] == 2
    assert limits_seen == [100, 200]  # retried with double limit


def test_log_failure_writes_stub_but_not_durable_memory(cli_hermetic, monkeypatch):
    mrai_dir = cli_hermetic / ".mrai"
    memory_file = mrai_dir / "memories.md"
    cli._log_failure(mrai_dir, "why is the sky blue", "boom — tool timeout", no_write=False)
    topic = (mrai_dir / "why-is-the-sky-blue.md").read_text()
    assert "boom — tool timeout" in topic
    assert not memory_file.exists()  # no durable memory note for a failure


def test_log_failure_respects_no_write(cli_hermetic):
    mrai_dir = cli_hermetic / ".mrai"
    cli._log_failure(mrai_dir, "q", "boom", no_write=True)
    assert not mrai_dir.exists()


def test_pipeline_failure_exits_and_logs_stub(cli_hermetic, monkeypatch):
    monkeypatch.setattr(cli, "build_researcher", lambda *a, **k: (object(), None))
    monkeypatch.setattr(cli, "build_writers", lambda *a, **k: [object()] * 3)

    async def _stub(researcher, writers, query, history, deps, settings):
        raise cli.ToolCallTimeout("delegate_task")

    monkeypatch.setattr(cli, "_run_pipeline", _stub)
    result = CliRunner().invoke(app, ["hello"])
    assert result.exit_code == 1
    assert "retrying run once" in result.output
    assert "tool call exceeded" in result.output


def test_write_render_writes_html_under_state_dir(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    state = tmp_path / "state"
    path = cli._write_render(state, "hello world", "# Hi\n\nSee `x`.")

    assert path == state / "renders" / "hello-world.html"
    assert "<h1>Hi</h1>" in path.read_text()
    assert not (repo / ".mrai").exists()  # never written into the (possibly ro) repo


def test_host_path_maps_container_state_dir_to_host(tmp_path, monkeypatch):
    state = tmp_path / "state"
    inside = state / "renders" / "x.html"
    assert cli._host_path(inside, state) == inside  # no mapping outside the container
    monkeypatch.setenv("MRAI_HOST_DIR", "/home/u/.mrai/workspaces/repo-1234abcd")
    assert cli._host_path(inside, state) == cli.Path(
        "/home/u/.mrai/workspaces/repo-1234abcd/renders/x.html"
    )


def test_resolve_mrai_dir_precedence(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    default = cli.resolve_mrai_dir(None, repo)
    assert default.parent == tmp_path / "mrai-home" / "workspaces"
    assert default.name.startswith("repo-")
    monkeypatch.setenv("MRAI_DIR", str(tmp_path / "env"))
    assert cli.resolve_mrai_dir(None, repo) == tmp_path / "env"
    assert cli.resolve_mrai_dir(tmp_path / "flag", repo) == tmp_path / "flag"  # --dir wins


def test_default_state_dir_differs_per_workspace(tmp_path):
    a = cli.resolve_mrai_dir(None, tmp_path / "x" / "api")
    b = cli.resolve_mrai_dir(None, tmp_path / "y" / "api")
    assert a != b  # same basename, different checkouts: no shared session
    assert a == cli.resolve_mrai_dir(None, tmp_path / "x" / "api")  # stable


def _workspace_user_error() -> cli.UserError:
    return cli.UserError(
        "The message history continues in workspace `local:/elsewhere`, but none of "
        "the agent's workspace capabilities recognized it."
    )


def test_stale_workspace_history_is_dropped_and_rerun(cli_hermetic, monkeypatch):
    monkeypatch.setattr(cli, "build_researcher", lambda *a, **k: (object(), None))
    monkeypatch.setattr(cli, "build_writers", lambda *a, **k: [object()] * 3)
    monkeypatch.setattr(cli, "_load_history", lambda *a, **k: ["stale"])
    seen = []

    async def _stub(researcher, writers, query, history, deps, settings):
        seen.append(history)
        if history:
            raise _workspace_user_error()
        raise cli.UnexpectedModelBehavior("stop here")  # fresh run reached; end the test

    monkeypatch.setattr(cli, "_run_pipeline", _stub)
    result = CliRunner().invoke(app, ["hello"])
    assert seen == [["stale"], None]
    assert "starting fresh" in result.output


def test_workspace_user_error_without_history_propagates(cli_hermetic, monkeypatch):
    monkeypatch.setattr(cli, "build_researcher", lambda *a, **k: (object(), None))
    monkeypatch.setattr(cli, "build_writers", lambda *a, **k: [object()] * 3)
    calls = {"n": 0}

    async def _stub(researcher, writers, query, history, deps, settings):
        calls["n"] += 1
        raise _workspace_user_error()

    monkeypatch.setattr(cli, "_run_pipeline", _stub)
    result = CliRunner().invoke(app, ["hello"])
    assert calls["n"] == 1  # nothing to drop: no silent retry loop
    assert isinstance(result.exception, cli.UserError)
