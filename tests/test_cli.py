"""CLI wiring tests.

Hermetic by construction: no key in env (env vars removed) and no `.env`
(chdir to a temp dir), and Jev routing is monkeypatched offline — so the
invocation never touches the network and always fails at model setup.
"""

from mrai import cli, router
from mrai.agent import Answer, AnswerCandidates
from mrai.cli import app
from pydantic_ai import UnexpectedModelBehavior
from typer.testing import CliRunner


def _no_network(*args, **kwargs):
    raise AssertionError("network call should not happen")


def _raise(*args, **kwargs):
    raise AssertionError("should not be reached")


def _invoke_missing_key(monkeypatch, tmp_path, *argv):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MRAI_OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here, so Settings() is keyless
    monkeypatch.setattr(router.httpx, "post", _no_network)  # skip Jev routing offline
    return CliRunner().invoke(app, [*argv, "hello world"])


def test_missing_key_exits_gracefully(monkeypatch, tmp_path):
    result = _invoke_missing_key(monkeypatch, tmp_path)
    assert result.exit_code == 1
    assert "model setup failed" in result.output


def test_no_query_shows_help(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MRAI_OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(router.httpx, "post", _no_network)
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "model setup failed" not in result.output


def test_flags_without_query_show_help(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MRAI_OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(router.httpx, "post", _no_network)  # guard must exit before routing
    result = CliRunner().invoke(app, ["--flash"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "model setup failed" not in result.output


def test_version_flag_prints_version(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MRAI_OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(router.httpx, "post", _no_network)  # exits before any routing
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip().startswith("mrai ")
    assert "Usage:" not in result.output


def test_opt_out_flags_parse_and_still_exit_gracefully(monkeypatch, tmp_path):
    result = _invoke_missing_key(monkeypatch, tmp_path, "--no-memory", "--no-write")
    assert result.exit_code == 1
    assert "model setup failed" in result.output


def test_machine_error_exits_gracefully(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MRAI_OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here, so Settings() is keyless
    monkeypatch.setattr(router.httpx, "post", _no_network)  # skip Jev routing offline
    monkeypatch.setattr(cli, "build_agent", lambda *a, **k: (object(), None))
    monkeypatch.setattr(
        cli,
        "_run_with_watchdog",
        lambda *a, **k: (_ for _ in ()).throw(
            UnexpectedModelBehavior("Tool 'file_info' exceeded max retries count of 1")
        ),
    )
    result = CliRunner().invoke(app, ["hello world"])
    assert result.exit_code == 1
    assert "model gave up on a failing tool call" in result.output


def _fake_result(cands):
    class _R:
        output = AnswerCandidates.model_construct(candidates=cands)

        def all_messages(self):
            return []

    return _R()


def test_jev_pick_success_path(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MRAI_OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(router.httpx, "post", _no_network)
    monkeypatch.setattr(cli, "build_agent", lambda *a, **k: (object(), None))
    monkeypatch.setattr(cli, "select_final_answer", lambda *a, **k: (2, [0.5, 0.7, 0.9], 0.1, None))

    async def _stub(*args, **kwargs):
        return _fake_result(
            [
                Answer(topic="t1", title="T1", markdown="m1"),
                Answer(topic="t2", title="T2", markdown="m2"),
                Answer(topic="t3", title="T3", markdown="m3"),
            ]
        )

    monkeypatch.setattr(cli, "_run_with_watchdog", _stub)
    result = CliRunner().invoke(app, ["--no-write", "hello"])
    assert result.exit_code == 0
    assert "Jev scores" in result.output
    assert "0.90  " in result.output  # candidate 3's noul is displayed
    assert "3 · T3" in result.output and "→ chosen" in result.output
    assert "m3" in result.output  # winner's markdown is rendered


def test_jev_pick_falls_back_without_jev(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MRAI_OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(router.httpx, "post", _no_network)
    monkeypatch.setattr(cli, "build_agent", lambda *a, **k: (object(), None))
    monkeypatch.setattr(cli, "select_final_answer", lambda *a, **k: (None, [], None, "jev is down"))

    async def _stub(*args, **kwargs):
        return _fake_result(
            [
                Answer(topic="t1", title="T1", markdown="m1"),
                Answer(topic="t2", title="T2", markdown="m2"),
                Answer(topic="t3", title="T3", markdown="m3"),
            ]
        )

    monkeypatch.setattr(cli, "_run_with_watchdog", _stub)
    result = CliRunner().invoke(app, ["--no-write", "hello"])
    assert result.exit_code == 0
    assert "Jev final selection unavailable" in result.output
    assert "Jev scores" not in result.output
    assert "m1" in result.output  # falls back to the first candidate


def test_jev_pick_single_candidate_skips_jev(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MRAI_OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(router.httpx, "post", _no_network)
    monkeypatch.setattr(cli, "build_agent", lambda *a, **k: (object(), None))
    monkeypatch.setattr(cli, "select_final_answer", _raise)  # must not be called

    async def _stub(*args, **kwargs):
        return _fake_result([Answer(topic="t1", title="T1", markdown="m1")])

    monkeypatch.setattr(cli, "_run_with_watchdog", _stub)
    result = CliRunner().invoke(app, ["--no-write", "hello"])
    assert result.exit_code == 0
    assert "fewer than two candidates" in result.output
    assert "Jev scores" not in result.output
    assert "m1" in result.output
