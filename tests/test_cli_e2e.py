"""The real two-phase pipeline end to end, with pydantic-ai's offline TestModel
standing in for every LLM: researcher -> 3 writers -> Jev fallback -> logbook."""

import json

import pytest
from mrai import agent, cli
from pydantic_ai.models.test import TestModel
from typer.testing import CliRunner


@pytest.fixture
def offline_models(monkeypatch):
    """The researcher (built first) makes one real git_status call; writers just answer."""
    models = iter([TestModel(call_tools=["git_status"])])
    monkeypatch.setattr(agent, "_model", lambda *a: next(models, TestModel(call_tools=[])))


def _ai(*argv: str):
    return CliRunner().invoke(cli.app, ["--flash", "--no-web", *argv, "where is main?"])


def test_full_run_answers_logs_and_saves_session(git_repo, cli_hermetic, offline_models):
    result = _ai()
    state = cli_hermetic / ".mrai"

    assert result.exit_code == 0, result.output
    assert "git_status" in result.output  # researcher's tool call was streamed
    assert "tool calls (1): git_status: 1" in result.output  # and counted in the footer
    assert "Jev final selection unavailable" in result.output  # offline: first candidate wins
    assert json.loads((state / "session.json").read_text())  # history persisted for next turn
    assert list((state / "renders").glob("*.html"))  # HTML copy of the answer
    assert list(state.glob("*.md"))  # logbook entry for the topic


def test_no_write_leaves_no_state(git_repo, cli_hermetic, offline_models):
    result = _ai("--no-write")

    assert result.exit_code == 0, result.output
    assert not (cli_hermetic / ".mrai").exists()
