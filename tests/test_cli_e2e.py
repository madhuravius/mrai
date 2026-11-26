"""End-to-end pipeline with pydantic-ai's offline TestModel."""

import json

import pytest
from mrai import agent, cli
from pydantic_ai.models.test import TestModel
from typer.testing import CliRunner


@pytest.fixture
def offline_models(monkeypatch):
    models = iter([TestModel(call_tools=["git_status"])])
    monkeypatch.setattr(agent, "_model", lambda *a: next(models, TestModel(call_tools=[])))


def _ai(*argv: str):
    return CliRunner().invoke(cli.app, ["--flash", "--no-web", *argv, "where is main?"])


def test_full_run_answers_logs_and_saves_session(git_repo, cli_hermetic, offline_models):
    result = _ai()
    state = cli_hermetic / ".mrai"

    assert result.exit_code == 0, result.output
    assert "git_status" in result.output
    assert "tool calls (1): git_status: 1" in result.output
    assert "Jev final selection unavailable" in result.output
    assert json.loads((state / "session.json").read_text())
    assert list((state / "renders").glob("*.html"))
    assert list(state.glob("*.md"))


def test_no_write_leaves_no_state(git_repo, cli_hermetic, offline_models):
    result = _ai("--no-write")

    assert result.exit_code == 0, result.output
    assert not (cli_hermetic / ".mrai").exists()
