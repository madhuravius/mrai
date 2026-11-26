from pathlib import Path

import mrai.agent as agent_mod
from mrai.agent import Answer, MemoryDeps
from mrai.prompt import SYSTEM_PROMPT
from mrai.settings import Settings
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness import FileSystem

READ_TOOLS = {"read_file", "list_directory", "find_files", "search_files", "file_info"}
WRITE_TOOLS = {"write_file", "edit_file", "create_directory"}


def _offered_tools(tmp_path: Path) -> set[str]:
    test_model = TestModel()
    agent = Agent(
        model=test_model,
        output_type=Answer,
        instructions=SYSTEM_PROMPT,
        capabilities=[
            LocalWorkspace(tmp_path, read_only=True),
            FileSystem(read_only=True, content_hashes=False),
        ],
    )
    agent.run_sync("hello")
    assert test_model.last_model_request_parameters is not None
    return {t.name for t in test_model.last_model_request_parameters.function_tools}


def test_only_read_tools_offered_to_model(tmp_path: Path):
    names = _offered_tools(tmp_path)
    assert names >= READ_TOOLS
    assert not WRITE_TOOLS & names


RESULT = {
    "candidates": [
        {"topic": "a", "title": "A", "markdown": "one"},
        {"topic": "b", "title": "B", "markdown": "two"},
    ]
}


def test_model_gateway_when_base_url_set():
    s = Settings(base_url="http://gw.invalid", openrouter_api_key="k")
    assert type(agent_mod._model(s, "k", "some/model")).__name__ == "OpenAIChatModel"


def test_model_defaults_to_openrouter():
    s = Settings(openrouter_api_key="k")
    assert type(agent_mod._model(s, "k", "some/model")).__name__ == "OpenRouterModel"


def test_researcher_gets_tools_web(cli_hermetic, monkeypatch):
    tm = TestModel()  # researcher output_type=str (TestModel emits plain text)
    monkeypatch.setattr(agent_mod, "_model", lambda *a, **k: tm)

    agent, webfetch = agent_mod.build_researcher(
        cli_hermetic, "test", settings=Settings(openrouter_api_key=None)
    )
    assert webfetch is not None
    agent.run_sync("hello", deps=MemoryDeps())

    assert tm.last_model_request_parameters is not None
    names = {t.name for t in tm.last_model_request_parameters.function_tools}
    assert {"git_diff", "git_status", "git_compare", "web_fetch"} <= names


def test_writers_are_tool_less(cli_hermetic, monkeypatch):
    tm = TestModel(
        custom_output_args={"topic": "t", "title": "T", "markdown": "m"}
    )  # valid single Answer
    monkeypatch.setattr(agent_mod, "_model", lambda *a, **k: tm)

    writers = agent_mod.build_writers(
        cli_hermetic, "test", ["a", "b", "c"], settings=Settings(openrouter_api_key=None)
    )
    assert len(writers) == 3
    writers[0].run_sync("Question: q", deps=MemoryDeps())

    assert tm.last_model_request_parameters is not None
    names = {t.name for t in tm.last_model_request_parameters.function_tools}
    assert names == set()  # no tools, no web, no git — pure synthesis
