from pathlib import Path

from mrai.agent import Answer
from mrai.prompt import SYSTEM_PROMPT
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
