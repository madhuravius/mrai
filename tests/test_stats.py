from collections import Counter
from types import SimpleNamespace

from mrai.cli import _fmt_args, _one_line, count_tool_calls
from pydantic_ai.messages import ToolCallPart


def _msg(*parts):
    return SimpleNamespace(parts=list(parts))


def test_count_tool_calls_aggregates():
    parts = [
        ToolCallPart(tool_call_id="1", tool_name="read_file", args={}),
        ToolCallPart(tool_call_id="2", tool_name="read_file", args={}),
        ToolCallPart(tool_call_id="3", tool_name="web_fetch", args={}),
    ]
    result = SimpleNamespace(all_messages=lambda: [_msg(*parts)])
    assert count_tool_calls(result) == Counter({"read_file": 2, "web_fetch": 1})


def test_one_line_collapses_and_truncates():
    assert _one_line("a\n\n  b\t c") == "a b c"
    long = "x" * 300
    assert len(_one_line(long)) == 200 and _one_line(long).endswith("...")


def test_fmt_args_short_json_and_truncation():
    assert _fmt_args({"a": 1}) == '{"a": 1}'
    assert _fmt_args(None) == ""
    assert len(_fmt_args({"blob": "z" * 500})) <= 200
