import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from agent.tools import ToolError, registry
from agent.tools import builtin  # noqa: F401  -- registers the tools


def test_calculator_basic():
    result = registry.get("calculator").call({"expression": "(3 + 4) * 2"})
    assert result["result"] == 14


def test_calculator_rejects_unsafe_expression():
    with pytest.raises(ToolError):
        registry.get("calculator").call({"expression": "__import__('os').system('echo hi')"})


def test_calculator_division_by_zero_is_tool_error():
    with pytest.raises(ToolError):
        registry.get("calculator").call({"expression": "1 / 0"})


def test_write_then_read_file_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(builtin, "WORKSPACE", tmp_path)
    registry.get("write_file").call({"path": "note.txt", "content": "hello"})
    result = registry.get("read_file").call({"path": "note.txt"})
    assert result["content"] == "hello"


def test_read_file_path_traversal_is_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(builtin, "WORKSPACE", tmp_path)
    with pytest.raises(ToolError, match="escapes"):
        registry.get("read_file").call({"path": "../../etc/passwd"})


def test_read_missing_file_raises_tool_error(tmp_path, monkeypatch):
    monkeypatch.setattr(builtin, "WORKSPACE", tmp_path)
    with pytest.raises(ToolError):
        registry.get("read_file").call({"path": "does_not_exist.txt"})


def test_tool_schema_shape():
    schema = registry.get("calculator").to_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "calculator"
    assert "expression" in schema["function"]["parameters"]["properties"]
