"""Testes de ferramentas (tools)."""

from __future__ import annotations

from pathlib import Path

import pytest

from project.tools.calculator import CalculatorTool
from project.tools.file_reader import FileReadTool


def test_calculator_tool_basic() -> None:
    tool = CalculatorTool()
    result = tool.invoke({"expression": "(2+3)*4"})
    assert str(result) == "20.0"


def test_calculator_tool_rejects_unsafe_expression() -> None:
    tool = CalculatorTool()
    with pytest.raises(ValueError):
        tool.invoke({"expression": "__import__('os').system('echo hi')"})


def test_file_read_tool_reads_allowed_file(dummy_settings, tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")
    tool = FileReadTool(dummy_settings)
    result = tool.invoke({"path": str(file_path)})
    assert "hello" in str(result)


def test_file_read_tool_denies_outside_root(dummy_settings, tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("nope", encoding="utf-8")
    tool = FileReadTool(dummy_settings)
    with pytest.raises(ValueError):
        tool.invoke({"path": str(outside)})
