"""Testes do ToolLearningStore."""

from __future__ import annotations

from pathlib import Path

from project.learning.tool_learning import ToolLearningStore


def test_tool_learning_priors(tmp_path: Path) -> None:
    store = ToolLearningStore(sqlite_path=tmp_path / "tool_learning.sqlite3")
    store.record(tool_name="web_search", success=True)
    store.record(tool_name="web_search", success=True)
    store.record(tool_name="web_search", success=False)
    priors = store.priors()
    assert "web_search" in priors
    assert priors["web_search"] > 0.0
