"""Testes do ranking de tools com priors."""

from __future__ import annotations

from project.tools.tool_ranker import ToolRanker


class _T:
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description


def test_tool_ranker_uses_priors() -> None:
    tools = [_T("a", "x"), _T("b", "x")]
    ranked = ToolRanker().rank(query="x", tools=tools, priors={"b": 10.0})
    assert ranked[0].name == "b"
