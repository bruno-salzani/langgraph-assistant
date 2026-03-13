"""Sistema de aprendizado de ferramentas (Tool Learning) com ranking por histórico."""

from __future__ import annotations

from dataclasses import dataclass

from project.learning.tool_learning import ToolLearningStore
from project.tools.tool_ranker import ToolRanker


@dataclass(frozen=True)
class RankedToolSuggestion:
    tool_name: str
    score: float


class ToolLearningSystem:
    def __init__(self, *, store: ToolLearningStore) -> None:
        self._store = store
        self._ranker = ToolRanker()

    def rank_tools(self, question: str, tools: list) -> list[RankedToolSuggestion]:
        priors = self._store.priors()
        ranked = self._ranker.rank(query=question, tools=tools, priors=priors)
        return [RankedToolSuggestion(tool_name=r.name, score=r.score) for r in ranked]
