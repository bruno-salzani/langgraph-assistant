"""Seleção automática de tools via ranking baseado em texto."""

from __future__ import annotations

from dataclasses import dataclass

from project.utils.text import bm25_scores


@dataclass(frozen=True)
class RankedTool:
    name: str
    score: float


class ToolRanker:
    def rank(
        self,
        *,
        query: str,
        tools: list,
        priors: dict[str, float] | None = None,
    ) -> list[RankedTool]:
        docs = [f"{t.name}\n{t.description}" for t in tools]
        scores = bm25_scores(query=query, documents=docs)
        priors = priors or {}
        ranked = sorted(
            (
                RankedTool(
                    name=t.name,
                    score=float(s) + float(priors.get(t.name, 0.0)),
                )
                for t, s in zip(tools, scores, strict=False)
            ),
            key=lambda rt: rt.score,
            reverse=True,
        )
        return ranked
