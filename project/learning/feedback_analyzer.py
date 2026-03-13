"""Analisa feedback e detecta padrões para melhoria contínua."""

from __future__ import annotations

from dataclasses import dataclass

from project.learning.feedback_store import FeedbackEntry


@dataclass(frozen=True)
class FeedbackSummary:
    count: int
    avg_rating: float
    low_rating_count: int
    needs_sources_count: int


class FeedbackAnalyzer:
    def summarize(self, *, entries: list[FeedbackEntry]) -> FeedbackSummary:
        if not entries:
            return FeedbackSummary(
                count=0, avg_rating=0.0, low_rating_count=0, needs_sources_count=0
            )
        total = sum(e.rating for e in entries)
        low = sum(1 for e in entries if e.rating <= 2)
        needs_sources = sum(1 for e in entries if "fonte" in e.comments.lower())
        return FeedbackSummary(
            count=len(entries),
            avg_rating=total / len(entries),
            low_rating_count=low,
            needs_sources_count=needs_sources,
        )
