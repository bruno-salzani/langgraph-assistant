from __future__ import annotations

from dataclasses import dataclass

from project.learning.feedback_analyzer import FeedbackSummary


@dataclass(frozen=True)
class PromptEval:
    score: float
    reasons: list[str]


class PromptEvaluator:
    def evaluate(self, *, prompt: str, summary: FeedbackSummary) -> PromptEval:
        p = (prompt or "").lower()
        score = 0.0
        reasons: list[str] = []

        if "regras adicionais" in p:
            score += 0.1

        if summary.needs_sources_count > 0:
            if "fontes" in p:
                score += 0.6
                reasons.append("fontes_rule_present")
            else:
                score -= 0.8
                reasons.append("missing_sources_rule")

        if summary.low_rating_count > 0:
            if "conciso" in p or "concis" in p:
                score += 0.3
                reasons.append("concise_rule_present")
            else:
                score -= 0.2
                reasons.append("missing_concise_rule")

        score = max(-1.0, min(1.0, score))
        return PromptEval(score=score, reasons=reasons)

