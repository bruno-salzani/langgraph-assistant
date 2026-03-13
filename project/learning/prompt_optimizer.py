"""Otimiza prompts automaticamente com base em feedback (heurístico)."""

from __future__ import annotations

from dataclasses import dataclass

from project.learning.feedback_analyzer import FeedbackSummary


@dataclass(frozen=True)
class OptimizedPrompt:
    version: str
    system_prompt: str


class PromptOptimizer:
    def optimize(
        self, *, base_prompt: str, summary: FeedbackSummary, target_version: str
    ) -> OptimizedPrompt:
        prompt = base_prompt.strip()
        if summary.needs_sources_count > 0 and "Fontes:" not in prompt:
            prompt += "\n\nRegras adicionais:\n- Sempre inclua uma seção 'Fontes:' quando usar RAG ou web."
        if summary.low_rating_count > 0 and "Seja conciso" not in prompt:
            prompt += "\n- Seja conciso e direto quando possível."
        return OptimizedPrompt(version=target_version, system_prompt=prompt + "\n")
