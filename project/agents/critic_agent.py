"""Critic agent: avalia e melhora uma resposta com base em evidências."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from project.services.llm_router import ModelRouter


@dataclass(frozen=True)
class CriticResult:
    verdict: str
    improved_answer: str


class CriticAgent:
    """Avalia qualidade e sugere melhorias na resposta final."""

    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    def evaluate(self, question: str, answer: str) -> dict[str, Any]:
        q = (question or "").strip()
        a = (answer or "").strip()
        problems: list[str] = []
        suggestions: list[str] = []

        if not a:
            problems.append("Resposta vazia.")
            suggestions.append("Fornecer uma resposta objetiva e completa.")
        if len(a) < 40:
            problems.append("Resposta curta demais.")
            suggestions.append("Adicionar detalhes relevantes, mantendo concisão.")
        if "talvez" in a.lower() or "provavelmente" in a.lower():
            problems.append("Risco de imprecisão (linguagem incerta).")
            suggestions.append("Explicitar limitações e, quando possível, citar evidências.")
        if "fonte" in q.lower() and "http" not in a.lower():
            problems.append("Pedido de fontes não atendido.")
            suggestions.append("Incluir links ou referências verificáveis.")

        score = 1.0
        score -= 0.25 * len(problems)
        score = max(0.0, min(1.0, score))
        return {
            "score": float(score),
            "problems_found": problems,
            "improvement_suggestions": suggestions,
        }

    def review(self, *, goal: str, draft_answer: str, tool_trace: str) -> CriticResult:
        prompt = (
            "Você é um crítico rigoroso para um assistant.\n"
            "Avalie se a resposta atende o objetivo e se está fundamentada nas evidências.\n"
            "Se necessário, reescreva a resposta final de forma melhor.\n\n"
            f"Objetivo: {goal}\n\n"
            f"Rascunho: {draft_answer}\n\n"
            f"Evidências/trace:\n{tool_trace}\n\n"
            "Responda no formato:\n"
            "VEREDITO: <aprovado|reprovado>\n"
            "RESPOSTA_FINAL: <texto>\n"
        )
        result = self._router.generate(prompt, task="critic")
        text = result.text.strip()
        verdict = "aprovado" if "VEREDITO:" in text and "aprovado" in text.lower() else "reprovado"
        improved = text
        if "RESPOSTA_FINAL:" in text:
            improved = text.split("RESPOSTA_FINAL:", 1)[1].strip()
        return CriticResult(verdict=verdict, improved_answer=improved)
