"""Agente auxiliar para síntese de pesquisa (web results -> resumo)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage


class ResearchAgent:
    """Resumo estruturado de resultados de pesquisa."""

    def __init__(self, llm):
        self._llm = llm

    def run(self, *, question: str, raw_results: str) -> str:
        prompt = (
            f"Pergunta: {question}\n\nResultados:\n{raw_results}\n\n"
            "Resumo curto com links e pontos verificáveis:"
        )
        response = self._llm.invoke(
            [
                SystemMessage(
                    content="Você sintetiza resultados de pesquisa em um resumo curto, com links e pontos verificáveis."
                ),
                HumanMessage(content=prompt),
            ]
        )
        return str(getattr(response, "content", "")).strip()
