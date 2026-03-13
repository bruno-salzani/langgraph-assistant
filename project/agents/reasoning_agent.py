"""Agente auxiliar para raciocínio direto (sem tool calling)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage


class ReasoningAgent:
    """Encapsula raciocínio direto via LLM (sem chains lineares)."""

    def __init__(self, llm):
        self._llm = llm

    def run(self, user_input: str) -> str:
        response = self._llm.invoke(
            [
                SystemMessage(
                    content="Você resolve problemas complexos de forma objetiva e verificável."
                ),
                HumanMessage(content=str(user_input)),
            ]
        )
        return str(getattr(response, "content", "")).strip()
