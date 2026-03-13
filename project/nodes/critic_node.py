"""Nó de Crítica e Reflexão para validação de respostas."""

from __future__ import annotations

from project.agents.critic_agent import CriticAgent
from project.graph.state import GraphState


def critic_node(critic: CriticAgent):
    """Valida a resposta com critérios de factualidade e completude."""

    def _run(state: GraphState) -> GraphState:
        user_input = state.get("user_input", "")
        draft_answer = state.get("draft_answer", "")
        tool_results = str(state.get("tool_results", []))

        # O CriticAgent analisa a resposta
        cr = critic.review(goal=user_input, draft_answer=draft_answer, tool_trace=tool_results)

        return {
            "critique": cr.critique,
            "approved": cr.verdict == "aprovado",
            "final_answer": cr.improved_answer,
        }

    return _run
