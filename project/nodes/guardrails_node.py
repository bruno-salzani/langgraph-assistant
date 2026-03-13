"""Nó de Guardrails para sanitização e segurança."""

from __future__ import annotations

from project.graph.state import GraphState
from project.security.prompt_guard import PromptGuard
from project.security.tool_guard import ToolGuard


def guardrails_node(guard: PromptGuard, tool_guard: ToolGuard | None = None):
    """Sanitiza o input e detecta riscos."""

    def _run(state: GraphState) -> GraphState:
        user_input = state.get("user_input", "")
        # Em uma implementação real, poderíamos classificar intent e risk_level aqui
        sanitized = guard.enforce(user_input)

        if tool_guard is not None:
            tool_guard.reset()
            tool_guard.set_context(question=sanitized)

        return {
            "user_input": sanitized,
            "intent": "general_query",  # Simplificado para este exemplo
            "risk_level": "low",
            "metadata": {**(state.get("metadata") or {}), "guardrails_checked": True},
        }

    return _run
