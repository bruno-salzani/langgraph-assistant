"""Nó de aprovação humana para ações críticas."""

from __future__ import annotations

from project.config.settings import Settings
from project.graph.state import GraphState


def approval_node(settings: Settings):
    """Ponto de interrupção para aprovação humana em ações críticas."""

    def _run(state: GraphState) -> GraphState:
        branches = state.get("branches") or {}
        if not bool(branches.get("tools")):
            return {"approved": True, "metadata": {**(state.get("metadata") or {}), "tools_disabled": True}}

        if not settings.human_in_loop_enabled:
            return {"approved": True}

        plan_steps = state.get("plan_steps") or []
        critical_tools = ["read_file", "web_scrape", "calculator"]  # Exemplo

        is_critical = any(
            str(s.get("tool") or "") in critical_tools for s in plan_steps if isinstance(s, dict)
        )

        if is_critical:
            # No LangGraph, isso seria um breakpoint.
            # Aqui simulamos pedindo aprovação se não estiver aprovado.
            if not state.get("approved"):
                return {
                    "approved": False,
                    "metadata": {**(state.get("metadata") or {}), "waiting_for_approval": True},
                }

        return {"approved": True}

    return _run
