"""Nó de Planejamento para decompor tarefas complexas."""

from __future__ import annotations

from project.agents.planner_agent import PlannerAgent
from project.config.settings import Settings
from project.graph.state import GraphState


def planner_node(planner: PlannerAgent, settings: Settings):
    """Decompõe tarefas em passos lógicos."""

    def _run(state: GraphState) -> GraphState:
        user_input = state.get("user_input", "")
        if settings.tree_of_thought_enabled:
            plan = planner.plan_best(goal=str(user_input), n=3)
        else:
            plan = planner.plan(str(user_input))

        steps = [f"{s.tool}({s.args}): {s.rationale}" for s in plan.steps]
        structured = [
            {"tool": s.tool, "args": dict(s.args), "rationale": s.rationale} for s in plan.steps
        ]

        return {
            "plan": steps,
            "plan_steps": structured,
            "metadata": {**(state.get("metadata") or {}), "plan_generated": True},
        }

    return _run
