"""Nó de Execução de ferramentas."""

from __future__ import annotations

from project.agents.executor_agent import ExecutorAgent
from project.agents.planner_agent import Plan, PlanStep
from project.graph.state import GraphState


def executor_node(executor: ExecutorAgent):
    """Executa as ferramentas conforme planejado."""

    def _run(state: GraphState) -> GraphState:
        branches = state.get("branches") or {}
        if not bool(branches.get("tools")):
            return {"metadata": {**(state.get("metadata") or {}), "tools_skipped": True}}

        user_input = state.get("user_input", "")
        structured = state.get("plan_steps") or []

        def expand(step: PlanStep) -> list[PlanStep]:
            tool = (step.tool or "").strip()
            if not tool.startswith("skill:"):
                return [step]
            name = tool.split(":", 1)[1].strip().lower()
            if name == "web_research":
                return [
                    PlanStep(
                        tool="web_search",
                        args={"query": user_input, "max_results": 5},
                        rationale=step.rationale,
                    )
                ]
            if name == "doc_rag":
                return [
                    PlanStep(
                        tool="rag_lookup",
                        args={"question": user_input},
                        rationale=step.rationale,
                    )
                ]
            return []

        steps: list[PlanStep] = []
        for item in structured:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool") or "").strip()
            if not tool:
                continue
            args = item.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            rationale = str(item.get("rationale") or "")
            steps.extend(expand(PlanStep(tool=tool, args=dict(args), rationale=rationale)))

        plan = Plan(goal=user_input, steps=steps)

        attempts = 2
        for i in range(attempts):
            try:
                execution = executor.execute(plan)
                results = [f"- {t.step.tool}({t.step.args}): {t.output}" for t in execution.trace]
                return {"tool_results": results, "intermediate_steps": results, "error": ""}
            except Exception as exc:
                if i < attempts - 1:
                    continue
                return {"error": f"Executor: Falha após {attempts} tentativas: {exc}"}

        return {}

    return _run
