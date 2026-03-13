"""Executor agent: executa um plano chamando ferramentas."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import BaseTool

from project.agents.planner_agent import Plan, PlanStep


@dataclass(frozen=True)
class ToolExecution:
    step: PlanStep
    output: str


@dataclass(frozen=True)
class ExecutionResult:
    goal: str
    trace: list[ToolExecution]


class ExecutorAgent:
    """Executa ferramentas de forma determinística conforme um plano."""

    def __init__(self, *, tools: list[BaseTool]) -> None:
        self._tools_by_name: dict[str, BaseTool] = {t.name: t for t in tools}

    def run(self, question: str) -> ExecutionResult:
        q = (question or "").strip()
        steps: list[PlanStep] = []
        if "calculator" in self._tools_by_name and any(ch in q for ch in "+-*/"):
            steps.append(
                PlanStep(tool="calculator", args={"expression": q}, rationale="cálculo direto")
            )
        elif "web_scrape" in self._tools_by_name and q.startswith("http"):
            steps.append(PlanStep(tool="web_scrape", args={"url": q}, rationale="ler URL"))
        elif "web_search" in self._tools_by_name:
            steps.append(PlanStep(tool="web_search", args={"query": q}, rationale="buscar na web"))

        return self.execute(Plan(goal=q or "responder pergunta", steps=steps))

    def execute(self, plan: Plan) -> ExecutionResult:
        trace: list[ToolExecution] = []
        for step in plan.steps:
            tool = self._tools_by_name.get(step.tool)
            if tool is None:
                trace.append(ToolExecution(step=step, output=f"Tool não encontrada: {step.tool}"))
                continue
            try:
                output = tool.invoke(step.args)
            except Exception as exc:
                output = f"Erro ao executar tool {step.tool}: {exc}"
            trace.append(ToolExecution(step=step, output=str(output)))
        return ExecutionResult(goal=plan.goal, trace=trace)
