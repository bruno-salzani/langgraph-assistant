"""Testes de agentes auxiliares (planner/executor/critic) sem chamadas externas."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import tool

from project.agents.executor_agent import ExecutorAgent
from project.agents.planner_agent import PlannerAgent
from project.services.llm_router import GenerationResult


@dataclass(frozen=True)
class _FakeRouter:
    response: str

    def generate(self, prompt: str) -> GenerationResult:
        return GenerationResult(model="fake", text=self.response)


@tool
def echo(text: str) -> str:
    """Echo tool."""
    return text


def test_planner_agent_parses_plan() -> None:
    router = _FakeRouter(
        response='{"steps":[{"tool":"echo","args":{"text":"ok"},"rationale":"test"}]}'
    )
    planner = PlannerAgent(router, tool_names=["echo"])
    plan = planner.plan("say ok")
    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "echo"


def test_executor_agent_runs_tool() -> None:
    executor = ExecutorAgent(tools=[echo])
    router = _FakeRouter(
        response='{"steps":[{"tool":"echo","args":{"text":"hello"},"rationale":"test"}]}'
    )
    plan = PlannerAgent(router, tool_names=["echo"]).plan("say hello")
    result = executor.execute(plan)
    assert "hello" in result.trace[0].output
