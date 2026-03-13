"""Gera código Python para novos agentes a partir de uma spec."""

from __future__ import annotations

from dataclasses import dataclass

from project.agents.agent_factory.agent_designer import AgentSpec


@dataclass(frozen=True)
class GeneratedAgentCode:
    module_name: str
    class_name: str
    code: str


def _snake(name: str) -> str:
    out: list[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


class AgentGenerator:
    def generate_agent_code(self, agent_spec: AgentSpec) -> str:
        return self.generate(spec=agent_spec).code

    def generate(self, *, spec: AgentSpec) -> GeneratedAgentCode:
        module_name = f"{_snake(spec.name)}"
        class_name = spec.name
        tool_list = ", ".join(repr(t) for t in spec.tools)

        code = f"""from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentRunResult:
    answer: str
    trace: list[str]


class {class_name}:
    name: str = {class_name!r}
    goal: str = {spec.goal!r}
    tools: list[str] = [{tool_list}]

    def run(self, query: str, *, tools: dict[str, Any]) -> AgentRunResult:
        trace: list[str] = []
        parts: list[str] = []

        for tool_name in self.tools:
            tool = tools.get(tool_name)
            if tool is None:
                trace.append(f"Tool ausente: {{tool_name}}")
                continue
            try:
                if tool_name == "web_search":
                    out = tool.invoke({{"query": query, "max_results": 5}})
                elif tool_name == "web_scrape":
                    out = tool.invoke({{"url": query}})
                elif tool_name == "calculator":
                    out = tool.invoke({{"expression": query}})
                else:
                    out = tool.invoke({{"query": query}})
                trace.append(f"{{tool_name}}: ok")
                parts.append(str(out))
            except Exception as exc:
                trace.append(f"{{tool_name}}: erro: {{exc}}")

        answer = "\\n\\n".join(parts).strip() or "Sem saída das ferramentas."
        return AgentRunResult(answer=answer, trace=trace)
"""
        return GeneratedAgentCode(module_name=module_name, class_name=class_name, code=code)
