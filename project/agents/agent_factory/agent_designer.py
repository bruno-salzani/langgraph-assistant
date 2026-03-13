"""Projeta um novo agente quando faltam capacidades."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentSpec:
    name: str
    goal: str
    tools: list[str]
    skills: list[str]


class AgentDesigner:
    def analyze_capability_gap(
        self,
        task: str,
        available_agents: list[str],
        available_tools: list[str] | None = None,
    ) -> AgentSpec | None:
        if available_tools is None:
            return None
        return self.design(
            user_request=task,
            available_tools=available_tools,
            existing_agents=available_agents,
        )

    def design(
        self,
        *,
        user_request: str,
        available_tools: list[str],
        existing_agents: list[str],
    ) -> AgentSpec | None:
        text = user_request.lower()

        if "financial" in text or "finance" in text or "ações" in text or "bolsa" in text:
            name = "FinancialDataAgent"
            if name in existing_agents:
                return None
            tools = [t for t in ["web_search", "web_scrape", "calculator"] if t in available_tools]
            if not tools:
                return None
            return AgentSpec(
                name=name,
                goal="Buscar dados financeiros atualizados e resumir os principais indicadores.",
                tools=tools,
                skills=["parsing de dados", "análise de números", "sumarização objetiva"],
            )

        if "paper" in text or "artigo" in text or "pesquisa" in text or "estado da arte" in text:
            name = "AutonomousResearchAgent"
            if name in existing_agents:
                return None
            tools = [t for t in ["web_search", "web_scrape"] if t in available_tools]
            if not tools:
                return None
            return AgentSpec(
                name=name,
                goal="Pesquisar referências, extrair pontos-chave e gerar um relatório com links.",
                tools=tools,
                skills=["pesquisa", "leitura", "síntese", "organização de relatório"],
            )

        return None
