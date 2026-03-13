"""Planner agent: transforma um objetivo em um plano executável."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from project.services.llm_router import ModelRouter


@dataclass(frozen=True)
class PlanStep:
    tool: str
    args: dict[str, Any]
    rationale: str


@dataclass(frozen=True)
class Plan:
    goal: str
    steps: list[PlanStep]


class PlannerAgent:
    """Cria um plano de execução baseado em ferramentas disponíveis."""

    def __init__(self, router: ModelRouter, *, tool_names: list[str]) -> None:
        self._router = router
        self._tool_names = tool_names

    def plan(self, goal: str) -> Plan:
        prompt = self._build_prompt(goal=goal, variant=None)
        return self._parse_plan(goal=goal, raw=self._generate(prompt, task="planning"))

    def plan_best(self, *, goal: str, n: int = 3) -> Plan:
        candidates: list[Plan] = []
        for i in range(max(1, n)):
            prompt = self._build_prompt(goal=goal, variant=i + 1)
            candidates.append(
                self._parse_plan(goal=goal, raw=self._generate(prompt, task="planning"))
            )

        if not candidates:
            return Plan(goal=goal, steps=[])

        scored = [(self._score_plan(goal=goal, plan=p), p) for p in candidates]
        _, best = max(scored, key=lambda x: x[0])
        return best

    def _build_prompt(self, *, goal: str, variant: int | None) -> str:
        skills: list[str] = []
        if "web_search" in self._tool_names:
            skills.append("skill:web_research")
        if "rag_lookup" in self._tool_names:
            skills.append("skill:doc_rag")

        parts: list[str] = [
            "Você é um planner para um sistema de agentes.\n"
            "Crie um plano com passos que usem ferramentas quando necessário.\n"
            "Ferramentas disponíveis:\n"
            f"{', '.join(self._tool_names)}\n\n"
        ]
        if skills:
            parts.append(f"Skills disponíveis:\n{', '.join(skills)}\n\n")
        parts.append("Responda APENAS com JSON no formato:\n")
        parts.append('{ "steps": [ { "tool": "nome", "args": { ... }, "rationale": "..." } ] }\n\n')
        header = "".join(parts)
        if variant is None:
            return header + f"Objetivo: {goal}"
        return header + f"Objetivo: {goal}\nVariante: {variant} (seja diferente e objetivo)"

    def _generate(self, prompt: str, *, task: str) -> str:
        try:
            return str(self._router.generate(prompt, task=task).text)
        except TypeError:
            return str(self._router.generate(prompt).text)

    def _parse_plan(self, *, goal: str, raw: str) -> Plan:
        text = (raw or "").strip()
        try:
            payload = json.loads(text)
        except Exception:
            return Plan(goal=goal, steps=[])
        steps = [
            PlanStep(
                tool=str(s.get("tool") or ""),
                args=dict(s.get("args") or {}),
                rationale=str(s.get("rationale") or ""),
            )
            for s in (payload.get("steps") or [])
            if isinstance(s, dict)
        ]
        steps = [s for s in steps if s.tool]
        return Plan(goal=goal, steps=steps)

    def _score_plan(self, *, goal: str, plan: Plan) -> float:
        q = (goal or "").lower()
        score = 0.0

        tools_used = [s.tool for s in plan.steps]
        unknown = [t for t in tools_used if t not in set(self._tool_names)]
        score -= 2.0 * len(unknown)

        if plan.steps:
            score += 1.0

        if len(plan.steps) <= 6:
            score += 0.5
        else:
            score -= 0.2 * (len(plan.steps) - 6)

        if any(
            k in q
            for k in ["document", "pdf", "documento", "repositorio", "repositório", "arquivo"]
        ):
            if "rag_lookup" in tools_used:
                score += 1.5
        if any(
            k in q
            for k in ["pesquise", "pesquisar", "internet", "web", "site", "notícia", "noticia"]
        ):
            if "web_search" in tools_used:
                score += 1.5
        if any(ch in q for ch in ["+", "-", "*", "/"]) and "calculator" in tools_used:
            score += 1.0

        return score
