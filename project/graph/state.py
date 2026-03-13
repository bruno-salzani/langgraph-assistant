"""Definição do estado global do grafo LangGraph."""

from __future__ import annotations

from typing import Any, Annotated, TypedDict


def merge_metadata(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if a:
        out.update(a)
    if b:
        out.update(b)
    return out


class GraphState(TypedDict, total=False):
    """Estado global para o AI Operating System baseado em LangGraph."""

    # Input do usuário
    user_input: str

    # Classificação e intenção (pelo Guardrails)
    intent: str
    risk_level: str

    # Planejamento
    plan: list[str]
    plan_steps: list[dict[str, Any]]

    # Resultados de recuperação (RAG)
    retrieved_docs: list[Any]
    rag_answer: str
    web_results: list[Any]

    # Resultados de ferramentas
    tool_results: list[Any]
    intermediate_steps: list[Any]

    # Respostas e refinamentos
    draft_answer: str
    final_answer: str

    # Crítica e reflexão
    critique: str
    approved: bool

    # Controle e metadados
    metadata: Annotated[dict[str, Any], merge_metadata]
    error: str

    # Extras para compatibilidade (opcionais)
    session_id: str
    route: str
    branches: dict[str, bool]
