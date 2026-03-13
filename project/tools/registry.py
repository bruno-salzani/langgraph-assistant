"""Registro dinâmico de tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolContext:
    settings: Any
    rag_llm: Any
    logger: Any
    metrics: Any
    router: Any


ToolFactory = Callable[[ToolContext], Any]

_FACTORIES: dict[str, ToolFactory] = {}


def register_tool_factory(name: str, factory: ToolFactory) -> None:
    _FACTORIES[name] = factory


def build_registered_tools(ctx: ToolContext) -> list[Any]:
    tools: list[Any] = []
    for _, factory in sorted(_FACTORIES.items(), key=lambda t: t[0]):
        tools.append(factory(ctx))
    return tools


def registered_tool_names() -> list[str]:
    return sorted(_FACTORIES.keys())
