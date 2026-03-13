"""Guardas para execução de ferramentas.

Inclui:
- limite de chamadas de tools por request
- validação de allowlist de tools (opcional)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool

from project.config.settings import Settings
from project.learning.tool_learning import ToolLearningStore
from project.services.metrics import MetricsService


@dataclass
class ToolGuardState:
    calls: int = 0
    question: str = ""


class ToolGuard:
    """Enforça políticas de execução de Tools."""

    def __init__(
        self,
        *,
        settings: Settings,
        metrics: MetricsService,
        tool_learning: ToolLearningStore | None = None,
    ) -> None:
        self._settings = settings
        self._metrics = metrics
        self._tool_learning = tool_learning
        self._state = ToolGuardState()

    def reset(self) -> None:
        self._state.calls = 0
        self._state.question = ""

    def set_context(self, *, question: str) -> None:
        self._state.question = str(question)

    def wrap(self, tool: BaseTool) -> BaseTool:
        return _GuardedTool(tool=tool, guard=self, metrics=self._metrics)

    def before_call(self, tool_name: str) -> None:
        if tool_name in set(self._settings.tool_blocklist or []):
            raise RuntimeError(f"Tool bloqueada por política: {tool_name}")
        allow = [t for t in (self._settings.tool_allowlist or []) if t]
        if allow and tool_name not in set(allow):
            raise RuntimeError(f"Tool não permitida: {tool_name}")
        self._state.calls += 1
        if self._state.calls > self._settings.max_tool_calls:
            raise RuntimeError(f"Limite de tools excedido ({self._settings.max_tool_calls}).")


class _GuardedTool(BaseTool):
    """Wrapper para instrumentar e limitar execução de uma Tool."""

    def __init__(self, *, tool: BaseTool, guard: ToolGuard, metrics: MetricsService):
        super().__init__(
            name=tool.name,
            description=tool.description,
            args_schema=getattr(tool, "args_schema", None),
        )
        self._tool = tool
        self._guard = guard
        self._metrics = metrics

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        self._guard.before_call(self._tool.name)
        timer_name = f"tool:{self._tool.name}"
        self._metrics.start_timer(timer_name)
        success = False
        try:
            out = self._tool._run(*args, **kwargs)
            success = True
            return out
        except Exception:
            raise
        finally:
            duration_ms = self._metrics.end_timer(timer_name)
            if self._guard._tool_learning is not None:
                self._guard._tool_learning.record(
                    question=self._guard._state.question,
                    tool_name=self._tool.name,
                    success=success,
                    latency_ms=duration_ms,
                )

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
