from __future__ import annotations

from datetime import UTC, datetime

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from project.tools.registry import ToolContext, register_tool_factory


class _Input(BaseModel):
    pass


class TimeNowTool(BaseTool):
    name: str = "time_now"
    description: str = "Retorna o timestamp atual em UTC (útil para relatórios)."
    args_schema: type[BaseModel] = _Input

    def _run(self, **kwargs) -> str:
        return datetime.now(UTC).isoformat()


def _factory(ctx: ToolContext) -> TimeNowTool:
    return TimeNowTool()


register_tool_factory("time_now", _factory)
