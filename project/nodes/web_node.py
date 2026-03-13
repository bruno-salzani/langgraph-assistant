from __future__ import annotations

from typing import Any

from project.graph.state import GraphState
from project.security.tool_guard import ToolGuard


def web_node(*, tools_by_name: dict[str, Any], tool_guard: ToolGuard | None = None):
    def _run(state: GraphState) -> GraphState:
        branches = state.get("branches") or {}
        if not bool(branches.get("web")):
            return {"metadata": {**(state.get("metadata") or {}), "web_skipped": True}}

        tool = tools_by_name.get("web_search")
        if tool is None:
            return {"web_results": [], "metadata": {**(state.get("metadata") or {}), "web_tool_missing": True}}

        q = str(state.get("user_input", "")).strip()
        if tool_guard is not None:
            tool_guard.set_context(question=q)

        out = tool.invoke({"query": q, "max_results": 5})
        return {"web_results": [out]}

    return _run

