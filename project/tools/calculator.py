"""Tool de calculadora segura (sem eval)."""

from __future__ import annotations

import ast
import operator as op
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from project.tools.registry import ToolContext, register_tool_factory

_OPS: dict[type[ast.AST], Any] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


class CalculatorInput(BaseModel):
    """Input schema para a calculadora."""

    expression: str = Field(..., description="Expressão matemática, ex: (2+3)*4/5")


def _eval_expr(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_expr(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)

    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _eval_expr(node.left)
        right = _eval_expr(node.right)
        return float(_OPS[type(node.op)](left, right))

    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        operand = _eval_expr(node.operand)
        return float(_OPS[type(node.op)](operand))

    raise ValueError("Expressão inválida ou insegura.")


class CalculatorTool(BaseTool):
    """Avalia expressões matemáticas permitidas via AST."""

    name: str = "calculator"
    description: str = (
        "Resolve expressões matemáticas com segurança "
        "(apenas +, -, *, /, **, %, // e parênteses)."
    )
    args_schema: type[BaseModel] = CalculatorInput

    def _run(self, expression: str, **kwargs) -> str:
        expr = expression.strip()
        parsed = ast.parse(expr, mode="eval")
        result = _eval_expr(parsed)
        return str(result)


def _factory(ctx: ToolContext) -> CalculatorTool:
    return CalculatorTool()


register_tool_factory("calculator", _factory)
