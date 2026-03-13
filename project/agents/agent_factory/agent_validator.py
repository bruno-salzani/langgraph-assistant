"""Valida código gerado antes de registrar um novo agente."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from project.agents.agent_factory.agent_designer import AgentSpec


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str = ""


_BANNED_CALLS = {"eval", "exec", "compile", "open", "__import__"}


class AgentValidator:
    def validate(
        self,
        code: str,
        *,
        spec: AgentSpec | None = None,
        available_tools: list[str] | None = None,
    ) -> ValidationResult:
        if spec is not None and available_tools is not None:
            missing = [t for t in spec.tools if t not in available_tools]
            if missing:
                return ValidationResult(ok=False, reason=f"Tools não disponíveis: {missing}")

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return ValidationResult(ok=False, reason=f"SyntaxError: {exc}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in {"typing", "dataclasses"}:
                        return ValidationResult(ok=False, reason=f"Import proibido: {alias.name}")
            if isinstance(node, ast.ImportFrom):
                if node.module not in {"__future__", "typing", "dataclasses"}:
                    return ValidationResult(ok=False, reason=f"ImportFrom proibido: {node.module}")
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id in _BANNED_CALLS:
                    return ValidationResult(ok=False, reason=f"Chamada proibida: {fn.id}")
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                return ValidationResult(ok=False, reason="Acesso a dunder proibido.")

        return ValidationResult(ok=True)
