"""Testes do Agent Factory (designer/generator/validator/registry)."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from project.agents.agent_factory.agent_designer import AgentDesigner, AgentSpec
from project.agents.agent_factory.agent_generator import AgentGenerator
from project.agents.agent_factory.agent_registry import instantiate_agent, register_generated_agent
from project.agents.agent_factory.agent_validator import AgentValidator


class _DummyTool:
    def __init__(self, name: str) -> None:
        self.name = name

    def invoke(self, args):
        return f"{self.name}:{args}"


def test_designer_creates_financial_agent_spec() -> None:
    designer = AgentDesigner()
    spec = designer.design(
        user_request="Quero dados financeiros atualizados de AAPL",
        available_tools=["web_search", "calculator"],
        existing_agents=[],
    )
    assert spec is not None
    assert spec.name == "FinancialDataAgent"


def test_generator_and_validator_and_registry(tmp_path: Path, dummy_settings) -> None:
    spec = AgentSpec(
        name="FinancialDataAgent",
        goal="Buscar dados financeiros.",
        tools=["web_search", "calculator"],
        skills=["análise"],
    )
    generated = AgentGenerator().generate(spec=spec)
    validation = AgentValidator().validate(
        spec=spec, code=generated.code, available_tools=["web_search", "calculator"]
    )
    assert validation.ok

    pkg_root = tmp_path / "pkg"
    (pkg_root / "tmp_pkg").mkdir(parents=True, exist_ok=True)
    (pkg_root / "tmp_pkg" / "__init__.py").write_text("", encoding="utf-8")
    out_dir = pkg_root / "tmp_pkg" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "__init__.py").write_text("", encoding="utf-8")

    sys.path.insert(0, str(pkg_root))
    try:
        entry = register_generated_agent(
            generated=generated,
            project_root=replace(dummy_settings, project_root=Path.cwd()).project_root,
            package="tmp_pkg.generated",
            target_dir=out_dir,
        )
        agent = instantiate_agent(entry)
        tools = {"web_search": _DummyTool("web_search"), "calculator": _DummyTool("calculator")}
        result = agent.run("2+2", tools=tools)
        assert "calculator" in result.answer or "web_search" in result.answer
    finally:
        sys.path.remove(str(pkg_root))
