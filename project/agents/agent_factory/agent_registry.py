"""Registro e carregamento de agentes gerados dinamicamente."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from project.agents.agent_factory.agent_generator import GeneratedAgentCode


@dataclass(frozen=True)
class RegisteredAgent:
    name: str
    module: str
    class_name: str


_REGISTRY: dict[str, RegisteredAgent] = {}


def register_agent(agent_class: type) -> RegisteredAgent:
    name = str(getattr(agent_class, "__name__", "")).strip() or "UnnamedAgent"
    module = str(getattr(agent_class, "__module__", "")).strip()
    entry = RegisteredAgent(name=name, module=module, class_name=name)
    _REGISTRY[entry.name] = entry
    return entry


def register_generated_agent(
    *,
    generated: GeneratedAgentCode,
    project_root: Path,
    package: str = "project.agents.generated",
    target_dir: Path | None = None,
) -> RegisteredAgent:
    out_dir = target_dir or (project_root / "project" / "agents" / "generated")
    out_dir.mkdir(parents=True, exist_ok=True)
    init_file = out_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("from __future__ import annotations\n", encoding="utf-8")

    file_path = out_dir / f"{generated.module_name}.py"
    file_path.write_text(generated.code, encoding="utf-8")

    module_path = f"{package}.{generated.module_name}"
    mod = importlib.import_module(module_path)
    importlib.reload(mod)

    agent_class = getattr(mod, generated.class_name)
    return register_agent(agent_class)


def get_registered_agent(name: str) -> RegisteredAgent | None:
    return _REGISTRY.get(name)


def list_registered_agents() -> list[RegisteredAgent]:
    return list(_REGISTRY.values())


def list_agents() -> list[RegisteredAgent]:
    return list_registered_agents()


def load_agents(
    *,
    project_root: Path,
    package: str = "project.agents.generated",
    target_dir: Path | None = None,
) -> list[RegisteredAgent]:
    out_dir = target_dir or (project_root / "project" / "agents" / "generated")
    if not out_dir.exists():
        return []

    loaded: list[RegisteredAgent] = []
    for path in sorted(out_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        module_path = f"{package}.{path.stem}"
        try:
            mod = importlib.import_module(module_path)
            importlib.reload(mod)
        except Exception:
            continue

        for _, obj in vars(mod).items():
            if not inspect.isclass(obj):
                continue
            if getattr(obj, "__module__", "") != mod.__name__:
                continue
            if not callable(getattr(obj, "run", None)):
                continue
            loaded.append(register_agent(obj))
    return loaded


def instantiate_agent(entry: RegisteredAgent) -> Any:
    mod: ModuleType = importlib.import_module(entry.module)
    cls = getattr(mod, entry.class_name)
    return cls()
