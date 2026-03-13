"""Loader de plugins para estender tools/prompts/agentes."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path


def load_plugins(*, plugins_dir: Path, package: str = "plugins") -> list[str]:
    if not plugins_dir.exists():
        return []

    loaded: list[str] = []
    for mod in pkgutil.iter_modules([str(plugins_dir)]):
        name = mod.name
        full = f"{package}.{name}"
        importlib.import_module(full)
        loaded.append(full)
    return loaded
