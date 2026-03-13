"""Testes de registry de tools e plugins."""

from __future__ import annotations

from pathlib import Path

from plugins.loader import load_plugins
from project.tools.registry import registered_tool_names


def test_tool_registry_has_default_tools() -> None:
    import project.tools.calculator  # noqa: F401
    import project.tools.data_lookup_tool  # noqa: F401
    import project.tools.file_reader  # noqa: F401
    import project.tools.scraper  # noqa: F401
    import project.tools.web_search  # noqa: F401

    names = registered_tool_names()
    assert "calculator" in names
    assert "web_search" in names
    assert "web_scrape" in names
    assert "read_file" in names
    assert "rag_lookup" in names


def test_plugin_loader_registers_tool(tmp_path: Path) -> None:
    import project.tools.registry  # noqa: F401

    plugins_dir = Path(__file__).resolve().parents[1] / "plugins"
    loaded = load_plugins(plugins_dir=plugins_dir)
    assert loaded
    assert "time_now" in registered_tool_names()
