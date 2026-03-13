"""Testes de prompt versioning."""

from __future__ import annotations

from pathlib import Path

from project.prompts.system_prompt import load_system_prompt


def test_load_system_prompt_reads_versioned_file() -> None:
    prompts_dir = Path(__file__).resolve().parents[1] / "project" / "prompts"
    text = load_system_prompt(prompts_dir=prompts_dir, version="v1")
    assert "Regras:" in text


def test_load_system_prompt_fallback() -> None:
    prompts_dir = Path(__file__).resolve().parents[1] / "project" / "prompts"
    text = load_system_prompt(prompts_dir=prompts_dir, version="v999")
    assert "AI Assistant" in text
