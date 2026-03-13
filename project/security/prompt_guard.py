"""Proteções básicas contra prompt injection e entradas maliciosas."""

from __future__ import annotations

import re

from project.config.settings import Settings


class PromptGuard:
    """Valida e sanitiza entradas do usuário antes de enviar ao agent."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def sanitize(self, user_input: str) -> str:
        text = (user_input or "").strip()
        if not text:
            raise ValueError("Entrada vazia.")
        if len(text) > self._settings.max_input_chars:
            text = text[: self._settings.max_input_chars]
        return text

    def detect_prompt_injection(self, user_input: str) -> bool:
        text = user_input.lower()
        patterns = [
            r"ignore\s+previous\s+instructions",
            r"ignore\s+all\s+instructions",
            r"reveal\s+system\s+prompt",
            r"show\s+your\s+system\s+prompt",
            r"print\s+the\s+prompt",
            r"developer\s+message",
            r"system\s+message",
            r"you\s+are\s+now\s+in\s+developer\s+mode",
        ]
        return any(re.search(p, text) for p in patterns)

    def enforce(self, user_input: str) -> str:
        sanitized = self.sanitize(user_input)
        if self.detect_prompt_injection(sanitized):
            raise ValueError(
                "Entrada bloqueada por suspeita de prompt injection. Reformule sua solicitação sem instruções para ignorar regras."
            )
        return sanitized
