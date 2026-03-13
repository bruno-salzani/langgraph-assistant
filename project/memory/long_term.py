"""Memória de longo prazo persistente em SQLite."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path


class LongTermMemory:
    """Armazena histórico e aprendizados do sistema de forma persistente."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._setup()

    def _setup(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_input TEXT,
                    final_answer TEXT,
                    feedback TEXT,
                    failure_pattern TEXT,
                    timestamp DATETIME
                )
                """
            )

    def store_interaction(
        self, user_input: str, final_answer: str, feedback: str = None, failure: str = None
    ):
        """Salva a interação e metadados para aprendizado futuro."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO interactions (user_input, final_answer, feedback, failure_pattern, timestamp) VALUES (?, ?, ?, ?, ?)",
                (user_input, final_answer, feedback, failure, datetime.now().isoformat()),
            )

    def load_patterns(self, limit: int = 10):
        """Carrega padrões recentes de falha ou feedback para o nó de aprendizado."""
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT user_input, feedback, failure_pattern FROM interactions WHERE feedback IS NOT NULL OR failure_pattern IS NOT NULL ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            return cur.fetchall()
