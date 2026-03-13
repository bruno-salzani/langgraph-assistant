"""Aprendizado de ferramentas (tool learning) baseado em sucesso/falha."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from time import time


@dataclass(frozen=True)
class ToolStats:
    tool_name: str
    successes: int
    failures: int
    calls: int
    avg_latency_ms: float


class ToolLearningStore:
    def __init__(self, *, sqlite_path: Path) -> None:
        self._path = sqlite_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS tool_learning ("
                "tool_name TEXT PRIMARY KEY, "
                "successes INTEGER NOT NULL, "
                "failures INTEGER NOT NULL, "
                "calls INTEGER NOT NULL DEFAULT 0, "
                "total_latency_ms REAL NOT NULL DEFAULT 0.0"
                ")"
            )
            cols = {
                row[1]: row[2]
                for row in conn.execute("PRAGMA table_info(tool_learning)").fetchall()
            }
            if "calls" not in cols:
                conn.execute(
                    "ALTER TABLE tool_learning ADD COLUMN calls INTEGER NOT NULL DEFAULT 0"
                )
            if "total_latency_ms" not in cols:
                conn.execute(
                    "ALTER TABLE tool_learning ADD COLUMN total_latency_ms REAL NOT NULL DEFAULT 0.0"
                )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS tool_events ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "question TEXT NOT NULL, "
                "tool_name TEXT NOT NULL, "
                "success INTEGER NOT NULL, "
                "latency_ms REAL NOT NULL, "
                "ts REAL NOT NULL"
                ")"
            )
            conn.commit()

    def record(
        self,
        *,
        question: str = "",
        tool_name: str,
        success: bool,
        latency_ms: float | None = None,
    ) -> None:
        latency_ms = float(latency_ms or 0.0)
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                "INSERT INTO tool_learning(tool_name, successes, failures, calls, total_latency_ms) "
                "VALUES (?, 0, 0, 0, 0.0) "
                "ON CONFLICT(tool_name) DO NOTHING",
                (tool_name,),
            )
            conn.execute(
                (
                    "INSERT INTO tool_events(question, tool_name, success, latency_ms, ts) "
                    "VALUES (?, ?, ?, ?, ?)"
                ),
                (
                    str(question),
                    str(tool_name),
                    1 if success else 0,
                    latency_ms,
                    float(time()),
                ),
            )
            if success:
                conn.execute(
                    "UPDATE tool_learning SET successes = successes + 1 WHERE tool_name = ?",
                    (tool_name,),
                )
            else:
                conn.execute(
                    "UPDATE tool_learning SET failures = failures + 1 WHERE tool_name = ?",
                    (tool_name,),
                )
            conn.execute(
                "UPDATE tool_learning SET calls = calls + 1, total_latency_ms = total_latency_ms + ? "
                "WHERE tool_name = ?",
                (latency_ms, tool_name),
            )
            conn.commit()

    def list_stats(self) -> list[ToolStats]:
        with sqlite3.connect(self._path) as conn:
            cur = conn.execute(
                "SELECT tool_name, successes, failures, calls, total_latency_ms FROM tool_learning"
            )
            rows = cur.fetchall()
        out: list[ToolStats] = []
        for r in rows:
            calls = int(r[3])
            total_latency_ms = float(r[4] or 0.0)
            avg_latency_ms = total_latency_ms / calls if calls > 0 else 0.0
            out.append(
                ToolStats(
                    tool_name=str(r[0]),
                    successes=int(r[1]),
                    failures=int(r[2]),
                    calls=calls,
                    avg_latency_ms=avg_latency_ms,
                )
            )
        return out

    def priors(self) -> dict[str, float]:
        priors: dict[str, float] = {}
        for s in self.list_stats():
            ratio = math.log((s.successes + 1.0) / (s.failures + 1.0))
            latency_penalty = float(s.avg_latency_ms) / 5000.0
            priors[s.tool_name] = float(ratio - latency_penalty)
        return priors
