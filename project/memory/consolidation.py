from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any

from project.services.llm_router import ModelRouter


@dataclass(frozen=True)
class MemoryTriple:
    subject: str
    predicate: str
    object: str
    score: float
    kind: str


class SQLiteMemoryGraph:
    def __init__(self, *, db_path: Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_triples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    score REAL NOT NULL,
                    ts REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_triples_thread ON memory_triples(thread_id, ts DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_triples_spo ON memory_triples(thread_id, subject, predicate)"
            )
            conn.commit()

    def add_many(self, *, thread_id: str, triples: list[MemoryTriple]) -> None:
        if not triples:
            return
        rows = [
            (
                str(thread_id),
                t.kind,
                t.subject,
                t.predicate,
                t.object,
                float(t.score),
                float(time()),
            )
            for t in triples
            if t.subject and t.predicate and t.object
        ]
        if not rows:
            return
        with sqlite3.connect(self._path) as conn:
            conn.executemany(
                """
                INSERT INTO memory_triples(thread_id, kind, subject, predicate, object, score, ts)
                VALUES(?,?,?,?,?,?,?)
                """,
                rows,
            )
            conn.commit()

    def list_recent(self, *, thread_id: str, limit: int = 50) -> list[MemoryTriple]:
        with sqlite3.connect(self._path) as conn:
            cur = conn.execute(
                """
                SELECT kind, subject, predicate, object, score
                FROM memory_triples
                WHERE thread_id = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (str(thread_id), int(limit)),
            )
            rows = cur.fetchall()
        return [
            MemoryTriple(
                kind=str(r[0]),
                subject=str(r[1]),
                predicate=str(r[2]),
                object=str(r[3]),
                score=float(r[4]),
            )
            for r in rows
        ]


class MemoryExtractor:
    def __init__(self, *, router: ModelRouter) -> None:
        self._router = router

    def extract(self, *, thread_id: str, user_input: str, final_answer: str) -> list[MemoryTriple]:
        prompt = (
            "Extraia memórias úteis da interação como triplas (sujeito, predicado, objeto).\n"
            "Tipos de memória (kind): user_profile | episodic | semantic | tool_memory | system_learning.\n"
            "Responda APENAS com JSON no formato:\n"
            '{ "triples": [ { "kind": "...", "subject": "...", "predicate": "...", "object": "...", "confidence": 0.0 } ] }\n\n'
            f"thread_id: {thread_id}\n\n"
            f"Pergunta:\n{user_input}\n\n"
            f"Resposta final:\n{final_answer}\n"
        )
        raw = self._router.generate(prompt, task="learning").text.strip()
        try:
            payload = json.loads(raw)
        except Exception:
            return []
        triples_raw = payload.get("triples") or []
        out: list[MemoryTriple] = []
        for t in triples_raw:
            if not isinstance(t, dict):
                continue
            kind = str(t.get("kind") or "").strip() or "semantic"
            subj = str(t.get("subject") or "").strip()
            pred = str(t.get("predicate") or "").strip()
            obj = str(t.get("object") or "").strip()
            conf = t.get("confidence")
            try:
                score = float(conf)
            except Exception:
                score = 0.5
            score = max(0.0, min(1.0, score))
            if subj and pred and obj:
                out.append(MemoryTriple(subject=subj, predicate=pred, object=obj, score=score, kind=kind))
        return out


def format_triples(triples: list[MemoryTriple]) -> list[str]:
    rows: list[str] = []
    for t in triples:
        rows.append(f"{t.kind}: {t.subject} -> {t.predicate} -> {t.object} ({t.score:.2f})")
    return rows

