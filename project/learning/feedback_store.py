"""Armazenamento de feedback do usuário."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Literal


@dataclass(frozen=True)
class FeedbackEntry:
    question: str
    answer: str
    rating: int
    comments: str
    ts: float


class FeedbackStore:
    def add(self, *, question: str, answer: str, rating: int, comments: str) -> None:
        raise NotImplementedError

    def list_recent(self, *, limit: int = 100) -> list[FeedbackEntry]:
        raise NotImplementedError


class SQLiteFeedbackStore(FeedbackStore):
    def __init__(self, *, sqlite_path: Path) -> None:
        self._path = sqlite_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS feedback ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "question TEXT NOT NULL, "
                "answer TEXT NOT NULL, "
                "rating INTEGER NOT NULL, "
                "comments TEXT NOT NULL, "
                "ts REAL NOT NULL"
                ")"
            )
            conn.commit()

    def add(self, *, question: str, answer: str, rating: int, comments: str) -> None:
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                "INSERT INTO feedback(question, answer, rating, comments, ts) VALUES (?, ?, ?, ?, ?)",
                (question, answer, int(rating), comments, float(time())),
            )
            conn.commit()

    def list_recent(self, *, limit: int = 100) -> list[FeedbackEntry]:
        with sqlite3.connect(self._path) as conn:
            cur = conn.execute(
                "SELECT question, answer, rating, comments, ts FROM feedback ORDER BY id DESC LIMIT ?",
                (int(limit),),
            )
            rows = cur.fetchall()
        return [
            FeedbackEntry(
                question=str(r[0]),
                answer=str(r[1]),
                rating=int(r[2]),
                comments=str(r[3]),
                ts=float(r[4]),
            )
            for r in rows
        ]


class PostgresFeedbackStore(FeedbackStore):
    def __init__(self, *, postgres_url: str) -> None:
        import psycopg

        self._psycopg = psycopg
        self._url = postgres_url
        self._init_db()

    def _init_db(self) -> None:
        with self._psycopg.connect(self._url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS feedback ("
                    "id BIGSERIAL PRIMARY KEY, "
                    "question TEXT NOT NULL, "
                    "answer TEXT NOT NULL, "
                    "rating INTEGER NOT NULL, "
                    "comments TEXT NOT NULL, "
                    "ts DOUBLE PRECISION NOT NULL"
                    ")"
                )
            conn.commit()

    def add(self, *, question: str, answer: str, rating: int, comments: str) -> None:
        with self._psycopg.connect(self._url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO feedback(question, answer, rating, comments, ts) VALUES (%s, %s, %s, %s, %s)",
                    (question, answer, int(rating), comments, float(time())),
                )
            conn.commit()

    def list_recent(self, *, limit: int = 100) -> list[FeedbackEntry]:
        with self._psycopg.connect(self._url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT question, answer, rating, comments, ts FROM feedback ORDER BY id DESC LIMIT %s",
                    (int(limit),),
                )
                rows = cur.fetchall()
        return [
            FeedbackEntry(
                question=str(r[0]),
                answer=str(r[1]),
                rating=int(r[2]),
                comments=str(r[3]),
                ts=float(r[4]),
            )
            for r in rows
        ]


def build_feedback_store(
    *, backend: Literal["sqlite", "postgres"], sqlite_path: Path, postgres_url: str
):
    if backend == "postgres":
        return PostgresFeedbackStore(postgres_url=postgres_url)
    return SQLiteFeedbackStore(sqlite_path=sqlite_path)
