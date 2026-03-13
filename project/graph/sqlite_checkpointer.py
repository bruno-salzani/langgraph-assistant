from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)


class SqliteCheckpointSaver(BaseCheckpointSaver[str]):
    def __init__(self, *, path: Path) -> None:
        super().__init__()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._init_schema()

    @classmethod
    def from_path(cls, path: Path) -> SqliteCheckpointSaver:
        return cls(path=path)

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                checkpoint_type TEXT NOT NULL,
                checkpoint_blob BLOB NOT NULL,
                metadata_type TEXT NOT NULL,
                metadata_blob BLOB NOT NULL,
                parent_checkpoint_id TEXT,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS blobs (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL,
                channel TEXT NOT NULL,
                version TEXT NOT NULL,
                blob_type TEXT NOT NULL,
                blob BLOB NOT NULL,
                PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                channel TEXT NOT NULL,
                value_type TEXT NOT NULL,
                value_blob BLOB NOT NULL,
                task_path TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_checkpoints_lookup ON checkpoints(thread_id, checkpoint_ns, checkpoint_id)"
        )
        self._conn.commit()

    def _load_blobs(
        self, *, thread_id: str, checkpoint_ns: str, versions: ChannelVersions
    ) -> dict[str, Any]:
        channel_values: dict[str, Any] = {}
        if not versions:
            return channel_values
        cur = self._conn.cursor()
        for channel, version in versions.items():
            row = cur.execute(
                """
                SELECT blob_type, blob
                FROM blobs
                WHERE thread_id=? AND checkpoint_ns=? AND channel=? AND version=?
                """,
                (thread_id, checkpoint_ns, channel, str(version)),
            ).fetchone()
            if row is None:
                continue
            blob_type, blob = row
            if blob_type != "empty":
                channel_values[channel] = self.serde.loads_typed((blob_type, blob))
        return channel_values

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        requested = get_checkpoint_id(config)
        cur = self._conn.cursor()

        if requested:
            row = cur.execute(
                """
                SELECT checkpoint_type, checkpoint_blob, metadata_type, metadata_blob, parent_checkpoint_id
                FROM checkpoints
                WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?
                """,
                (thread_id, checkpoint_ns, requested),
            ).fetchone()
            if row is None:
                return None
            checkpoint_type, checkpoint_blob, metadata_type, metadata_blob, parent_checkpoint_id = (
                row
            )
            checkpoint: Checkpoint = self.serde.loads_typed((checkpoint_type, checkpoint_blob))
            metadata: CheckpointMetadata = self.serde.loads_typed((metadata_type, metadata_blob))
            writes_rows = cur.execute(
                """
                SELECT task_id, channel, value_type, value_blob
                FROM writes
                WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?
                ORDER BY idx ASC
                """,
                (thread_id, checkpoint_ns, requested),
            ).fetchall()
            return CheckpointTuple(
                config=config,
                checkpoint={
                    **checkpoint,
                    "channel_values": self._load_blobs(
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        versions=checkpoint["channel_versions"],
                    ),
                },
                metadata=metadata,
                pending_writes=[
                    (task_id, channel, self.serde.loads_typed((value_type, value_blob)))
                    for task_id, channel, value_type, value_blob in writes_rows
                ],
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_checkpoint_id,
                        }
                    }
                    if parent_checkpoint_id
                    else None
                ),
            )

        row = cur.execute(
            """
            SELECT checkpoint_id, checkpoint_type, checkpoint_blob, metadata_type, metadata_blob, parent_checkpoint_id
            FROM checkpoints
            WHERE thread_id=? AND checkpoint_ns=?
            ORDER BY checkpoint_id DESC
            LIMIT 1
            """,
            (thread_id, checkpoint_ns),
        ).fetchone()
        if row is None:
            return None
        (
            checkpoint_id,
            checkpoint_type,
            checkpoint_blob,
            metadata_type,
            metadata_blob,
            parent_checkpoint_id,
        ) = row
        checkpoint = self.serde.loads_typed((checkpoint_type, checkpoint_blob))
        metadata = self.serde.loads_typed((metadata_type, metadata_blob))
        writes_rows = cur.execute(
            """
            SELECT task_id, channel, value_type, value_blob
            FROM writes
            WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?
            ORDER BY idx ASC
            """,
            (thread_id, checkpoint_ns, checkpoint_id),
        ).fetchall()
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint={
                **checkpoint,
                "channel_values": self._load_blobs(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    versions=checkpoint["channel_versions"],
                ),
            },
            metadata=metadata,
            pending_writes=[
                (task_id, channel, self.serde.loads_typed((value_type, value_blob)))
                for task_id, channel, value_type, value_blob in writes_rows
            ],
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
                if parent_checkpoint_id
                else None
            ),
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        cur = self._conn.cursor()
        params: list[Any] = []
        where: list[str] = []

        if config is not None:
            where.append("thread_id=?")
            params.append(config["configurable"]["thread_id"])
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                where.append("checkpoint_ns=?")
                params.append(checkpoint_ns)
            config_checkpoint_id = get_checkpoint_id(config)
            if config_checkpoint_id:
                where.append("checkpoint_id=?")
                params.append(config_checkpoint_id)

        if before is not None and (before_id := get_checkpoint_id(before)):
            where.append("checkpoint_id < ?")
            params.append(before_id)

        sql = (
            "SELECT thread_id, checkpoint_ns, checkpoint_id, checkpoint_type, checkpoint_blob, metadata_type, metadata_blob, parent_checkpoint_id "
            "FROM checkpoints "
        )
        if where:
            sql += "WHERE " + " AND ".join(where) + " "
        sql += "ORDER BY checkpoint_id DESC "
        if limit is not None:
            sql += "LIMIT ?"
            params.append(limit)

        for (
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            checkpoint_type,
            checkpoint_blob,
            metadata_type,
            metadata_blob,
            parent_checkpoint_id,
        ) in cur.execute(sql, params):
            metadata = self.serde.loads_typed((metadata_type, metadata_blob))
            if filter and not all(metadata.get(k) == v for k, v in filter.items()):
                continue
            checkpoint = self.serde.loads_typed((checkpoint_type, checkpoint_blob))
            writes_rows = cur.execute(
                """
                SELECT task_id, channel, value_type, value_blob
                FROM writes
                WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?
                ORDER BY idx ASC
                """,
                (thread_id, checkpoint_ns, checkpoint_id),
            ).fetchall()
            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                },
                checkpoint={
                    **checkpoint,
                    "channel_values": self._load_blobs(
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        versions=checkpoint["channel_versions"],
                    ),
                },
                metadata=metadata,
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_checkpoint_id,
                        }
                    }
                    if parent_checkpoint_id
                    else None
                ),
                pending_writes=[
                    (task_id, channel, self.serde.loads_typed((value_type, value_blob)))
                    for task_id, channel, value_type, value_blob in writes_rows
                ],
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        with self._lock:
            c = checkpoint.copy()
            thread_id: str = config["configurable"]["thread_id"]
            checkpoint_ns: str = config["configurable"]["checkpoint_ns"]
            parent_checkpoint_id: str | None = config["configurable"].get("checkpoint_id")
            values: dict[str, Any] = c.pop("channel_values")  # type: ignore[misc]

            cur = self._conn.cursor()
            for channel, version in new_versions.items():
                if channel in values:
                    vtype, vblob = self.serde.dumps_typed(values[channel])
                else:
                    vtype, vblob = ("empty", b"")
                cur.execute(
                    """
                    INSERT OR REPLACE INTO blobs(thread_id, checkpoint_ns, channel, version, blob_type, blob)
                    VALUES(?,?,?,?,?,?)
                    """,
                    (thread_id, checkpoint_ns, channel, str(version), vtype, vblob),
                )

            checkpoint_type, checkpoint_blob = self.serde.dumps_typed(c)
            metadata_type, metadata_blob = self.serde.dumps_typed(
                get_checkpoint_metadata(config, metadata)
            )
            cur.execute(
                """
                INSERT OR REPLACE INTO checkpoints(
                    thread_id, checkpoint_ns, checkpoint_id,
                    checkpoint_type, checkpoint_blob,
                    metadata_type, metadata_blob,
                    parent_checkpoint_id
                )
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    checkpoint_type,
                    checkpoint_blob,
                    metadata_type,
                    metadata_blob,
                    parent_checkpoint_id,
                ),
            )
            self._conn.commit()
            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint["id"],
                }
            }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        with self._lock:
            thread_id: str = config["configurable"]["thread_id"]
            checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
            checkpoint_id: str = config["configurable"]["checkpoint_id"]
            cur = self._conn.cursor()

            for idx, (channel, value) in enumerate(writes):
                mapped_idx = WRITES_IDX_MAP.get(channel, idx)
                value_type, value_blob = self.serde.dumps_typed(value)
                if mapped_idx >= 0:
                    cur.execute(
                        """
                        INSERT OR IGNORE INTO writes(
                            thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
                            channel, value_type, value_blob, task_path
                        )
                        VALUES(?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            thread_id,
                            checkpoint_ns,
                            checkpoint_id,
                            task_id,
                            int(mapped_idx),
                            channel,
                            value_type,
                            value_blob,
                            task_path,
                        ),
                    )
                else:
                    cur.execute(
                        """
                        INSERT OR REPLACE INTO writes(
                            thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
                            channel, value_type, value_blob, task_path
                        )
                        VALUES(?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            thread_id,
                            checkpoint_ns,
                            checkpoint_id,
                            task_id,
                            int(mapped_idx),
                            channel,
                            value_type,
                            value_blob,
                            task_path,
                        ),
                    )

            self._conn.commit()

    def delete_thread(self, thread_id: str) -> None:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM writes WHERE thread_id=?", (thread_id,))
        cur.execute("DELETE FROM blobs WHERE thread_id=?", (thread_id,))
        cur.execute("DELETE FROM checkpoints WHERE thread_id=?", (thread_id,))
        self._conn.commit()

    def list_checkpoint_ids(self, *, thread_id: str, checkpoint_ns: str = "", limit: int = 50) -> list[str]:
        cur = self._conn.cursor()
        rows = cur.execute(
            """
            SELECT checkpoint_id
            FROM checkpoints
            WHERE thread_id=? AND checkpoint_ns=?
            ORDER BY checkpoint_id DESC
            LIMIT ?
            """,
            (thread_id, checkpoint_ns, int(limit)),
        ).fetchall()
        return [str(r[0]) for r in rows]

    def fork_thread(
        self,
        *,
        from_thread_id: str,
        to_thread_id: str,
        checkpoint_id: str,
        checkpoint_ns: str = "",
    ) -> None:
        cur = self._conn.cursor()
        checkpoints = cur.execute(
            """
            SELECT checkpoint_id, checkpoint_type, checkpoint_blob, metadata_type, metadata_blob, parent_checkpoint_id
            FROM checkpoints
            WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id <= ?
            ORDER BY checkpoint_id ASC
            """,
            (from_thread_id, checkpoint_ns, checkpoint_id),
        ).fetchall()
        if not checkpoints:
            raise RuntimeError("Checkpoint não encontrado para fork.")

        checkpoint_ids = [str(r[0]) for r in checkpoints]

        blobs = cur.execute(
            """
            SELECT channel, version, blob_type, blob
            FROM blobs
            WHERE thread_id=? AND checkpoint_ns=?
            """,
            (from_thread_id, checkpoint_ns),
        ).fetchall()

        cur.execute("DELETE FROM writes WHERE thread_id=? AND checkpoint_ns=?", (to_thread_id, checkpoint_ns))
        cur.execute("DELETE FROM blobs WHERE thread_id=? AND checkpoint_ns=?", (to_thread_id, checkpoint_ns))
        cur.execute(
            "DELETE FROM checkpoints WHERE thread_id=? AND checkpoint_ns=?",
            (to_thread_id, checkpoint_ns),
        )

        for (
            cid,
            checkpoint_type,
            checkpoint_blob,
            metadata_type,
            metadata_blob,
            parent_checkpoint_id,
        ) in checkpoints:
            cur.execute(
                """
                INSERT OR REPLACE INTO checkpoints(
                    thread_id, checkpoint_ns, checkpoint_id,
                    checkpoint_type, checkpoint_blob,
                    metadata_type, metadata_blob,
                    parent_checkpoint_id
                )
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    to_thread_id,
                    checkpoint_ns,
                    str(cid),
                    checkpoint_type,
                    checkpoint_blob,
                    metadata_type,
                    metadata_blob,
                    parent_checkpoint_id,
                ),
            )

        for channel, version, blob_type, blob in blobs:
            cur.execute(
                """
                INSERT OR REPLACE INTO blobs(thread_id, checkpoint_ns, channel, version, blob_type, blob)
                VALUES(?,?,?,?,?,?)
                """,
                (to_thread_id, checkpoint_ns, str(channel), str(version), blob_type, blob),
            )

        writes = cur.execute(
            """
            SELECT checkpoint_id, task_id, idx, channel, value_type, value_blob, task_path
            FROM writes
            WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id IN (
                SELECT checkpoint_id
                FROM checkpoints
                WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id <= ?
            )
            """,
            (from_thread_id, checkpoint_ns, from_thread_id, checkpoint_ns, checkpoint_id),
        ).fetchall()
        for cid, task_id, idx, channel, value_type, value_blob, task_path in writes:
            cur.execute(
                """
                INSERT OR REPLACE INTO writes(
                    thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
                    channel, value_type, value_blob, task_path
                )
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    to_thread_id,
                    checkpoint_ns,
                    str(cid),
                    str(task_id),
                    int(idx),
                    str(channel),
                    value_type,
                    value_blob,
                    str(task_path or ""),
                ),
            )

        self._conn.commit()
