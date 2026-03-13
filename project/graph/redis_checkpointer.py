from __future__ import annotations

from collections.abc import Iterator, Sequence
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


class RedisCheckpointSaver(BaseCheckpointSaver[str]):
    def __init__(self, *, redis_url: str, prefix: str) -> None:
        super().__init__()
        import redis

        self._client = redis.Redis.from_url(redis_url)
        self._prefix = str(prefix).strip() or "langgraph:checkpoints"

    def _base(self, *, thread_id: str, checkpoint_ns: str) -> str:
        return f"{self._prefix}:{thread_id}:{checkpoint_ns}"

    def _seq_key(self, *, base: str) -> str:
        return f"{base}:seq"

    def _zset_key(self, *, base: str) -> str:
        return f"{base}:z"

    def _scores_key(self, *, base: str) -> str:
        return f"{base}:scores"

    def _checkpoint_key(self, *, base: str, checkpoint_id: str) -> str:
        return f"{base}:checkpoint:{checkpoint_id}"

    def _blob_key(self, *, base: str, channel: str, version: str | int | float) -> str:
        return f"{base}:blob:{channel}:{version}"

    def _writes_key(self, *, base: str, checkpoint_id: str) -> str:
        return f"{base}:writes:{checkpoint_id}"

    @staticmethod
    def _pack_typed(value: tuple[str, bytes]) -> bytes:
        t, b = value
        return t.encode("utf-8") + b"\x00" + b

    @staticmethod
    def _unpack_typed(raw: bytes) -> tuple[str, bytes]:
        if b"\x00" not in raw:
            return ("empty", b"")
        t, b = raw.split(b"\x00", 1)
        return (t.decode("utf-8"), b)

    def _load_blobs(
        self, *, thread_id: str, checkpoint_ns: str, versions: ChannelVersions
    ) -> dict[str, Any]:
        base = self._base(thread_id=thread_id, checkpoint_ns=checkpoint_ns)
        out: dict[str, Any] = {}
        for channel, version in versions.items():
            raw = self._client.get(self._blob_key(base=base, channel=str(channel), version=str(version)))
            if not raw:
                continue
            t, b = self._unpack_typed(raw if isinstance(raw, (bytes, bytearray)) else bytes(raw))
            if t != "empty":
                out[str(channel)] = self.serde.loads_typed((t, b))
        return out

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        base = self._base(thread_id=thread_id, checkpoint_ns=checkpoint_ns)
        requested = get_checkpoint_id(config)

        if requested:
            key = self._checkpoint_key(base=base, checkpoint_id=str(requested))
            data = self._client.hgetall(key)
            if not data:
                return None
            checkpoint_type = (data.get(b"checkpoint_type") or b"").decode("utf-8")
            checkpoint_blob = data.get(b"checkpoint_blob") or b""
            metadata_type = (data.get(b"metadata_type") or b"").decode("utf-8")
            metadata_blob = data.get(b"metadata_blob") or b""
            parent_checkpoint_id = data.get(b"parent_checkpoint_id")
            parent = parent_checkpoint_id.decode("utf-8") if parent_checkpoint_id else None

            checkpoint: Checkpoint = self.serde.loads_typed((checkpoint_type, checkpoint_blob))
            metadata: CheckpointMetadata = self.serde.loads_typed((metadata_type, metadata_blob))

            writes_raw = self._client.hgetall(self._writes_key(base=base, checkpoint_id=str(requested)))
            pending_writes: list[tuple[str, str, Any]] = []
            for field, packed in writes_raw.items():
                try:
                    task_id, _idx, channel = field.decode("utf-8").split(":", 2)
                    t, b = self._unpack_typed(packed)
                    pending_writes.append((task_id, channel, self.serde.loads_typed((t, b))))
                except Exception:
                    continue

            return CheckpointTuple(
                config=config,
                checkpoint={
                    **checkpoint,
                    "channel_values": self._load_blobs(
                        thread_id=thread_id, checkpoint_ns=checkpoint_ns, versions=checkpoint["channel_versions"]
                    ),
                },
                metadata=metadata,
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent,
                        }
                    }
                    if parent
                    else None
                ),
                pending_writes=pending_writes,
            )

        zkey = self._zset_key(base=base)
        latest = self._client.zrevrange(zkey, 0, 0)
        if not latest:
            return None
        checkpoint_id = latest[0].decode("utf-8") if isinstance(latest[0], (bytes, bytearray)) else str(latest[0])
        key = self._checkpoint_key(base=base, checkpoint_id=checkpoint_id)
        data = self._client.hgetall(key)
        if not data:
            return None

        checkpoint_type = (data.get(b"checkpoint_type") or b"").decode("utf-8")
        checkpoint_blob = data.get(b"checkpoint_blob") or b""
        metadata_type = (data.get(b"metadata_type") or b"").decode("utf-8")
        metadata_blob = data.get(b"metadata_blob") or b""
        parent_checkpoint_id = data.get(b"parent_checkpoint_id")
        parent = parent_checkpoint_id.decode("utf-8") if parent_checkpoint_id else None

        checkpoint: Checkpoint = self.serde.loads_typed((checkpoint_type, checkpoint_blob))
        metadata: CheckpointMetadata = self.serde.loads_typed((metadata_type, metadata_blob))

        writes_raw = self._client.hgetall(self._writes_key(base=base, checkpoint_id=checkpoint_id))
        pending_writes: list[tuple[str, str, Any]] = []
        for field, packed in writes_raw.items():
            try:
                task_id, _idx, channel = field.decode("utf-8").split(":", 2)
                t, b = self._unpack_typed(packed)
                pending_writes.append((task_id, channel, self.serde.loads_typed((t, b))))
            except Exception:
                continue

        return CheckpointTuple(
            config={"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": checkpoint_id}},
            checkpoint={
                **checkpoint,
                "channel_values": self._load_blobs(
                    thread_id=thread_id, checkpoint_ns=checkpoint_ns, versions=checkpoint["channel_versions"]
                ),
            },
            metadata=metadata,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent,
                    }
                }
                if parent
                else None
            ),
            pending_writes=pending_writes,
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        if config is None:
            return iter(())
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        base = self._base(thread_id=thread_id, checkpoint_ns=checkpoint_ns)
        zkey = self._zset_key(base=base)
        count = int(limit) if limit is not None else 200
        ids = self._client.zrevrange(zkey, 0, max(0, count - 1))
        for raw_id in ids:
            checkpoint_id = raw_id.decode("utf-8") if isinstance(raw_id, (bytes, bytearray)) else str(raw_id)
            tup = self.get_tuple(
                {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": checkpoint_id}}
            )
            if tup is None:
                continue
            if filter and not all(tup.metadata.get(k) == v for k, v in filter.items()):
                continue
            yield tup

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        c = checkpoint.copy()
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"]["checkpoint_ns"]
        parent_checkpoint_id: str | None = config["configurable"].get("checkpoint_id")
        base = self._base(thread_id=thread_id, checkpoint_ns=checkpoint_ns)

        values: dict[str, Any] = c.pop("channel_values")  # type: ignore[misc]
        for channel, version in new_versions.items():
            if channel in values:
                packed = self._pack_typed(self.serde.dumps_typed(values[channel]))
            else:
                packed = self._pack_typed(("empty", b""))
            self._client.set(self._blob_key(base=base, channel=str(channel), version=str(version)), packed)

        checkpoint_type, checkpoint_blob = self.serde.dumps_typed(c)
        metadata_type, metadata_blob = self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
        seq = int(self._client.incr(self._seq_key(base=base)))
        zkey = self._zset_key(base=base)
        scores_key = self._scores_key(base=base)

        key = self._checkpoint_key(base=base, checkpoint_id=str(checkpoint["id"]))
        self._client.hset(
            key,
            mapping={
                "checkpoint_type": checkpoint_type,
                "checkpoint_blob": checkpoint_blob,
                "metadata_type": metadata_type,
                "metadata_blob": metadata_blob,
                "parent_checkpoint_id": parent_checkpoint_id or "",
            },
        )
        self._client.zadd(zkey, {str(checkpoint["id"]): float(seq)})
        self._client.hset(scores_key, str(checkpoint["id"]), str(seq))

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
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id: str = config["configurable"]["checkpoint_id"]
        base = self._base(thread_id=thread_id, checkpoint_ns=checkpoint_ns)
        wkey = self._writes_key(base=base, checkpoint_id=str(checkpoint_id))

        mapping: dict[str, bytes] = {}
        for idx, (channel, value) in enumerate(writes):
            mapped_idx = WRITES_IDX_MAP.get(channel, idx)
            packed = self._pack_typed(self.serde.dumps_typed(value))
            mapping[f"{task_id}:{int(mapped_idx)}:{channel}"] = packed
        if mapping:
            self._client.hset(wkey, mapping=mapping)

    def delete_thread(self, thread_id: str) -> None:
        pattern = f"{self._prefix}:{thread_id}:*"
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=pattern, count=500)
            if keys:
                self._client.delete(*keys)
            if int(cursor) == 0:
                break

    def list_checkpoint_ids(self, *, thread_id: str, checkpoint_ns: str = "", limit: int = 50) -> list[str]:
        base = self._base(thread_id=thread_id, checkpoint_ns=checkpoint_ns)
        ids = self._client.zrevrange(self._zset_key(base=base), 0, max(0, int(limit) - 1))
        return [i.decode("utf-8") if isinstance(i, (bytes, bytearray)) else str(i) for i in ids]

    def fork_thread(
        self,
        *,
        from_thread_id: str,
        to_thread_id: str,
        checkpoint_id: str,
        checkpoint_ns: str = "",
    ) -> None:
        from_base = self._base(thread_id=from_thread_id, checkpoint_ns=checkpoint_ns)
        to_base = self._base(thread_id=to_thread_id, checkpoint_ns=checkpoint_ns)

        scores_key = self._scores_key(base=from_base)
        score_raw = self._client.hget(scores_key, str(checkpoint_id))
        if not score_raw:
            raise RuntimeError("Checkpoint não encontrado para fork.")
        score = int(score_raw.decode("utf-8") if isinstance(score_raw, (bytes, bytearray)) else str(score_raw))

        self.delete_thread(to_thread_id)

        zkey = self._zset_key(base=from_base)
        ids = self._client.zrangebyscore(zkey, min=0, max=score)
        if not ids:
            raise RuntimeError("Checkpoint não encontrado para fork.")

        seq_key = self._seq_key(base=to_base)
        scores_to = self._scores_key(base=to_base)
        zkey_to = self._zset_key(base=to_base)

        pipe = self._client.pipeline()
        for raw_id in ids:
            cid = raw_id.decode("utf-8") if isinstance(raw_id, (bytes, bytearray)) else str(raw_id)
            data = self._client.hgetall(self._checkpoint_key(base=from_base, checkpoint_id=cid))
            if not data:
                continue
            pipe.hset(self._checkpoint_key(base=to_base, checkpoint_id=cid), mapping=data)
            seq = int(self._client.incr(seq_key))
            pipe.zadd(zkey_to, {cid: float(seq)})
            pipe.hset(scores_to, cid, str(seq))

            writes = self._client.hgetall(self._writes_key(base=from_base, checkpoint_id=cid))
            if writes:
                pipe.hset(self._writes_key(base=to_base, checkpoint_id=cid), mapping=writes)

        cursor = 0
        pattern = f"{from_base}:blob:*"
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=pattern, count=500)
            for k in keys:
                raw = self._client.get(k)
                if raw is None:
                    continue
                kk = k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else str(k)
                to_k = kk.replace(f"{from_base}:", f"{to_base}:")
                pipe.set(to_k, raw)
            if int(cursor) == 0:
                break

        pipe.execute()

