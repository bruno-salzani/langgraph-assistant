"""Chunking adaptativo para documentos do RAG."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from project.config.settings import Settings


@dataclass(frozen=True)
class ChunkingParams:
    chunk_size: int
    chunk_overlap: int


def _clamp(v: int, *, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _pick_params_for_source(
    *, settings: Settings, source: str, docs: list[Document]
) -> ChunkingParams:
    ext = Path(source).suffix.lower()
    texts = [d.page_content for d in docs]
    avg_len = (sum(len(t) for t in texts) / max(len(texts), 1)) if texts else 0.0

    base = settings.rag_chunk_size
    if ext == ".pdf":
        base = int(base * 0.7)
    elif ext == ".md":
        base = int(base * 1.1)
    elif ext == ".txt":
        base = int(base * 1.0)

    if avg_len > 3500:
        base = int(base * 0.7)
    elif avg_len < 800:
        base = int(base * 1.1)

    chunk_size = _clamp(
        base,
        lo=settings.rag_chunk_min,
        hi=settings.rag_chunk_max,
    )
    overlap = _clamp(settings.rag_chunk_overlap, lo=0, hi=max(int(chunk_size * 0.3), 1))
    return ChunkingParams(chunk_size=chunk_size, chunk_overlap=overlap)


def split_documents_adaptive(*, settings: Settings, documents: list[Document]) -> list[Document]:
    grouped: dict[str, list[Document]] = defaultdict(list)
    for d in documents:
        source = str(d.metadata.get("source", "unknown"))
        grouped[source].append(d)

    chunks: list[Document] = []
    for source, docs in grouped.items():
        params = _pick_params_for_source(settings=settings, source=source, docs=docs)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=params.chunk_size,
            chunk_overlap=params.chunk_overlap,
        )
        chunks.extend(splitter.split_documents(docs))
    return chunks


def split_documents_fixed(*, settings: Settings, documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    return splitter.split_documents(documents)
