"""Retriever avançado para FAISS com suporte a scores."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from project.config.settings import Settings
from project.rag.chunking import split_documents_adaptive, split_documents_fixed
from project.rag.reranker import ScoredDocument
from project.services.llm_service import build_embeddings


def list_doc_files(docs_dir: Path) -> list[Path]:
    files: list[Path] = []
    if not docs_dir.exists():
        return files
    for ext in ("*.pdf", "*.txt", "*.md"):
        files.extend(docs_dir.glob(ext))
    return sorted(files)


def load_documents(files: Sequence[Path]) -> list[Document]:
    docs: list[Document] = []
    for file in files:
        if file.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(file)).load())
        else:
            docs.extend(TextLoader(str(file), encoding="utf-8").load())
    return docs


def build_or_load_faiss_store(settings: Settings, logger: logging.Logger) -> FAISS:
    settings.rag_index_dir.mkdir(parents=True, exist_ok=True)
    index_file = settings.rag_index_dir / "index.faiss"
    embeddings = build_embeddings(settings)

    if index_file.exists():
        return FAISS.load_local(
            str(settings.rag_index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    files = list_doc_files(settings.docs_dir)
    if not files:
        raise RuntimeError(
            f"Nenhum documento encontrado em {settings.docs_dir}. Ajuste DOCS_DIR ou adicione arquivos."
        )

    raw_docs = load_documents(files)
    if settings.rag_chunking_mode == "adaptive":
        chunks = split_documents_adaptive(settings=settings, documents=raw_docs)
    else:
        chunks = split_documents_fixed(settings=settings, documents=raw_docs)

    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(settings.rag_index_dir))
    logger.info("RAG index criado em %s (%d chunks)", settings.rag_index_dir, len(chunks))
    return store


def rebuild_faiss_store(settings: Settings, logger: logging.Logger) -> FAISS:
    embeddings = build_embeddings(settings)
    files = list_doc_files(settings.docs_dir)
    if not files:
        raise RuntimeError(
            f"Nenhum documento encontrado em {settings.docs_dir}. Ajuste DOCS_DIR ou adicione arquivos."
        )

    raw_docs = load_documents(files)
    if settings.rag_chunking_mode == "adaptive":
        chunks = split_documents_adaptive(settings=settings, documents=raw_docs)
    else:
        chunks = split_documents_fixed(settings=settings, documents=raw_docs)

    settings.rag_index_dir.mkdir(parents=True, exist_ok=True)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(settings.rag_index_dir))
    logger.info("RAG index recriado em %s (%d chunks)", settings.rag_index_dir, len(chunks))
    return store


@dataclass(frozen=True)
class RetrieverResult:
    query: str
    documents: list[ScoredDocument]


class FaissRetriever:
    """Retriever com scores via `similarity_search_with_score`."""

    def __init__(self, store: FAISS):
        self._store = store

    def retrieve(self, query: str, *, k: int) -> RetrieverResult:
        results = self._store.similarity_search_with_score(query, k=k)
        docs = [ScoredDocument(document=d, score=float(score)) for d, score in results]
        return RetrieverResult(query=query, documents=docs)
