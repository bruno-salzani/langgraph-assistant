"""Testes do build/load do FAISS store (com embeddings fake)."""

from __future__ import annotations

import logging

from langchain_core.embeddings import FakeEmbeddings

from project.rag.retriever import (
    build_or_load_faiss_store,
    list_doc_files,
    rebuild_faiss_store,
)


def test_list_doc_files(tmp_path) -> None:
    docs = tmp_path / "documentos"
    docs.mkdir()
    (docs / "a.txt").write_text("a", encoding="utf-8")
    (docs / "b.md").write_text("b", encoding="utf-8")
    files = list_doc_files(docs)
    assert len(files) == 2


def test_build_or_load_faiss_store_with_fake_embeddings(dummy_settings, monkeypatch) -> None:
    logger = logging.getLogger("test")
    dummy_settings.docs_dir.mkdir(parents=True, exist_ok=True)
    (dummy_settings.docs_dir / "doc.txt").write_text("cartão platinum", encoding="utf-8")

    monkeypatch.setattr(
        "project.rag.retriever.build_embeddings",
        lambda settings: FakeEmbeddings(size=8),
    )

    store1 = build_or_load_faiss_store(dummy_settings, logger)
    assert store1.index.ntotal > 0

    store2 = build_or_load_faiss_store(dummy_settings, logger)
    assert store2.index.ntotal == store1.index.ntotal


def test_rebuild_faiss_store(dummy_settings, monkeypatch) -> None:
    logger = logging.getLogger("test")
    dummy_settings.docs_dir.mkdir(parents=True, exist_ok=True)
    (dummy_settings.docs_dir / "doc.txt").write_text("benefícios", encoding="utf-8")

    monkeypatch.setattr(
        "project.rag.retriever.build_embeddings",
        lambda settings: FakeEmbeddings(size=8),
    )

    store = rebuild_faiss_store(dummy_settings, logger)
    assert store.index.ntotal > 0
