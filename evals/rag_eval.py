"""Avaliação de RAG com Ragas.

Métricas:
- faithfulness
- context_precision
- answer_relevancy

Este script é um ponto de partida: você pode substituir o dataset por um CSV/JSON de benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from project.config.settings import load_settings
from project.rag.reranker import ScoreReranker
from project.rag.retriever import FaissRetriever, build_or_load_faiss_store
from project.services.llm_service import build_chat_llm, build_embeddings
from project.services.logging_service import setup_logging


def _build_samples() -> list[dict[str, Any]]:
    return [
        {
            "question": "Como proceder em caso de roubo de item comprado?",
            "ground_truth": "O procedimento deve seguir o que está descrito no documento do cartão.",
        }
    ]


def main() -> None:
    settings = load_settings(project_root=Path(__file__).resolve().parents[1])
    logger = setup_logging(settings.logs_dir)

    llm = build_chat_llm(settings, streaming=False)
    embeddings = build_embeddings(settings)

    store = build_or_load_faiss_store(settings, logger)
    retriever = FaissRetriever(store)
    reranker = ScoreReranker()

    samples = _build_samples()
    rows: list[dict[str, Any]] = []
    for s in samples:
        question = s["question"]
        retrieved = retriever.retrieve(question, k=settings.rag_initial_k).documents
        top_docs = reranker.rerank(retrieved, top_k=settings.rag_top_k)
        contexts = [d.document.page_content for d in top_docs]

        prompt = (
            "Responda usando exclusivamente o contexto. Se o contexto não tiver a resposta, diga que não há evidência.\n\n"
            f"Pergunta: {question}\n\nContexto:\n" + "\n\n".join(contexts) + "\n\nResposta:"
        )
        answer = llm.invoke(prompt).content
        rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": s["ground_truth"],
            }
        )

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        ds = Dataset.from_list(rows)
        result = evaluate(
            ds,
            metrics=[faithfulness, context_precision, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
        )
        out = Path(settings.logs_dir) / "ragas_report.json"
        out.write_text(result.to_pandas().to_json(orient="records"), encoding="utf-8")
        print(f"Relatório salvo em: {out}")
    except Exception as exc:
        raw_out = Path(settings.logs_dir) / "rag_eval_raw.json"
        raw_out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Falha ao rodar Ragas ({exc}). Dataset salvo em: {raw_out}")


if __name__ == "__main__":
    main()
