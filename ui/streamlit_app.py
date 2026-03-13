"""Streamlit UI para o assistant (chat + upload de documentos + fontes)."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from project.app import AssistantManager
from project.config.settings import load_settings
from project.learning.feedback_store import build_feedback_store
from project.learning.learning_loop import LearningLoop
from project.rag.retriever import rebuild_faiss_store
from project.security.prompt_guard import PromptGuard
from project.services.logging_service import setup_logging


def _init() -> tuple[AssistantManager, PromptGuard]:
    settings = load_settings(project_root=Path(__file__).resolve().parents[1])
    logger = setup_logging(settings.logs_dir)
    return AssistantManager(settings=settings, logger=logger), PromptGuard(settings)


manager, guard = _init()

st.set_page_config(page_title="LangChain AI Assistant", layout="wide")
st.title("🤖 LangChain AI Assistant")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = "streamlit"
if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    st.subheader("⚙️ Sessão")
    st.session_state["session_id"] = st.text_input("session_id", st.session_state["session_id"])
    view = st.selectbox("Tela", ["Chat", "Metrics"], index=0)

    st.subheader("📄 Documentos (RAG)")
    uploaded = st.file_uploader("Upload (pdf/txt/md)", type=["pdf", "txt", "md"])
    if uploaded is not None:
        settings = load_settings(project_root=Path(__file__).resolve().parents[1])
        target = settings.docs_dir / uploaded.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(uploaded.getvalue())
        st.success(f"Arquivo salvo em: {target}")

    if st.button("Reindexar documentos (FAISS)"):
        settings = load_settings(project_root=Path(__file__).resolve().parents[1])
        logger = setup_logging(settings.logs_dir)
        rebuild_faiss_store(settings, logger)
        st.success("Índice RAG recriado.")


def _load_metrics(logs_dir: Path) -> list[dict]:
    path = logs_dir / "metrics.jsonl"
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


if view == "Metrics":
    settings = load_settings(project_root=Path(__file__).resolve().parents[1])
    rows = _load_metrics(settings.logs_dir)
    st.subheader("📈 Métricas")
    st.caption(f"Arquivo: {settings.logs_dir / 'metrics.jsonl'}")
    if not rows:
        st.info("Ainda não há métricas registradas.")
        st.stop()

    timers = [r for r in rows if r.get("type") == "timer"]
    tokens = [r for r in rows if r.get("type") == "tokens"]

    st.write({"timers": len(timers), "tokens": len(tokens)})

    if timers:
        by_name: dict[str, list[float]] = {}
        for r in timers:
            name = str(r.get("name", "unknown"))
            by_name.setdefault(name, []).append(float(r.get("duration_ms", 0.0)))
        names = sorted(by_name)
        data = {n: sum(by_name[n]) / max(len(by_name[n]), 1) for n in names}
        st.subheader("⏱️ Latência média (ms)")
        st.bar_chart(data)

    if tokens:
        total_in = sum(int(r.get("input_tokens", 0)) for r in tokens)
        total_out = sum(int(r.get("output_tokens", 0)) for r in tokens)
        total_cost = sum(float(r.get("estimated_cost_usd", 0.0)) for r in tokens)
        st.subheader("🧾 Tokens e custo")
        st.write(
            {
                "input_tokens": total_in,
                "output_tokens": total_out,
                "estimated_cost_usd": total_cost,
            }
        )
    st.stop()

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_text = st.chat_input("Digite sua mensagem…")
if user_text:
    try:
        user_text = guard.enforce(user_text)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    assistant = manager.get(session_id=st.session_state["session_id"], streaming=False)
    with st.chat_message("assistant"):
        with st.spinner("Pensando…"):
            result = assistant.invoke(user_text)
            answer = str(result.get("output", ""))
            st.markdown(answer)
    st.session_state["messages"].append({"role": "assistant", "content": answer})

    with st.expander("👍 Enviar feedback"):
        settings = load_settings(project_root=Path(__file__).resolve().parents[1])
        rating = st.slider("Nota (1–5)", min_value=1, max_value=5, value=4)
        comments = st.text_input("Comentários", "")
        if st.button("Salvar feedback"):
            store = build_feedback_store(
                backend=settings.feedback_backend,
                sqlite_path=settings.feedback_sqlite_path,
                postgres_url=settings.postgres_url,
            )
            store.add(question=user_text, answer=answer, rating=rating, comments=comments)
            st.success("Feedback salvo.")

        if st.button("Rodar learning loop (gera prompt v2)"):
            store = build_feedback_store(
                backend=settings.feedback_backend,
                sqlite_path=settings.feedback_sqlite_path,
                postgres_url=settings.postgres_url,
            )
            loop = LearningLoop(
                store=store, prompts_dir=settings.project_root / "project" / "prompts"
            )
            out = loop.run_once(
                base_version=settings.prompt_version, target_version="v2", limit=200
            )
            if out.wrote_version:
                st.success(f"Prompt gerado: {out.wrote_version}")
            else:
                st.info("Sem mudança (poucos dados ou prompt base ausente).")
