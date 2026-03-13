"""CLI do assistant (modo chat e modo query única)."""

from __future__ import annotations

import argparse
from pathlib import Path

from project.app import AssistantManager
from project.config.settings import load_settings
from project.rag.index_watcher import start_background_watcher
from project.security.prompt_guard import PromptGuard
from project.services.logging_service import setup_logging


def _run_once(query: str, *, stream: bool) -> None:
    settings = load_settings(project_root=Path(__file__).resolve().parent)
    logger = setup_logging(settings.logs_dir)
    if settings.rag_watch_enabled:
        start_background_watcher(settings=settings, logger=logger)
    manager = AssistantManager(settings=settings, logger=logger)
    assistant = manager.get(session_id="cli", streaming=stream)
    guard = PromptGuard(settings)

    if stream:
        try:
            print("IA: ", end="")
            for token in assistant.stream_tokens(guard.enforce(query)):
                print(token, end="", flush=True)
        except Exception as exc:
            print(f"Erro: {exc}")
        print()
        return

    try:
        result = assistant.invoke(guard.enforce(query))
        print("IA:", result.get("output", ""))
    except Exception as exc:
        print(f"Erro: {exc}")


def _run_chat(*, stream: bool) -> None:
    settings = load_settings(project_root=Path(__file__).resolve().parent)
    logger = setup_logging(settings.logs_dir)
    if settings.rag_watch_enabled:
        start_background_watcher(settings=settings, logger=logger)
    manager = AssistantManager(settings=settings, logger=logger)
    assistant = manager.get(session_id="cli", streaming=stream)
    guard = PromptGuard(settings)

    print("AI Assistant (LangGraph) — digite 'sair' para encerrar.")
    while True:
        try:
            user_input = input("\nVocê: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            return

        if not user_input:
            continue
        if user_input.lower() in {"sair", "exit", "quit"}:
            print("Encerrando.")
            return

        if stream:
            try:
                print("IA: ", end="")
                for token in assistant.stream_tokens(guard.enforce(user_input)):
                    print(token, end="", flush=True)
            except Exception as exc:
                print(f"Erro: {exc}")
            print()
        else:
            try:
                result = assistant.invoke(guard.enforce(user_input))
                print("IA:", result.get("output", ""))
            except Exception as exc:
                print(f"Erro: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()

    stream = not args.no_stream

    if args.query:
        _run_once(args.query, stream=stream)
    else:
        _run_chat(stream=stream)


if __name__ == "__main__":
    main()
