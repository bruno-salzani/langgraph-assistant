"""Exemplo de execução do AI Operating System baseado em LangGraph 100%."""

import asyncio
import logging
from pathlib import Path

from project.app import AssistantManager
from project.config.settings import load_settings
from project.services.logging_service import setup_logging


async def run_ai_os_example():
    # 1. Configuração do ambiente
    project_root = Path(__file__).parent
    settings = load_settings(project_root=project_root)
    logger = setup_logging(settings.logs_dir)

    # Força o uso do motor LangGraph (AI OS)
    settings.assistant_engine = "langgraph"

    # 2. Inicialização do Gerenciador de Assistentes
    manager = AssistantManager(settings=settings, logger=logger)

    # 3. Obtenção do assistente (AI OS)
    session_id = "ai_os_session"
    assistant = manager.get(session_id=session_id, streaming=True)

    # 4. Consulta complexa
    user_input = "Calcule o lucro projetado para 2026 baseado nos documentos de RAG e pesquise tendências de mercado."

    print(f"\n--- [AI OS] Iniciando Processamento: '{user_input}' ---\n")

    # 5. Streaming da resposta final
    print("AI OS Output:")
    for token in assistant.stream_tokens(user_input):
        print(token, end="", flush=True)

    print("\n\n--- Processamento Finalizado ---")


if __name__ == "__main__":
    # Configura logging para console
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_ai_os_example())
