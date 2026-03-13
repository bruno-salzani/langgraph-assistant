"""Exemplo completo do sistema baseado em LangGraph conforme prompt3.md."""

import asyncio
import logging
from pathlib import Path

from project.app import AssistantManager
from project.config.settings import load_settings
from project.services.logging_service import setup_logging


async def run_full_example():
    # 1. Configuração do ambiente
    project_root = Path(__file__).parent
    settings = load_settings(project_root=project_root)
    logger = setup_logging(settings.logs_dir)

    # Força o uso do motor LangGraph
    settings.assistant_engine = "langgraph"
    settings.agent_mode = "orchestrator"

    # 2. Inicialização do Gerenciador de Assistentes
    manager = AssistantManager(settings=settings, logger=logger)

    # 3. Obtenção do assistente para uma sessão específica
    session_id = "test_session_full"
    assistant = manager.get(session_id=session_id, streaming=True)

    # 4. Execução de uma consulta complexa
    user_input = "Pesquise sobre as tendências de IA para 2026 e compare com os documentos no meu repositório sobre RAG."

    print(f"\n--- Iniciando consulta: '{user_input}' ---\n")

    # 5. Streaming de resposta
    print("Resposta (Streaming):")
    full_response = ""
    for token in assistant.stream_tokens(user_input):
        print(token, end="", flush=True)
        full_response += token

    print("\n\n--- Fim da consulta ---")

    # 6. Verificação de memória (próxima interação)
    print("\n--- Próxima interação (Memória) ---")
    follow_up = "O que acabamos de discutir?"
    for token in assistant.stream_tokens(follow_up):
        print(token, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    # Configura logging básico para console
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_full_example())
