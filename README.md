# 🤖 AI Operating System (AI-OS) — LangGraph 100%

![Status](https://img.shields.io/badge/Status-Refatorado-brightgreen)
![Framework](https://img.shields.io/badge/Framework-LangGraph%20100%25-blueviolet)
![Stack](https://img.shields.io/badge/Stack-Python%20%7C%20AI--OS%20%7C%20Multi--Agent-blue)

Refatoração completa do projeto para operar como um **AI Operating System** baseado em grafos de agentes, eliminando cadeias lineares e utilizando **LangGraph** de ponta a ponta.

## 🧰 Tecnologias
- **Orquestração multi-agent**: LangGraph (StateGraph + subgrafos)
- **LLM**: OpenAI (via `openai` + `langchain-openai`)
- **Fallback local (opcional)**: Ollama (via `langchain-community`)
- **RAG**: FAISS (`faiss-cpu`) + pipeline de chunking/retrieval/rerank
- **Web**: DuckDuckGo Search (`duckduckgo-search`) + Scraping (`requests` + `beautifulsoup4`)
- **APIs**: FastAPI + Uvicorn (streaming SSE)
- **UI**: Streamlit (chat + upload + painel de métricas)
- **Cache**: SQLite/Redis (cache de prompts/LLM + cache LangChain quando disponível)
- **Persistência**:
  - Checkpoints do grafo: SQLite (arquivo em `logs/checkpoints.sqlite3`)
  - Long-term memory: SQLite/Redis/Postgres
  - Feedback: SQLite/Postgres
  - Tool learning: SQLite
- **Observabilidade**: timers e token usage em JSONL (`logs/metrics.jsonl`)
- **Evals**: Ragas + Datasets
- **Qualidade**: Pytest + Pytest-cov, Ruff, Black
- **Infra**: Docker + docker-compose, Postgres (opcional), Redis (opcional)

## 🏗️ Arquitetura AI-OS (LangGraph 100%)
O sistema opera como um grafo de decisão inteligente, com estados explícitos e observabilidade total:

### Estrutura de Pastas
- `project/graph/`: Orquestração do Grafo Principal e definições de Estado.
- `project/nodes/`: Implementações modulares de cada Agente/Nó (Guardrails, Planner, Router, Executor, Synthesizer, Critic, Learning).
- `project/subgraphs/`: Pipeline RAG avançado como um subgrafo independente.
- `project/memory/`: Sistema de memória de dois níveis (Short-term no estado, Long-term persistente).
- `project/tools/`: Ferramentas técnicas (Search, Scraper, Calculator, File Reader).
- `project/services/`: Roteamento de LLMs, Métricas e Cache.
- `project/security/`: Guardrails de prompt/tools e políticas de segurança.
- `project/learning/`: Feedback loop, armazenamento e melhoria contínua.
- `evals/`: Scripts de avaliação (RAG/Tools/Agent).
- `api/`: API FastAPI (inclui streaming).
- `ui/`: App Streamlit (UI e painel de métricas).

### Fluxo de Operação
1. **START** → **Guardrails** (Sanitização)
2. **Planner** (Decomposição de tarefas)
3. **Router** (Roteamento inteligente: RAG, Web, Tools)
4. **Execution Layer** (RAG Subgraph ou Tool Executor com retentativas)
5. **Synthesizer** (Consolidação de evidências)
6. **Critic/Reflection** (Loop de validação factual)
7. **Learning** (Persistência em Long-term memory e melhoria contínua) → **END**

## 🚀 Funcionalidades Chave
- **100% LangGraph**: Zero `AgentExecutor` ou cadeias implícitas.
- **Checkpoints persistentes do grafo (SQLite)**: retomada de execução por `thread_id` e replay/debug.
- **Observabilidade por nó e por tool**: timers + token usage em `logs/metrics.jsonl`.
- **RAG completo**: rewriter → retriever (FAISS) → reranker → generator, com suporte a reindex e watch de docs.
- **Ferramentas (tools)**: web search, web scrape, leitura de arquivos, cálculo e consulta a documentos via RAG.
- **Roteamento de modelo com fallback**: primário/secundário OpenAI e fallback opcional para Ollama.
- **Human-in-the-loop (opcional)**: nó de aprovação antes de executar ações críticas.
- **Segurança**:
  - sanitização/validação de prompt (PromptGuard)
  - limite de chamadas de tools por request (`MAX_TOOL_CALLS`)
  - allowlist e políticas de execução de tools
- **Memória em dois níveis**:
  - short-term no estado do grafo (com checkpoints persistentes)
  - long-term em SQLite/Redis/Postgres (histórico por sessão)
- **Aprendizado contínuo**:
  - feedback store (SQLite/Postgres)
  - learning loop para gerar novas versões de prompt
  - tool learning (sucesso/falha/latência por tool)
- **Interfaces completas**:
  - CLI (scripts `main*.py`)
  - API FastAPI (inclui streaming SSE)
  - UI Streamlit (chat + upload + métricas)

## ▶️ Executando o AI-OS
### Requisitos
- Python 3.11+ (testado no CI em 3.11/3.12)
- `.env` na raiz com `OPENAI_API_KEY=...`

### Teste Completo (AI-OS)
Demonstra o fluxo multi-agente, RAG e memória persistente:
```bash
python main_ai_os.py
```

### Chat Interativo (Streaming)
```bash
python main.py
```

### API FastAPI
```bash
uvicorn api.server:app --reload
```

### UI Streamlit
```bash
streamlit run ui/streamlit_app.py
```

## ⚙️ Configuração (env vars)
Principais variáveis (todas são lidas via `project/config/settings.py`):
- `OPENAI_API_KEY` (obrigatório)
- `LLM_MODEL`, `LLM_SECONDARY_MODEL`, `LLM_TEMPERATURE`
- `EMBEDDING_MODEL`
- `DOCS_DIR`, `RAG_INDEX_DIR`, `LOGS_DIR`
- `CACHE_BACKEND` (`none|sqlite|redis`), `CACHE_SQLITE_PATH`, `CACHE_TTL_S`, `REDIS_URL`
- `MEMORY_BACKEND` (`sqlite|redis|postgres`), `MEMORY_SQLITE_PATH`, `POSTGRES_URL`
- `MAX_TOOL_CALLS`, `MAX_TOOL_OUTPUT_CHARS`, `ALLOW_LOCAL_FILE_READS_UNDER`
- `RAG_*` (top_k, chunking, watch, híbrido, rerank, etc.)
- `HUMAN_IN_LOOP_ENABLED`, `HUMAN_IN_LOOP_REQUIRE_APPROVAL`
- `TOOL_LEARNING_ENABLED`, `TOOL_LEARNING_SQLITE_PATH`
- `FEEDBACK_BACKEND` (`sqlite|postgres`), `FEEDBACK_SQLITE_PATH`
- `TREE_OF_THOUGHT_ENABLED`, `AGENT_FACTORY_ENABLED`

## 📈 Observabilidade
- **Timers**: latência por nó (`node:<name>`) e por tool (`tool:<name>`)
- **Tokens/custo estimado**: quando disponíveis via callbacks do provider
- **Persistência**: `logs/metrics.jsonl`
- **Painel**: tela “Metrics” no Streamlit

## 🧪 Evals
Scripts prontos em `evals/`:
```bash
python -m evals.rag_eval
python -m evals.tool_eval
python -m evals.agent_eval
```

## ✅ Testes e Qualidade
```bash
python -m ruff check .
python -m black --check .
python -m pytest
```

## 🐳 Docker
```bash
docker-compose up --build
```
