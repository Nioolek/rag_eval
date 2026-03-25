# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains two integrated projects:

1. **rag_eval** (root): Enterprise-grade RAG evaluation framework with Gradio frontend. Supports annotation management, concurrent RAG evaluation, 20+ metrics, and result visualization.
2. **rag_rag**: RAG pipeline service built with LangGraph (14-node StateGraph with parallel retrieval). Provides the backend RAG service that rag_eval evaluates.

Detailed requirements: `requirement.md` (Chinese).

## Technology Stack

- **Language**: Python 3.13
- **Frontend**: Gradio 6.x (streaming, responsive UI)
- **RAG Interface**: LangGraph RemoteGraph for external RAG services
- **LLM**: LangChain with OpenAI (evaluation) / Alibaba Qwen (rag_rag)
- **Storage**: Local files (primary), SQLite (secondary), Chroma/Whoosh/Neo4j (rag_rag)

## Architecture

```
rag_eval/
├── src/
│   ├── core/           # Config (singleton), exceptions, logging
│   ├── models/         # Pydantic models: Annotation, RAGResponse, MetricResult, EvaluationResult
│   ├── storage/        # StorageBackend (abstract), LocalStorage, SQLiteStorage, StorageFactory
│   ├── annotation/     # AnnotationHandler, AnnotationIterator, AnnotationStatistics
│   ├── rag/            # RAGAdapter (abstract), LangGraphAdapter, MockAdapter
│   ├── evaluation/     # Runner, LLMJudge, ResultManager
│   │   └── metrics/    # BaseMetric, MetricRegistry, MetricFactory, metric implementations
│   ├── ui/             # Gradio app
│   │   └── components/ # annotation_tab, evaluation_tab, results_tab, statistics_tab
│   └── utils/          # validators, file_handlers
├── tests/              # pytest tests mirroring src/ structure
└── data/               # Local data storage

rag_rag/                # LangGraph RAG pipeline service
├── src/rag_rag/
│   ├── core/           # Config, exceptions, logging, constants
│   ├── graph/          # StateGraph definition, routers, 14 nodes
│   │   └── nodes/      # input, faq_match, query_rewrite, vector/fulltext/graph_retrieve, merge, rerank, etc.
│   ├── storage/        # FAQStore, VectorStore, FulltextStore, GraphStore, SessionStore
│   ├── services/       # LLMService, EmbeddingService, RerankService, SensitiveFilter
│   ├── degradation/    # CircuitBreaker, FallbackHandlers
│   └── prompts/        # TemplateManager
├── config/             # settings.yaml (business config)
└── scripts/            # Data generation and ingestion scripts
```

## Design Patterns

All code follows these patterns:

- **Strategy Pattern**: `BaseMetric` - each metric is an independent strategy
- **Factory Pattern**: `MetricFactory`, `StorageFactory` - centralized instantiation
- **Adapter Pattern**: `RAGAdapter` - adapts different RAG services (LangGraph, Mock)
- **Template Method**: Standardized evaluation/annotation workflows in handlers
- **Singleton Pattern**: `get_config()` via `@lru_cache`, storage backends
- **Iterator Pattern**: `AnnotationIterator` for batched data traversal

## Development Commands

### rag_eval (evaluation framework)

```bash
# Install dependencies
pip install -r requirements.txt

# Run Gradio application
python -m src.main
# Or with options:
python -m src.main --host 0.0.0.0 --port 7860 --debug

# Run tests
pytest tests/ -v

# Run single test file
pytest tests/test_annotation/test_annotation.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### rag_rag (RAG pipeline service)

```bash
cd rag_rag

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Start LangGraph API server (required for evaluation)
langgraph dev --port 8123 --allow-blocking

# Run tests
pytest tests/ -v

# Data ingestion
python scripts/ingest_electronics_data.py --generate
```

## Coding Standards

### Async Requirements
- All I/O operations (file, network, DB, RAG/LLM calls) must use `async/await`
- CPU-intensive operations (metrics, parsing) must use process pool
- Never block the Gradio main thread

### Security
- All secrets from environment variables only (never hardcode)
- Validate and sanitize all user inputs
- Protect against path traversal attacks

### Adding New Metrics
1. Create new file in `src/evaluation/metrics/`
2. Extend `BaseMetric` class
3. Implement `async def calculate(self, context: MetricContext) -> MetricResult`
4. Register in `MetricRegistry` (`src/evaluation/metrics/metric_registry.py`)

No core code changes needed.

### Adding New RAG Adapters
1. Create new file in `src/rag/`
2. Extend `RAGAdapter` class
3. Implement `query()`, `query_from_annotation()`, `health_check()`, `close()`
4. Optionally override `stream_query()` for streaming support

## Environment Variables

### rag_eval (.env)

Required:
- `OPENAI_API_KEY`: LLM-based evaluation metrics
- `RAG_SERVICE_URL`: RAG service endpoint (e.g., `http://localhost:8123`)

Optional:
- `OPENAI_BASE_URL` / `OPENAI_API_BASE`: Custom OpenAI endpoint
- `OPENAI_MODEL`: Model name (default: gpt-4)
- `STORAGE_TYPE`: "local" or "sqlite"
- `DATA_DIR`: Data storage directory
- `DATABASE_URL`: SQLite connection string
- `MAX_CONCURRENT_EVALUATIONS`: Concurrent evaluation limit
- `GRADIO_SERVER_NAME` / `GRADIO_SERVER_PORT`: UI server config

### rag_rag (.env)

Required:
- `DASHSCOPE_API_KEY`: Alibaba Cloud API key for embeddings and LLM

Optional:
- `LLM_MODEL`: qwen-plus (default)
- `EMBEDDING_MODEL`: text-embedding-v3
- `EMBEDDING_DIMENSION`: 1024
- `DATA_DIR`: Data storage directory

## Key Files

- `src/core/config.py`: Singleton config via `get_config()`
- `src/evaluation/metrics/base.py`: `BaseMetric` and `MetricContext`
- `src/evaluation/metrics/metric_registry.py`: Metric registration
- `src/rag/base_adapter.py`: `RAGAdapter` interface
- `src/rag/langgraph_adapter.py`: LangGraph RemoteGraph integration
- `rag_rag/src/rag_rag/graph/graph.py`: 14-node StateGraph definition
- `rag_rag/config/settings.yaml`: RAG pipeline business configuration