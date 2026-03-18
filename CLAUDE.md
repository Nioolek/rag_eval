# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Enterprise-grade RAG (Retrieval-Augmented Generation) evaluation framework with Gradio frontend. The system supports annotation management, concurrent RAG evaluation, comprehensive metrics calculation, and result visualization.
详细的需求描述存储在requirement.md

## Technology Stack

- **Language**: Python 3.13
- **Frontend**: Gradio (streaming, responsive UI)
- **RAG Interface**: LangGraph RemoteGraph for external RAG services
- **LLM Evaluation**: LangChain with OpenAI
- **Storage**: Local files (primary), SQLite (secondary), with version management

## Architecture

```
rag_eval/
├── src/
│   ├── core/           # Base classes, config, exceptions
│   ├── annotation/     # Annotation CRUD, storage, statistics
│   ├── evaluation/     # Metrics, RAG adapters, evaluation runner
│   ├── display/        # Gradio UI components
│   └── utils/          # Async helpers, validators, file handlers
├── tests/              # Unit tests (pytest)
├── data/               # Local data storage
└── configs/            # Configuration files
```

## Design Patterns (Required)

All code must follow these patterns for maintainability:

- **Strategy Pattern**: Metric implementations - each metric is an independent strategy
- **Factory Pattern**: Metric instantiation, storage backend creation
- **Adapter Pattern**: RAG interface adapters, data format converters
- **Template Method**: Standardized evaluation/annotation workflows
- **Singleton Pattern**: Global config, database connections, concurrency controller
- **Iterator Pattern**: Annotation data and result traversal

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python -m src.main

# Run tests
pytest tests/ -v

# Run single test file
pytest tests/test_annotation.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
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

### Metrics Development
New metrics extend `BaseMetric` class in `src/evaluation/metrics/base.py`. No core code changes needed.

## Environment Variables

Required variables (see `.env.example`):
- `OPENAI_API_KEY`: For LLM-based evaluation
- `RAG_SERVICE_URL`: Remote RAG endpoint
- `DATABASE_URL`: SQLite connection string (optional)