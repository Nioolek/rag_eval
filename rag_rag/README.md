# RAG Pipeline Service

Enterprise-grade RAG system built with LangGraph for knowledge base Q&A scenarios.

## Features

- **LangGraph StateGraph**: 14-node pipeline with parallel retrieval
- **Multiple Retrieval Sources**: Chroma (vector), Whoosh (fulltext), Neo4j (graph)
- **FAQ-First Strategy**: Direct FAQ matching for common questions
- **Query Rewriting**: LLM-powered query expansion and clarification
- **Reranking**: Alibaba gte-rerank with BM25 fallback
- **Refusal Logic**: Out-of-domain, sensitive, and low-relevance detection
- **Degradation Strategies**: Circuit breaker and fallback handlers
- **Hot-Reload Config**: YAML-based configuration with watchdog

## Architecture

```
Query → FAQ Match → Query Rewrite → Parallel Retrieval (Vector|Fulltext|Graph)
                                              ↓
                                          Merge → Rerank → Build Prompt
                                              ↓           ↓
                                       Refusal Check → Generate
                                              ↓
                                           Output
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env`
2. Configure API keys and service URLs
3. Edit `config/settings.yaml` for business configuration

## Running

### Development Mode

```bash
cd rag_rag
langgraph dev
```

### Production Mode

```bash
langgraph up --port 8123
```

### Docker

```bash
docker-compose up -d
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/threads` | POST | Create conversation |
| `/threads/{id}/runs` | POST | Execute query |
| `/threads/{id}/runs/stream` | POST | Stream execution |
| `/health` | GET | Health check |

## Integration with Evaluation System

The RAG service outputs data in a format compatible with `RAGResponseAdapter.from_langgraph()`:

```python
from langgraph.pregel.remote import RemoteGraph

rag_client = RemoteGraph("http://localhost:8123")
result = await rag_client.ainvoke({"query": "如何申请年假？"})
```

## Development

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/rag_rag
```