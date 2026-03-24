"""
Test configuration and fixtures.

Provides shared fixtures for RAG Pipeline testing.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"


# === Event Loop ===

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# === Sample Data Fixtures ===

@pytest.fixture
def sample_query() -> str:
    """Sample query for testing."""
    return "iPhone如何截屏"


@pytest.fixture
def sample_faq_data() -> list[dict[str, Any]]:
    """Sample FAQ data for testing."""
    return [
        {
            "id": "faq-001",
            "question": "iPhone 15 Pro Max如何截屏？",
            "answer": "Apple iPhone 15 Pro Max截屏方法：同时按住电源键和音量加键，屏幕闪烁即表示截屏成功。",
            "category": "手机操作",
            "keywords": ["iPhone", "截屏"],
        },
        {
            "id": "faq-002",
            "question": "华为Mate 60 Pro续航怎么样？",
            "answer": "华为Mate 60 Pro配备5000mAh大电池，正常使用可续航一整天。",
            "category": "手机续航",
            "keywords": ["华为", "续航"],
        },
        {
            "id": "faq-003",
            "question": "MacBook Pro如何连接外接显示器？",
            "answer": "使用USB-C或HDMI转接线连接显示器，系统会自动识别。",
            "category": "电脑操作",
            "keywords": ["MacBook", "显示器"],
        },
    ]


@pytest.fixture
def sample_documents() -> list[dict[str, Any]]:
    """Sample documents for testing."""
    return [
        {
            "id": "doc-001",
            "content": "iPhone 15 Pro Max 是 Apple 旗舰手机，配备 A17 Pro 芯片，支持高刷屏幕。",
            "metadata": {
                "title": "iPhone 15 Pro Max 介绍",
                "category": "产品介绍",
                "brand": "Apple",
            },
        },
        {
            "id": "doc-002",
            "content": "华为 Mate 60 Pro 搭载麒麟 9000S 芯片，支持卫星通话功能。",
            "metadata": {
                "title": "华为 Mate 60 Pro 介绍",
                "category": "产品介绍",
                "brand": "华为",
            },
        },
        {
            "id": "doc-003",
            "content": "MacBook Pro 配备 M3 芯片，性能强劲，适合专业创作。",
            "metadata": {
                "title": "MacBook Pro 介绍",
                "category": "产品介绍",
                "brand": "Apple",
            },
        },
    ]


@pytest.fixture
def sample_retrieval_results() -> list[dict[str, Any]]:
    """Sample retrieval results for testing."""
    return [
        {
            "document_id": "doc-001",
            "content": "iPhone 15 Pro Max 是 Apple 旗舰手机...",
            "score": 0.85,
            "source": "vector",
            "metadata": {"title": "iPhone 介绍"},
        },
        {
            "document_id": "doc-002",
            "content": "华为 Mate 60 Pro 搭载麒麟芯片...",
            "score": 0.75,
            "source": "fulltext",
            "metadata": {"title": "华为介绍"},
        },
    ]


# === State Fixtures ===

@pytest.fixture
def sample_state(sample_query: str) -> dict[str, Any]:
    """Sample RAG state for testing."""
    from rag_rag.graph.state import create_initial_state
    return create_initial_state(query=sample_query)


@pytest.fixture
def state_with_faq_matched(sample_state: dict) -> dict[str, Any]:
    """State with FAQ matched."""
    sample_state["faq_matched"] = True
    sample_state["faq_result"] = {
        "matched": True,
        "faq_id": "faq-001",
        "question": "iPhone 15 Pro Max如何截屏？",
        "answer": "同时按住电源键和音量加键...",
        "confidence": 0.95,
    }
    return sample_state


@pytest.fixture
def state_with_retrieval_results(sample_state: dict) -> dict[str, Any]:
    """State with retrieval results."""
    sample_state["faq_matched"] = False
    sample_state["vector_results"] = [
        {"document_id": "doc-001", "content": "内容1", "score": 0.85, "source": "vector", "metadata": {}}
    ]
    sample_state["fulltext_results"] = [
        {"document_id": "doc-002", "content": "内容2", "score": 5.5, "source": "fulltext", "metadata": {}}
    ]
    sample_state["merged_results"] = [
        {"document_id": "doc-001", "content": "内容1", "combined_score": 0.425, "metadata": {}},
        {"document_id": "doc-002", "content": "内容2", "combined_score": 0.165, "metadata": {}},
    ]
    sample_state["reranked_results"] = [
        {"document_id": "doc-001", "content": "内容1", "rerank_score": 0.9, "rank": 1, "metadata": {}},
    ]
    return sample_state


@pytest.fixture
def state_should_refuse(sample_state: dict) -> dict[str, Any]:
    """State that should trigger refusal."""
    sample_state["faq_matched"] = False
    sample_state["merged_results"] = []
    sample_state["reranked_results"] = []
    sample_state["should_refuse"] = True
    sample_state["refusal_reason"] = "知识库中没有找到相关信息"
    sample_state["refusal_type"] = "out_of_domain"
    return sample_state


# === Mock Fixtures ===

@pytest.fixture
def mock_embedding_service():
    """Mock EmbeddingService for testing."""
    mock = MagicMock()
    mock.embed = AsyncMock(return_value=[[0.1] * 1024 for _ in range(10)])
    mock.embed_single = AsyncMock(return_value=[0.1] * 1024)
    mock.initialize = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_llm_service():
    """Mock LLMService for testing."""
    mock = MagicMock()
    mock.generate = AsyncMock(return_value=MagicMock(
        content="这是一个测试回答。",
        thinking_process="",
        token_usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        model="qwen-plus",
        finish_reason="stop",
    ))
    mock.initialize = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_faq_store(sample_faq_data: list[dict]):
    """Mock FAQStore for testing."""
    mock = MagicMock()

    async def mock_search(query: str, top_k: int = 5, **kwargs):
        # Simulate exact match
        for faq in sample_faq_data:
            if query in faq["question"] or faq["question"] in query:
                return [{
                    "id": faq["id"],
                    "question": faq["question"],
                    "answer": faq["answer"],
                    "score": 0.95,
                    "match_type": "exact",
                }]
        return []

    mock.search = AsyncMock(side_effect=mock_search)
    mock.get = AsyncMock(return_value=sample_faq_data[0])
    mock.keys = AsyncMock(return_value=[f["id"] for f in sample_faq_data])
    mock.initialize = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_vector_store(sample_documents: list[dict]):
    """Mock VectorStore for testing."""
    mock = MagicMock()

    async def mock_search(query, top_k: int = 10, **kwargs):
        return [
            {
                "document_id": doc["id"],
                "content": doc["content"],
                "score": 0.8 - i * 0.1,
                "source": "vector",
                "metadata": doc["metadata"],
            }
            for i, doc in enumerate(sample_documents[:top_k])
        ]

    mock.search = AsyncMock(side_effect=mock_search)
    mock.count = AsyncMock(return_value=len(sample_documents))
    mock.initialize = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_fulltext_store(sample_documents: list[dict]):
    """Mock FulltextStore for testing."""
    mock = MagicMock()

    async def mock_search(query: str, top_k: int = 10, **kwargs):
        return [
            {
                "document_id": doc["id"],
                "content": doc["content"],
                "score": 8.0 - i * 1.5,  # BM25 style scores
                "source": "fulltext",
                "metadata": doc["metadata"],
            }
            for i, doc in enumerate(sample_documents[:top_k])
        ]

    mock.search = AsyncMock(side_effect=mock_search)
    mock.count = AsyncMock(return_value=len(sample_documents))
    mock.initialize = AsyncMock()
    mock.close = AsyncMock()
    return mock


# === Environment Fixtures ===

@pytest.fixture
def with_api_key(monkeypatch):
    """Set DASHSCOPE_API_KEY environment variable."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-api-key")


@pytest.fixture
def without_api_key(monkeypatch):
    """Remove DASHSCOPE_API_KEY environment variable."""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)


@pytest.fixture
def with_placeholder_api_key(monkeypatch):
    """Set placeholder API key (should be treated as missing)."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-placeholder-key")


# === Temporary Directory Fixtures ===

@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_faq_db(temp_data_dir: Path):
    """Create temporary FAQ database."""
    return temp_data_dir / "test_faq.db"


@pytest.fixture
def temp_chroma_dir(temp_data_dir: Path):
    """Create temporary Chroma directory."""
    return temp_data_dir / "chroma"


@pytest.fixture
def temp_whoosh_dir(temp_data_dir: Path):
    """Create temporary Whoosh directory."""
    return temp_data_dir / "whoosh"


# === Utility Fixtures ===

@pytest.fixture
def assert_timing_recorded():
    """Assert that timing is recorded in state."""
    def _assert(state: dict, stage: str):
        timing = state.get("stage_timing", {})
        key = f"{stage}_ms"
        assert key in timing, f"Missing timing for {stage}"
        assert timing[key] >= 0, f"Invalid timing for {stage}"
    return _assert


@pytest.fixture
def assert_no_errors():
    """Assert that no errors occurred in state."""
    def _assert(state: dict):
        errors = state.get("errors", [])
        assert len(errors) == 0, f"Unexpected errors: {errors}"
    return _assert