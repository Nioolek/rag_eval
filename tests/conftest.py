"""Test configuration and fixtures."""

import asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio

# Set up test environment
import os
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["DATA_DIR"] = "./test_data"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def temp_storage():
    """Create temporary storage for testing."""
    from src.storage.local_storage import LocalStorage

    temp_dir = Path("./test_data")
    temp_dir.mkdir(parents=True, exist_ok=True)

    storage = LocalStorage(temp_dir)
    await storage.initialize()

    yield storage

    # Cleanup
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_annotation_data():
    """Sample annotation data for testing."""
    return {
        "query": "什么是机器学习？",
        "conversation_history": ["你好", "请问有什么可以帮助你的？"],
        "agent_id": "test-agent",
        "language": "zh",
        "enable_thinking": False,
        "gt_documents": ["机器学习是人工智能的一个分支..."],
        "faq_matched": False,
        "should_refuse": False,
        "standard_answers": ["机器学习是一种使计算机能够从数据中学习的技术。"],
        "answer_style": "专业",
        "notes": "测试标注",
    }


@pytest.fixture
def sample_rag_response_data():
    """Sample RAG response data for testing."""
    return {
        "query": "什么是机器学习？",
        "query_rewrite": {
            "rewritten_query": "请解释机器学习的概念和原理",
            "type": "expansion",
            "confidence": 0.9,
        },
        "faq_match": {
            "matched": False,
        },
        "retrieval": [
            {"id": "doc1", "content": "机器学习是人工智能的一个分支...", "score": 0.95},
            {"id": "doc2", "content": "机器学习算法可以从数据中学习...", "score": 0.88},
        ],
        "answer": "机器学习是一种使计算机系统能够从数据中学习并改进的技术，无需明确编程。",
    }