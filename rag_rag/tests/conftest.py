"""Test configuration and fixtures."""

import pytest
import asyncio
from pathlib import Path

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "如何申请年假？"


@pytest.fixture
def sample_state(sample_query):
    """Sample RAG state for testing."""
    from rag_rag.graph.state import RAGState
    return RAGState(
        query=sample_query,
        conversation_history=[],
        conversation_id="test-conv-001",
        agent_id="test-agent",
        enable_thinking=False,
    )