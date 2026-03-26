"""
RAG module: adapters for RAG service integration.
"""

from .base_adapter import RAGAdapter, RAGAdapterConfig, StreamingChunk
from .langgraph_adapter import LangGraphAdapter
from .langgraph_sse_adapter import LangGraphSSEAdapter, LangGraphSSEAdapterConfig
from .mock_adapter import MockRAGAdapter
from .response_parser import RAGResponseParser
from .sse_parser import (
    SSEEventType,
    SSEEvent,
    ContentBlock,
    IntermediateResult,
    SSEEventParser,
    SSEStreamAccumulator,
)

__all__ = [
    "RAGAdapter",
    "RAGAdapterConfig",
    "StreamingChunk",
    "LangGraphAdapter",
    "LangGraphSSEAdapter",
    "LangGraphSSEAdapterConfig",
    "MockRAGAdapter",
    "RAGResponseParser",
    # SSE parser exports
    "SSEEventType",
    "SSEEvent",
    "ContentBlock",
    "IntermediateResult",
    "SSEEventParser",
    "SSEStreamAccumulator",
]