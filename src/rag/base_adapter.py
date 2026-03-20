"""
Abstract RAG adapter interface.
Implements Adapter pattern for different RAG implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, AsyncGenerator
from dataclasses import dataclass

from ..models.rag_response import RAGResponse
from ..models.annotation import Annotation


@dataclass
class StreamingChunk:
    """A chunk of streaming output from RAG."""

    stage: str  # e.g., "query_rewrite", "retrieval", "rerank", "generation"
    content: str
    is_final: bool = False
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RAGAdapter(ABC):
    """
    Abstract base class for RAG adapters.
    Implements the Adapter pattern for different RAG services.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get adapter name."""
        pass

    @abstractmethod
    async def query(
        self,
        query: str,
        conversation_history: Optional[list[str]] = None,
        agent_id: Optional[str] = None,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> RAGResponse:
        """
        Send a query to the RAG service.

        Args:
            query: User query
            conversation_history: Multi-turn conversation history
            agent_id: Agent identifier
            enable_thinking: Enable thinking mode
            **kwargs: Additional parameters

        Returns:
            RAGResponse with all pipeline results
        """
        pass

    @abstractmethod
    async def query_from_annotation(
        self,
        annotation: Annotation
    ) -> RAGResponse:
        """
        Query RAG using annotation data.

        Args:
            annotation: Annotation containing query data

        Returns:
            RAGResponse
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if RAG service is healthy.

        Returns:
            True if healthy
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close adapter and release resources."""
        pass

    async def stream_query(
        self,
        query: str,
        conversation_history: Optional[list[str]] = None,
        agent_id: Optional[str] = None,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamingChunk, None]:
        """
        Stream query execution with intermediate results.

        Yields StreamingChunk objects for each stage of RAG processing.
        Default implementation yields a single chunk with the final result.

        Args:
            query: User query
            conversation_history: Multi-turn conversation history
            agent_id: Agent identifier
            enable_thinking: Enable thinking mode
            **kwargs: Additional parameters

        Yields:
            StreamingChunk objects with stage information
        """
        # Default implementation: just query and yield final result
        response = await self.query(
            query=query,
            conversation_history=conversation_history,
            agent_id=agent_id,
            enable_thinking=enable_thinking,
            **kwargs,
        )

        # Yield final answer
        yield StreamingChunk(
            stage="final",
            content=response.final_answer,
            is_final=True,
            metadata={"success": response.success, "latency_ms": response.latency_ms}
        )


class RAGAdapterConfig:
    """Configuration for RAG adapters."""

    def __init__(
        self,
        service_url: str,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ):
        self.service_url = service_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.extra = kwargs