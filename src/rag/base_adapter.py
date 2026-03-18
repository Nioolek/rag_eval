"""
Abstract RAG adapter interface.
Implements Adapter pattern for different RAG implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..models.rag_response import RAGResponse
from ..models.annotation import Annotation


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