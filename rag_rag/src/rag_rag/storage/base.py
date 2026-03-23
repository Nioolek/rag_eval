"""
Abstract Storage Interfaces for RAG Pipeline.

Defines base classes for all storage backends following the Strategy pattern.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Optional


class StoreStatus(Enum):
    """Storage backend status."""

    UNINITIALIZED = "uninitialized"
    READY = "ready"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class StoreInfo:
    """Storage backend information."""

    name: str
    status: StoreStatus
    document_count: int = 0
    last_updated: Optional[str] = None
    error_message: Optional[str] = None


class BaseStore(ABC):
    """
    Abstract base class for all storage backends.

    Defines the common interface that all storage implementations must follow.
    Uses the Strategy pattern to allow swapping storage backends.
    """

    def __init__(self, name: str):
        self.name = name
        self._status = StoreStatus.UNINITIALIZED
        self._error_message: Optional[str] = None

    @property
    def status(self) -> StoreStatus:
        """Get current store status."""
        return self._status

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the storage backend.

        This method should:
        - Create necessary directories/files/tables
        - Establish connections
        - Set up indexes
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the storage backend is healthy.

        Returns:
            True if healthy, False otherwise
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close the storage backend and release resources.

        This method should:
        - Flush any pending writes
        - Close connections
        - Clean up resources
        """
        pass

    def get_info(self) -> StoreInfo:
        """Get storage backend information."""
        return StoreInfo(
            name=self.name,
            status=self._status,
            error_message=self._error_message,
        )

    def _set_ready(self) -> None:
        """Mark store as ready."""
        self._status = StoreStatus.READY
        self._error_message = None

    def _set_error(self, error: str) -> None:
        """Mark store as error."""
        self._status = StoreStatus.ERROR
        self._error_message = error

    def _set_closed(self) -> None:
        """Mark store as closed."""
        self._status = StoreStatus.CLOSED


class RetrievableStore(BaseStore):
    """
    Abstract base class for retrievable storage backends.

    Extends BaseStore with search/retrieval capabilities.
    Used for Vector, Fulltext, and Graph stores.
    """

    @abstractmethod
    async def add(
        self,
        documents: list[dict[str, Any]],
        **kwargs: Any,
    ) -> int:
        """
        Add documents to the store.

        Args:
            documents: List of documents with 'id', 'content', and 'metadata'

        Returns:
            Number of documents added
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str | list[float],
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Search for documents.

        Args:
            query: Search query (string for fulltext, embedding for vector)
            top_k: Maximum number of results

        Returns:
            List of search results with 'id', 'content', 'score', 'metadata'
        """
        pass

    @abstractmethod
    async def delete(self, document_ids: list[str]) -> int:
        """
        Delete documents by IDs.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            Number of documents deleted
        """
        pass

    @abstractmethod
    async def get(self, document_id: str) -> Optional[dict[str, Any]]:
        """
        Get a document by ID.

        Args:
            document_id: Document ID

        Returns:
            Document dict or None if not found
        """
        pass

    @abstractmethod
    async def count(self) -> int:
        """
        Get total document count.

        Returns:
            Number of documents in the store
        """
        pass

    async def update(
        self,
        document_id: str,
        document: dict[str, Any],
    ) -> bool:
        """
        Update a document.

        Args:
            document_id: Document ID
            document: Updated document data

        Returns:
            True if updated, False if not found
        """
        doc = await self.get(document_id)
        if doc is None:
            return False

        await self.delete([document_id])
        document["id"] = document_id
        await self.add([document])
        return True

    async def clear(self) -> int:
        """
        Clear all documents from the store.

        Returns:
            Number of documents cleared
        """
        count = await self.count()
        # Subclasses should override this for better performance
        return count


class KeyValueStore(BaseStore):
    """
    Abstract base class for key-value storage backends.

    Used for FAQ and Session stores.
    """

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value by key.

        Args:
            key: Key to look up

        Returns:
            Value or None if not found
        """
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set key-value pair.

        Args:
            key: Key
            value: Value
            ttl: Optional time-to-live in seconds
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete by key.

        Args:
            key: Key to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if key exists.

        Args:
            key: Key to check

        Returns:
            True if exists
        """
        pass

    @abstractmethod
    async def keys(self, pattern: Optional[str] = None) -> list[str]:
        """
        List keys matching pattern.

        Args:
            pattern: Optional pattern to match (e.g., "session:*")

        Returns:
            List of matching keys
        """
        pass


class SearchableKeyValueStore(KeyValueStore):
    """
    Abstract base class for searchable key-value stores.

    Extends KeyValueStore with search capabilities.
    Used for FAQ store with semantic search.
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10,
        search_type: str = "semantic",  # exact, semantic, hybrid
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Search for entries.

        Args:
            query: Search query
            top_k: Maximum results
            search_type: Type of search (exact, semantic, hybrid)

        Returns:
            List of search results
        """
        pass