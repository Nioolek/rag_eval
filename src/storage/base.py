"""
Abstract storage backend interface.
Defines the contract for all storage implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.
    Implements the Strategy pattern for storage operations.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the storage backend and release resources."""
        pass

    @abstractmethod
    async def save(self, collection: str, data: dict[str, Any]) -> str:
        """
        Save data to the storage.

        Args:
            collection: Collection/table name
            data: Data to save

        Returns:
            ID of the saved record
        """
        pass

    @abstractmethod
    async def get(self, collection: str, record_id: str) -> Optional[dict[str, Any]]:
        """
        Get a single record by ID.

        Args:
            collection: Collection/table name
            record_id: Record ID

        Returns:
            Record data or None if not found
        """
        pass

    @abstractmethod
    async def get_all(
        self,
        collection: str,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Get all records from a collection with optional filtering.

        Args:
            collection: Collection/table name
            filters: Optional filter conditions
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of records
        """
        pass

    @abstractmethod
    async def update(
        self,
        collection: str,
        record_id: str,
        data: dict[str, Any]
    ) -> bool:
        """
        Update a record.

        Args:
            collection: Collection/table name
            record_id: Record ID
            data: Updated data

        Returns:
            True if updated successfully
        """
        pass

    @abstractmethod
    async def delete(self, collection: str, record_id: str) -> bool:
        """
        Delete a record.

        Args:
            collection: Collection/table name
            record_id: Record ID

        Returns:
            True if deleted successfully
        """
        pass

    @abstractmethod
    async def count(
        self,
        collection: str,
        filters: Optional[dict[str, Any]] = None
    ) -> int:
        """
        Count records in a collection.

        Args:
            collection: Collection/table name
            filters: Optional filter conditions

        Returns:
            Number of records
        """
        pass

    @abstractmethod
    async def iterate(
        self,
        collection: str,
        batch_size: int = 100,
        filters: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Iterate over records in batches.

        Args:
            collection: Collection/table name
            batch_size: Number of records per batch
            filters: Optional filter conditions

        Yields:
            Individual records
        """
        pass

    @abstractmethod
    async def save_version(
        self,
        collection: str,
        record_id: str,
        version_data: dict[str, Any]
    ) -> int:
        """
        Save a version of a record for version management.

        Args:
            collection: Collection/table name
            record_id: Record ID
            version_data: Version data to save

        Returns:
            Version number
        """
        pass

    @abstractmethod
    async def get_versions(
        self,
        collection: str,
        record_id: str
    ) -> list[dict[str, Any]]:
        """
        Get all versions of a record.

        Args:
            collection: Collection/table name
            record_id: Record ID

        Returns:
            List of versions
        """
        pass