"""
Storage factory using Factory pattern.
Creates appropriate storage backend based on configuration.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from .base import StorageBackend
from .local_storage import LocalStorage
from .sqlite_storage import SQLiteStorage
from ..core.config import get_config
from ..core.exceptions import ConfigurationError, StorageError
from ..core.logging import logger


class StorageFactory:
    """
    Factory for creating storage backends.
    Implements the Factory pattern for storage instantiation.
    """

    _instances: dict[str, StorageBackend] = {}

    @classmethod
    async def create(
        cls,
        storage_type: Optional[str] = None,
        data_dir: Optional[Path] = None,
        database_url: Optional[str] = None,
        chunk_size: int = 8192,
    ) -> StorageBackend:
        """
        Create a storage backend instance.

        Args:
            storage_type: "local" or "sqlite", defaults to config
            data_dir: Data directory for local storage
            database_url: Database URL for SQLite
            chunk_size: Chunk size for file operations

        Returns:
            StorageBackend instance
        """
        # Get defaults from config
        config = get_config()
        storage_type = storage_type or config.storage.storage_type
        data_dir = data_dir or config.storage.data_dir
        database_url = database_url or config.storage.database_url
        chunk_size = chunk_size or config.storage.chunk_size

        # Check for cached instance
        cache_key = f"{storage_type}:{data_dir}:{database_url}"
        if cache_key in cls._instances:
            return cls._instances[cache_key]

        # Create new instance
        if storage_type == "local":
            storage = LocalStorage(
                data_dir=data_dir,
                chunk_size=chunk_size
            )
        elif storage_type == "sqlite":
            if not database_url:
                # Default SQLite path
                database_url = f"sqlite:///{data_dir}/rag_eval.db"
            storage = SQLiteStorage(database_url=database_url)
        else:
            raise ConfigurationError(
                f"Unknown storage type: {storage_type}. "
                "Supported types: 'local', 'sqlite'"
            )

        # Initialize storage
        await storage.initialize()
        cls._instances[cache_key] = storage

        logger.info(f"Created {storage_type} storage backend")
        return storage

    @classmethod
    async def close_all(cls) -> None:
        """Close all storage instances."""
        for storage in cls._instances.values():
            await storage.close()
        cls._instances.clear()
        logger.info("Closed all storage backends")

    @classmethod
    def get_instance(cls, cache_key: str) -> Optional[StorageBackend]:
        """Get an existing storage instance by cache key."""
        return cls._instances.get(cache_key)


# Singleton storage instance
_storage_instance: Optional[StorageBackend] = None


async def get_storage() -> StorageBackend:
    """
    Get the singleton storage instance.
    Creates the instance if not exists.
    """
    global _storage_instance

    if _storage_instance is None:
        config = get_config()
        _storage_instance = await StorageFactory.create(
            storage_type=config.storage.storage_type,
            data_dir=config.storage.data_dir,
            database_url=config.storage.database_url,
            chunk_size=config.storage.chunk_size,
        )

    return _storage_instance


async def init_storage() -> StorageBackend:
    """Initialize and return storage backend."""
    return await get_storage()