"""
Local file storage backend.
Implements JSONL-based storage with chunked read/write support.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import aiofiles

from .base import StorageBackend
from ..core.exceptions import StorageError, PathTraversalError
from ..core.logging import logger


class LocalStorage(StorageBackend):
    """
    Local file storage using JSONL format.
    Supports chunked read/write for large files.
    """

    def __init__(self, data_dir: Path, chunk_size: int = 8192):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self._lock: Optional[asyncio.Lock] = None  # Lazy-initialized lock

    def _get_lock(self) -> asyncio.Lock:
        """Get or create the lock (lazy initialization)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _get_collection_path(self, collection: str) -> Path:
        """Get the file path for a collection with path traversal protection."""
        # Check for path traversal attempts
        if ".." in collection or "/" in collection or "\\" in collection:
            raise PathTraversalError(
                f"Path traversal attempt detected: {collection}"
            )

        path = self.data_dir / f"{collection}.jsonl"

        # Ensure path is within data_dir
        try:
            path.resolve().relative_to(self.data_dir.resolve())
        except ValueError:
            raise PathTraversalError(
                f"Path traversal attempt detected: {collection}"
            )

        return path

    def _get_version_path(self, collection: str, record_id: str) -> Path:
        """Get the version file path for a record."""
        record_id = record_id.replace("..", "").replace("/", "_").replace("\\", "_")
        version_dir = self.data_dir / "versions" / collection
        return version_dir / f"{record_id}.jsonl"

    async def initialize(self) -> None:
        """Create data directory if not exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "versions").mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized local storage at {self.data_dir}")

    async def close(self) -> None:
        """No resources to close for file storage."""
        pass

    async def save(self, collection: str, data: dict[str, Any]) -> str:
        """Save data to JSONL file."""
        path = self._get_collection_path(collection)

        # Generate ID if not present
        if "id" not in data:
            from uuid import uuid4
            data["id"] = str(uuid4())

        data["_saved_at"] = datetime.now().isoformat()

        async with self._get_lock():
            async with aiofiles.open(path, mode='a', encoding='utf-8') as f:
                await f.write(json.dumps(data, ensure_ascii=False) + '\n')

        logger.debug(f"Saved record {data['id']} to {collection}")
        return data["id"]

    async def get(self, collection: str, record_id: str) -> Optional[dict[str, Any]]:
        """Get a record by ID using chunked reading."""
        path = self._get_collection_path(collection)

        if not path.exists():
            return None

        # Use read lock for consistency during concurrent writes
        async with self._get_lock():
            async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("id") == record_id and not data.get("is_deleted"):
                            return data
                    except json.JSONDecodeError:
                        continue

        return None

    async def get_all(
        self,
        collection: str,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get all records with optional filtering."""
        path = self._get_collection_path(collection)

        if not path.exists():
            return []

        results = []
        skipped = 0

        # Use read lock for consistency during concurrent writes
        async with self._get_lock():
            async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)

                        # Skip deleted records
                        if data.get("is_deleted"):
                            continue

                        # Apply filters
                        if filters:
                            match = all(
                                data.get(k) == v or
                                (isinstance(v, list) and data.get(k) in v) or
                                (isinstance(v, dict) and data.get(k) is not None)
                                for k, v in filters.items()
                            )
                            if not match:
                                continue

                        # Apply offset
                        if skipped < offset:
                            skipped += 1
                            continue

                        results.append(data)

                        if len(results) >= limit:
                            break

                    except json.JSONDecodeError:
                        continue

        return results

    async def update(
        self,
        collection: str,
        record_id: str,
        data: dict[str, Any]
    ) -> bool:
        """Update a record by rewriting the file."""
        path = self._get_collection_path(collection)

        if not path.exists():
            return False

        # Lock the entire operation (read + write) to prevent race conditions
        async with self._get_lock():
            # Read all records
            records = []
            found = False

            async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if record.get("id") == record_id:
                            record.update(data)
                            record["updated_at"] = datetime.now().isoformat()
                            found = True
                        records.append(record)
                    except json.JSONDecodeError:
                        continue

            if not found:
                return False

            # Rewrite file
            async with aiofiles.open(path, mode='w', encoding='utf-8') as f:
                for record in records:
                    await f.write(json.dumps(record, ensure_ascii=False) + '\n')

        return True

    async def delete(self, collection: str, record_id: str) -> bool:
        """Soft delete a record."""
        return await self.update(collection, record_id, {"is_deleted": True})

    async def count(
        self,
        collection: str,
        filters: Optional[dict[str, Any]] = None
    ) -> int:
        """Count records in collection."""
        path = self._get_collection_path(collection)

        if not path.exists():
            return 0

        count = 0
        # Use read lock for consistency during concurrent writes
        async with self._get_lock():
            async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("is_deleted"):
                            continue
                        if filters:
                            match = all(data.get(k) == v for k, v in filters.items())
                            if not match:
                                continue
                        count += 1
                    except json.JSONDecodeError:
                        continue

        return count

    async def iterate(
        self,
        collection: str,
        batch_size: int = 100,
        filters: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Iterate over records for memory-efficient processing."""
        path = self._get_collection_path(collection)

        if not path.exists():
            return

        # Use read lock for consistency during concurrent writes
        async with self._get_lock():
            async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("is_deleted"):
                            continue
                        if filters:
                            match = all(data.get(k) == v for k, v in filters.items())
                            if not match:
                                continue
                        yield data
                    except json.JSONDecodeError:
                        continue

    async def save_version(
        self,
        collection: str,
        record_id: str,
        version_data: dict[str, Any]
    ) -> int:
        """Save a version snapshot."""
        version_path = self._get_version_path(collection, record_id)
        version_path.parent.mkdir(parents=True, exist_ok=True)

        # Get current version count
        current_version = 0
        if version_path.exists():
            async with aiofiles.open(version_path, mode='r', encoding='utf-8') as f:
                async for line in f:
                    current_version += 1

        version_data["version_number"] = current_version + 1
        version_data["versioned_at"] = datetime.now().isoformat()

        async with self._get_lock():
            async with aiofiles.open(version_path, mode='a', encoding='utf-8') as f:
                await f.write(json.dumps(version_data, ensure_ascii=False) + '\n')

        return version_data["version_number"]

    async def get_versions(
        self,
        collection: str,
        record_id: str
    ) -> list[dict[str, Any]]:
        """Get all versions of a record."""
        version_path = self._get_version_path(collection, record_id)

        if not version_path.exists():
            return []

        versions = []
        # Use read lock for consistency during concurrent writes
        async with self._get_lock():
            async with aiofiles.open(version_path, mode='r', encoding='utf-8') as f:
                async for line in f:
                    line = line.strip()
                    if line:
                        versions.append(json.loads(line))

        return versions