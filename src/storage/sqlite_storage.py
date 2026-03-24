"""
SQLite storage backend.
Implements database storage with indexing and async support.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import aiosqlite

from .base import StorageBackend
from ..core.exceptions import StorageError
from ..core.logging import logger


class SQLiteStorage(StorageBackend):
    """
    SQLite storage backend with async support.
    Includes proper indexing for performance.
    """

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._db: Optional[aiosqlite.Connection] = None
        self._lock: Optional[asyncio.Lock] = None  # Lazy-initialized lock

    def _get_lock(self) -> asyncio.Lock:
        """Get or create the lock (lazy initialization)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def initialize(self) -> None:
        """Initialize database connection and create tables."""
        # Ensure parent directory exists for SQLite file
        if self.database_url.startswith("sqlite:///"):
            db_path = Path(self.database_url[10:])
            db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(self.database_url.replace("sqlite:///", ""))
        self._db.row_factory = aiosqlite.Row

        # Enable WAL mode for better concurrency
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")

        await self._create_tables()
        logger.info(f"Initialized SQLite storage: {self.database_url}")

    async def _create_tables(self) -> None:
        """Create necessary tables with indexes."""
        # Main data table
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id TEXT PRIMARY KEY,
                collection TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                is_deleted INTEGER DEFAULT 0
            )
        """)

        # Indexes for common queries
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_records_collection
            ON records(collection)
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_records_collection_deleted
            ON records(collection, is_deleted)
        """)

        # Version history table
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL,
                collection TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                data TEXT NOT NULL,
                versioned_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_versions_record
            ON versions(collection, record_id)
        """)

        await self._db.commit()

    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def save(self, collection: str, data: dict[str, Any]) -> str:
        """Save data to SQLite."""
        from uuid import uuid4

        if "id" not in data:
            data["id"] = str(uuid4())

        now = datetime.now().isoformat()

        async with self._get_lock():
            await self._db.execute(
                """
                INSERT INTO records (id, collection, data, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (data["id"], collection, json.dumps(data, ensure_ascii=False), now, now)
            )
            await self._db.commit()

        logger.debug(f"Saved record {data['id']} to {collection}")
        return data["id"]

    async def get(self, collection: str, record_id: str) -> Optional[dict[str, Any]]:
        """Get a record by ID."""
        async with self._get_lock():
            cursor = await self._db.execute(
                """
                SELECT data FROM records
                WHERE id = ? AND collection = ? AND is_deleted = 0
                """,
                (record_id, collection)
            )
            row = await cursor.fetchone()

        if row:
            return json.loads(row["data"])
        return None

    async def get_all(
        self,
        collection: str,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get all records with filtering."""
        query = """
            SELECT data FROM records
            WHERE collection = ? AND is_deleted = 0
        """
        params = [collection]

        # Note: Complex filtering would require JSON path queries
        # For simplicity, we filter in Python after fetching

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with self._get_lock():
            cursor = await self._db.execute(query, params)
            rows = await cursor.fetchall()

        results = []
        for row in rows:
            data = json.loads(row["data"])
            if filters:
                match = all(
                    data.get(k) == v or
                    (isinstance(v, list) and data.get(k) in v)
                    for k, v in filters.items()
                )
                if not match:
                    continue
            results.append(data)

        return results

    async def update(
        self,
        collection: str,
        record_id: str,
        data: dict[str, Any]
    ) -> bool:
        """Update a record."""
        # First get existing data
        existing = await self.get(collection, record_id)
        if not existing:
            return False

        existing.update(data)
        now = datetime.now().isoformat()

        async with self._get_lock():
            await self._db.execute(
                """
                UPDATE records SET data = ?, updated_at = ?
                WHERE id = ? AND collection = ?
                """,
                (json.dumps(existing, ensure_ascii=False), now, record_id, collection)
            )
            await self._db.commit()

        return True

    async def delete(self, collection: str, record_id: str) -> bool:
        """Soft delete a record."""
        async with self._get_lock():
            await self._db.execute(
                """
                UPDATE records SET is_deleted = 1, updated_at = ?
                WHERE id = ? AND collection = ?
                """,
                (datetime.now().isoformat(), record_id, collection)
            )
            await self._db.commit()

        return True

    async def count(
        self,
        collection: str,
        filters: Optional[dict[str, Any]] = None
    ) -> int:
        """Count records in collection."""
        if not filters:
            async with self._get_lock():
                cursor = await self._db.execute(
                    """
                    SELECT COUNT(*) as cnt FROM records
                    WHERE collection = ? AND is_deleted = 0
                    """,
                    (collection,)
                )
                row = await cursor.fetchone()
            return row["cnt"] if row else 0

        # With filters, need to check each record
        # This is a limitation of storing JSON as text
        all_records = await self.get_all(collection, filters, limit=100000)
        return len(all_records)

    async def iterate(
        self,
        collection: str,
        batch_size: int = 100,
        filters: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Iterate over records."""
        offset = 0
        while True:
            records = await self.get_all(
                collection, filters=filters, limit=batch_size, offset=offset
            )
            if not records:
                break
            for record in records:
                yield record
            offset += batch_size

    async def save_version(
        self,
        collection: str,
        record_id: str,
        version_data: dict[str, Any]
    ) -> int:
        """Save a version snapshot."""
        # Get current max version
        async with self._get_lock():
            cursor = await self._db.execute(
                """
                SELECT MAX(version_number) as max_ver FROM versions
                WHERE collection = ? AND record_id = ?
                """,
                (collection, record_id)
            )
            row = await cursor.fetchone()
            current_version = row["max_ver"] if row and row["max_ver"] else 0

            new_version = current_version + 1
            now = datetime.now().isoformat()

            await self._db.execute(
                """
                INSERT INTO versions (record_id, collection, version_number, data, versioned_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (record_id, collection, new_version, json.dumps(version_data, ensure_ascii=False), now)
            )
            await self._db.commit()

        return new_version

    async def get_versions(
        self,
        collection: str,
        record_id: str
    ) -> list[dict[str, Any]]:
        """Get all versions of a record."""
        async with self._get_lock():
            cursor = await self._db.execute(
                """
                SELECT version_number, data, versioned_at FROM versions
                WHERE collection = ? AND record_id = ?
                ORDER BY version_number ASC
                """,
                (collection, record_id)
            )
            rows = await cursor.fetchall()

        versions = []
        for row in rows:
            version_data = json.loads(row["data"])
            version_data["version_number"] = row["version_number"]
            version_data["versioned_at"] = row["versioned_at"]
            versions.append(version_data)

        return versions