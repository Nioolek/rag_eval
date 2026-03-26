"""
MySQL storage backend with async support and connection pooling.
Implements database storage with dedicated tables and FULLTEXT search.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, AsyncIterator, Optional
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import aiomysql

from .base import StorageBackend
from ..core.exceptions import StorageError
from ..core.logging import logger


class MySQLStorage(StorageBackend):
    """
    MySQL storage backend implementing StorageBackend interface.

    Features:
    - Async operations with aiomysql
    - Connection pooling for better performance
    - Dedicated tables for main data types
    - JSON field support for flexible schema
    - FULLTEXT search support
    """

    def __init__(
        self,
        database_url: str,
        pool_size: int = 10,
        max_overflow: int = 5,
        pool_recycle: int = 3600,
        echo: bool = False,
    ):
        """
        Initialize MySQL storage.

        Args:
            database_url: MySQL connection URL in format:
                mysql+aiomysql://user:password@host:port/database?charset=utf8mb4
                or mysql://user:password@host:port/database
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
            pool_recycle: Recycle connections after this many seconds
            echo: Echo SQL statements for debugging
        """
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_recycle = pool_recycle
        self.echo = echo

        # Parse connection URL
        self._parse_url()

        self._pool: Optional[aiomysql.Pool] = None
        self._lock: Optional[asyncio.Lock] = None  # Lazy-initialized lock

    def _parse_url(self) -> None:
        """Parse MySQL connection URL into connection parameters."""
        url = self.database_url

        # Handle mysql+aiomysql:// prefix
        if url.startswith("mysql+aiomysql://"):
            url = url[17:]  # Remove prefix
        elif url.startswith("mysql://"):
            url = url[8:]  # Remove prefix

        # Parse the URL
        parsed = urlparse(f"mysql://{url}")

        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or 3306
        self.user = parsed.username or "root"
        self.password = parsed.password or ""
        self.database = parsed.path.lstrip("/") if parsed.path else "rag_eval"

        # Parse query parameters
        self.charset = "utf8mb4"
        if parsed.query:
            params = parse_qs(parsed.query)
            if "charset" in params:
                self.charset = params["charset"][0]

    def _get_lock(self) -> asyncio.Lock:
        """Get or create the lock (lazy initialization)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def initialize(self) -> None:
        """Create connection pool and tables."""
        try:
            # Create connection pool
            self._pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.database,
                charset=self.charset,
                minsize=1,
                maxsize=self.pool_size,
                pool_recycle=self.pool_recycle,
                echo=self.echo,
                autocommit=True,
            )

            # Create tables
            await self._create_tables()

            logger.info(
                f"Initialized MySQL storage: {self.host}:{self.port}/{self.database}"
            )

        except Exception as e:
            self._pool = None
            raise StorageError(f"Failed to initialize MySQL database: {e}")

    async def _create_tables(self) -> None:
        """Create necessary tables with indexes."""
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Annotations table
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS annotations (
                        id VARCHAR(36) PRIMARY KEY,
                        query TEXT NOT NULL,
                        conversation_history JSON,
                        agent_id VARCHAR(100) DEFAULT 'default',
                        language VARCHAR(10) DEFAULT 'auto',
                        enable_thinking BOOLEAN DEFAULT FALSE,
                        gt_documents JSON,
                        faq_matched BOOLEAN DEFAULT FALSE,
                        should_refuse BOOLEAN DEFAULT FALSE,
                        standard_answers JSON,
                        answer_style VARCHAR(500),
                        notes TEXT,
                        custom_fields JSON,
                        created_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
                        updated_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
                        version INT DEFAULT 1,
                        is_deleted BOOLEAN DEFAULT FALSE,

                        INDEX idx_agent_id (agent_id),
                        INDEX idx_created_at (created_at),
                        INDEX idx_is_deleted (is_deleted)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)

                # Try to add FULLTEXT index (may fail if not supported)
                try:
                    await cursor.execute("""
                        CREATE FULLTEXT INDEX ft_annotations_query
                        ON annotations(query)
                    """)
                except Exception:
                    pass  # FULLTEXT may not be available or index exists

                # Evaluation results table
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS evaluation_results (
                        id VARCHAR(36) PRIMARY KEY,
                        annotation_id VARCHAR(36),
                        run_id VARCHAR(36) NOT NULL,
                        rag_interface VARCHAR(100) DEFAULT 'default',
                        annotation JSON,
                        rag_response JSON,
                        metrics JSON,
                        tags JSON,
                        success BOOLEAN DEFAULT TRUE,
                        error_message TEXT,
                        duration_ms DOUBLE DEFAULT 0.0,
                        evaluated_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),

                        INDEX idx_run_id (run_id),
                        INDEX idx_rag_interface (rag_interface),
                        INDEX idx_evaluated_at (evaluated_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)

                # Evaluation runs table
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS evaluation_runs (
                        id VARCHAR(36) PRIMARY KEY,
                        name VARCHAR(255),
                        rag_interfaces JSON,
                        selected_metrics JSON,
                        concurrent_workers INT DEFAULT 10,
                        tags JSON,
                        total_annotations INT DEFAULT 0,
                        completed_count INT DEFAULT 0,
                        failed_count INT DEFAULT 0,
                        started_at DATETIME(6),
                        finished_at DATETIME(6),
                        duration_seconds DOUBLE DEFAULT 0.0,
                        status VARCHAR(50) DEFAULT 'pending',
                        summary_by_interface JSON,

                        INDEX idx_status (status),
                        INDEX idx_started_at (started_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)

                # Generic records table for other collections
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS records (
                        id VARCHAR(36) PRIMARY KEY,
                        collection VARCHAR(100) NOT NULL,
                        data JSON NOT NULL,
                        created_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
                        updated_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
                        is_deleted BOOLEAN DEFAULT FALSE,

                        INDEX idx_collection (collection),
                        INDEX idx_collection_deleted (collection, is_deleted)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)

                # Version history table
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS record_versions (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        record_id VARCHAR(36) NOT NULL,
                        collection VARCHAR(100) NOT NULL,
                        version_number INT NOT NULL,
                        data JSON NOT NULL,
                        versioned_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),

                        UNIQUE INDEX idx_collection_record_version (collection, record_id, version_number)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None
            logger.info("Closed MySQL connection pool")

    def _get_table_name(self, collection: str) -> str:
        """Get the appropriate table name for a collection."""
        # Map known collections to dedicated tables
        table_map = {
            "annotations": "annotations",
            "evaluation_results": "evaluation_results",
            "evaluation_runs": "evaluation_runs",
        }
        return table_map.get(collection, "records")

    def _is_dedicated_table(self, collection: str) -> bool:
        """Check if collection uses a dedicated table."""
        return collection in ("annotations", "evaluation_results", "evaluation_runs")

    async def save(self, collection: str, data: dict[str, Any]) -> str:
        """Save data to storage."""
        if "id" not in data:
            data["id"] = str(uuid4())

        record_id = data["id"]
        table = self._get_table_name(collection)

        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                if self._is_dedicated_table(collection):
                    await self._save_to_dedicated_table(
                        cursor, table, collection, data
                    )
                else:
                    await self._save_to_generic_table(
                        cursor, table, collection, data
                    )

        logger.debug(f"Saved record {record_id} to {collection}")
        return record_id

    async def _save_to_dedicated_table(
        self,
        cursor: aiomysql.Cursor,
        table: str,
        collection: str,
        data: dict[str, Any],
    ) -> None:
        """Save to a dedicated table with specific columns."""
        if table == "annotations":
            await cursor.execute(
                """
                INSERT INTO annotations (
                    id, query, conversation_history, agent_id, language,
                    enable_thinking, gt_documents, faq_matched, should_refuse,
                    standard_answers, answer_style, notes, custom_fields, version
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    query = VALUES(query),
                    conversation_history = VALUES(conversation_history),
                    agent_id = VALUES(agent_id),
                    language = VALUES(language),
                    enable_thinking = VALUES(enable_thinking),
                    gt_documents = VALUES(gt_documents),
                    faq_matched = VALUES(faq_matched),
                    should_refuse = VALUES(should_refuse),
                    standard_answers = VALUES(standard_answers),
                    answer_style = VALUES(answer_style),
                    notes = VALUES(notes),
                    custom_fields = VALUES(custom_fields),
                    version = version + 1
                """,
                (
                    data.get("id"),
                    data.get("query", ""),
                    self._to_json(data.get("conversation_history")),
                    data.get("agent_id", "default"),
                    data.get("language", "auto"),
                    data.get("enable_thinking", False),
                    self._to_json(data.get("gt_documents")),
                    data.get("faq_matched", False),
                    data.get("should_refuse", False),
                    self._to_json(data.get("standard_answers")),
                    data.get("answer_style"),
                    data.get("notes"),
                    self._to_json(data.get("custom_fields")),
                    data.get("version", 1),
                ),
            )

        elif table == "evaluation_results":
            await cursor.execute(
                """
                INSERT INTO evaluation_results (
                    id, annotation_id, run_id, rag_interface, annotation,
                    rag_response, metrics, tags, success, error_message, duration_ms
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    annotation_id = VALUES(annotation_id),
                    run_id = VALUES(run_id),
                    rag_interface = VALUES(rag_interface),
                    annotation = VALUES(annotation),
                    rag_response = VALUES(rag_response),
                    metrics = VALUES(metrics),
                    tags = VALUES(tags),
                    success = VALUES(success),
                    error_message = VALUES(error_message),
                    duration_ms = VALUES(duration_ms)
                """,
                (
                    data.get("id"),
                    data.get("annotation_id"),
                    data.get("run_id"),
                    data.get("rag_interface", "default"),
                    self._to_json(data.get("annotation")),
                    self._to_json(data.get("rag_response")),
                    self._to_json(data.get("metrics")),
                    self._to_json(data.get("tags")),
                    data.get("success", True),
                    data.get("error_message"),
                    data.get("duration_ms", 0.0),
                ),
            )

        elif table == "evaluation_runs":
            await cursor.execute(
                """
                INSERT INTO evaluation_runs (
                    id, name, rag_interfaces, selected_metrics, concurrent_workers,
                    tags, total_annotations, completed_count, failed_count,
                    started_at, finished_at, duration_seconds, status, summary_by_interface
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    name = VALUES(name),
                    rag_interfaces = VALUES(rag_interfaces),
                    selected_metrics = VALUES(selected_metrics),
                    concurrent_workers = VALUES(concurrent_workers),
                    tags = VALUES(tags),
                    total_annotations = VALUES(total_annotations),
                    completed_count = VALUES(completed_count),
                    failed_count = VALUES(failed_count),
                    started_at = VALUES(started_at),
                    finished_at = VALUES(finished_at),
                    duration_seconds = VALUES(duration_seconds),
                    status = VALUES(status),
                    summary_by_interface = VALUES(summary_by_interface)
                """,
                (
                    data.get("id"),
                    data.get("name"),
                    self._to_json(data.get("rag_interfaces")),
                    self._to_json(data.get("selected_metrics")),
                    data.get("concurrent_workers", 10),
                    self._to_json(data.get("tags")),
                    data.get("total_annotations", 0),
                    data.get("completed_count", 0),
                    data.get("failed_count", 0),
                    data.get("started_at"),
                    data.get("finished_at"),
                    data.get("duration_seconds", 0.0),
                    data.get("status", "pending"),
                    self._to_json(data.get("summary_by_interface")),
                ),
            )

    async def _save_to_generic_table(
        self,
        cursor: aiomysql.Cursor,
        table: str,
        collection: str,
        data: dict[str, Any],
    ) -> None:
        """Save to the generic records table."""
        await cursor.execute(
            """
            INSERT INTO records (id, collection, data)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
                data = VALUES(data)
            """,
            (data.get("id"), collection, self._to_json(data)),
        )

    def _to_json(self, value: Any) -> Optional[str]:
        """Convert value to JSON string if not None."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, default=str)

    def _from_json(self, value: Optional[str]) -> Any:
        """Parse JSON string if not None."""
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    async def get(self, collection: str, record_id: str) -> Optional[dict[str, Any]]:
        """Get a record by ID."""
        table = self._get_table_name(collection)

        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                if self._is_dedicated_table(collection):
                    await cursor.execute(
                        f"SELECT * FROM {table} WHERE id = %s AND is_deleted = FALSE",
                        (record_id,),
                    )
                else:
                    await cursor.execute(
                        """
                        SELECT id, data, created_at, updated_at
                        FROM records
                        WHERE id = %s AND collection = %s AND is_deleted = FALSE
                        """,
                        (record_id, collection),
                    )

                row = await cursor.fetchone()

        if not row:
            return None

        if self._is_dedicated_table(collection):
            return self._row_to_dict(row, collection)
        else:
            data = self._from_json(row["data"])
            data["id"] = row["id"]
            data["created_at"] = row["created_at"]
            data["updated_at"] = row["updated_at"]
            return data

    def _row_to_dict(self, row: dict, collection: str) -> dict[str, Any]:
        """Convert a database row to a dictionary."""
        result = {}

        for key, value in row.items():
            if key in ("is_deleted",):
                continue  # Skip internal fields

            if key in ("created_at", "updated_at", "evaluated_at", "started_at", "finished_at", "versioned_at"):
                result[key] = value.isoformat() if value else None
            elif key in ("conversation_history", "gt_documents", "standard_answers",
                        "custom_fields", "annotation", "rag_response", "metrics",
                        "tags", "rag_interfaces", "selected_metrics", "summary_by_interface", "data"):
                result[key] = self._from_json(value)
            else:
                result[key] = value

        return result

    async def get_all(
        self,
        collection: str,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get all records with filtering."""
        table = self._get_table_name(collection)

        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                if self._is_dedicated_table(collection):
                    query = f"""
                        SELECT * FROM {table}
                        WHERE is_deleted = FALSE
                    """
                    params: list = []

                    # Add simple filters for dedicated tables
                    if filters:
                        for key, value in filters.items():
                            if key in ("agent_id", "run_id", "rag_interface", "status", "success"):
                                query += f" AND {key} = %s"
                                params.append(value)

                    query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
                    params.extend([limit, offset])

                    await cursor.execute(query, params)
                    rows = await cursor.fetchall()

                    results = []
                    for row in rows:
                        data = self._row_to_dict(row, collection)
                        # Apply additional JSON field filters
                        if filters:
                            match = all(
                                self._match_filter(data, k, v)
                                for k, v in filters.items()
                                if k not in ("agent_id", "run_id", "rag_interface", "status", "success")
                            )
                            if not match:
                                continue
                        results.append(data)

                    return results
                else:
                    query = """
                        SELECT id, data, created_at, updated_at
                        FROM records
                        WHERE collection = %s AND is_deleted = FALSE
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                    """
                    await cursor.execute(query, (collection, limit, offset))
                    rows = await cursor.fetchall()

                    results = []
                    for row in rows:
                        data = self._from_json(row["data"])
                        data["id"] = row["id"]
                        data["created_at"] = row["created_at"]
                        data["updated_at"] = row["updated_at"]

                        if filters:
                            match = all(
                                self._match_filter(data, k, v)
                                for k, v in filters.items()
                            )
                            if not match:
                                continue
                        results.append(data)

                    return results

    def _match_filter(self, data: dict, key: str, value: Any) -> bool:
        """Check if data matches a filter condition."""
        data_value = data.get(key)
        if isinstance(value, list):
            return data_value in value
        return data_value == value

    async def update(
        self,
        collection: str,
        record_id: str,
        data: dict[str, Any],
    ) -> bool:
        """Update a record."""
        existing = await self.get(collection, record_id)
        if not existing:
            return False

        existing.update(data)
        table = self._get_table_name(collection)

        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                if self._is_dedicated_table(collection):
                    # For dedicated tables, we need to update specific columns
                    await self._save_to_dedicated_table(
                        cursor, table, collection, existing
                    )
                else:
                    await cursor.execute(
                        """
                        UPDATE records SET data = %s
                        WHERE id = %s AND collection = %s
                        """,
                        (self._to_json(existing), record_id, collection),
                    )

        return True

    async def delete(self, collection: str, record_id: str) -> bool:
        """Soft delete a record."""
        table = self._get_table_name(collection)

        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                if self._is_dedicated_table(collection):
                    await cursor.execute(
                        f"UPDATE {table} SET is_deleted = TRUE WHERE id = %s",
                        (record_id,),
                    )
                else:
                    await cursor.execute(
                        """
                        UPDATE records SET is_deleted = TRUE
                        WHERE id = %s AND collection = %s
                        """,
                        (record_id, collection),
                    )

        return True

    async def count(
        self,
        collection: str,
        filters: Optional[dict[str, Any]] = None,
    ) -> int:
        """Count records in collection."""
        table = self._get_table_name(collection)

        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                if self._is_dedicated_table(collection):
                    query = f"SELECT COUNT(*) FROM {table} WHERE is_deleted = FALSE"
                    params: list = []

                    if filters:
                        for key, value in filters.items():
                            if key in ("agent_id", "run_id", "rag_interface", "status", "success"):
                                query += f" AND {key} = %s"
                                params.append(value)

                    await cursor.execute(query, params)
                    result = await cursor.fetchone()

                    if not filters or all(
                        k in ("agent_id", "run_id", "rag_interface", "status", "success")
                        for k in filters.keys()
                    ):
                        return result[0] if result else 0

                    # Need to count with JSON field filters
                    return len(await self.get_all(collection, filters, limit=100000))
                else:
                    await cursor.execute(
                        """
                        SELECT COUNT(*) FROM records
                        WHERE collection = %s AND is_deleted = FALSE
                        """,
                        (collection,),
                    )
                    result = await cursor.fetchone()

                    if not filters:
                        return result[0] if result else 0

                    return len(await self.get_all(collection, filters, limit=100000))

    async def iterate(
        self,
        collection: str,
        batch_size: int = 100,
        filters: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Iterate over records in batches."""
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
        version_data: dict[str, Any],
    ) -> int:
        """Save a version snapshot."""
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Get current max version
                await cursor.execute(
                    """
                    SELECT MAX(version_number) FROM record_versions
                    WHERE collection = %s AND record_id = %s
                    """,
                    (collection, record_id),
                )
                result = await cursor.fetchone()
                current_version = result[0] if result and result[0] else 0

                new_version = current_version + 1

                await cursor.execute(
                    """
                    INSERT INTO record_versions
                    (record_id, collection, version_number, data)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (record_id, collection, new_version, self._to_json(version_data)),
                )

        return new_version

    async def get_versions(
        self,
        collection: str,
        record_id: str,
    ) -> list[dict[str, Any]]:
        """Get all versions of a record."""
        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    """
                    SELECT version_number, data, versioned_at
                    FROM record_versions
                    WHERE collection = %s AND record_id = %s
                    ORDER BY version_number ASC
                    """,
                    (collection, record_id),
                )
                rows = await cursor.fetchall()

        versions = []
        for row in rows:
            version_data = self._from_json(row["data"])
            version_data["version_number"] = row["version_number"]
            version_data["versioned_at"] = (
                row["versioned_at"].isoformat() if row["versioned_at"] else None
            )
            versions.append(version_data)

        return versions

    async def query_with_sort(
        self,
        collection: str,
        filters: Optional[dict[str, Any]] = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get records with sorting at the storage level."""
        table = self._get_table_name(collection)
        order = "DESC" if sort_desc else "ASC"

        # Validate sort field to prevent SQL injection
        valid_sort_fields = {
            "created_at", "updated_at", "evaluated_at", "started_at",
            "finished_at", "id", "agent_id", "run_id", "rag_interface",
            "status", "success", "name", "version",
        }

        if sort_by not in valid_sort_fields:
            sort_by = "created_at"

        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                if self._is_dedicated_table(collection):
                    query = f"""
                        SELECT * FROM {table}
                        WHERE is_deleted = FALSE
                    """
                    params: list = []

                    if filters:
                        for key, value in filters.items():
                            if key in valid_sort_fields:
                                query += f" AND {key} = %s"
                                params.append(value)

                    query += f" ORDER BY {sort_by} {order} LIMIT %s OFFSET %s"
                    params.extend([limit, offset])

                    await cursor.execute(query, params)
                    rows = await cursor.fetchall()

                    results = []
                    for row in rows:
                        data = self._row_to_dict(row, collection)
                        if filters:
                            match = all(
                                self._match_filter(data, k, v)
                                for k, v in filters.items()
                                if k not in valid_sort_fields
                            )
                            if not match:
                                continue
                        results.append(data)

                    return results
                else:
                    query = f"""
                        SELECT id, data, created_at, updated_at
                        FROM records
                        WHERE collection = %s AND is_deleted = FALSE
                        ORDER BY {sort_by} {order}
                        LIMIT %s OFFSET %s
                    """
                    await cursor.execute(query, (collection, limit, offset))
                    rows = await cursor.fetchall()

                    results = []
                    for row in rows:
                        data = self._from_json(row["data"])
                        data["id"] = row["id"]
                        data["created_at"] = row["created_at"]
                        data["updated_at"] = row["updated_at"]

                        if filters:
                            match = all(
                                self._match_filter(data, k, v)
                                for k, v in filters.items()
                            )
                            if not match:
                                continue
                        results.append(data)

                    return results

    async def search(
        self,
        collection: str,
        search_query: str,
        search_fields: Optional[list[str]] = None,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Search records by text query.

        Uses FULLTEXT search for supported tables, falls back to LIKE for others.
        """
        table = self._get_table_name(collection)
        search_pattern = f"%{search_query}%"

        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                if self._is_dedicated_table(collection) and table == "annotations":
                    # Try FULLTEXT search first
                    try:
                        query = """
                            SELECT * FROM annotations
                            WHERE is_deleted = FALSE
                            AND MATCH(query) AGAINST(%s IN NATURAL LANGUAGE MODE)
                        """
                        await cursor.execute(query, (search_query,))
                        rows = await cursor.fetchall()

                        if rows:
                            results = []
                            for row in rows:
                                data = self._row_to_dict(row, collection)
                                if filters:
                                    match = all(
                                        self._match_filter(data, k, v)
                                        for k, v in filters.items()
                                    )
                                    if not match:
                                        continue
                                results.append(data)
                            return results[offset:offset + limit]
                    except Exception:
                        pass  # Fall back to LIKE search

                # LIKE-based search for all tables
                if self._is_dedicated_table(collection):
                    query = f"""
                        SELECT * FROM {table}
                        WHERE is_deleted = FALSE
                    """
                    params: list = []

                    if filters:
                        for key, value in filters.items():
                            if key in ("agent_id", "run_id", "rag_interface", "status"):
                                query += f" AND {key} = %s"
                                params.append(value)

                    await cursor.execute(query, params)
                    rows = await cursor.fetchall()

                    results = []
                    for row in rows:
                        data = self._row_to_dict(row, collection)

                        # Search in specified fields or all text fields
                        found = False
                        fields_to_search = search_fields or ["query", "notes", "name", "error_message"]
                        for field in fields_to_search:
                            field_value = data.get(field, "")
                            if isinstance(field_value, str) and search_query.lower() in field_value.lower():
                                found = True
                                break

                        if not found:
                            continue

                        if filters:
                            match = all(
                                self._match_filter(data, k, v)
                                for k, v in filters.items()
                                if k not in ("agent_id", "run_id", "rag_interface", "status")
                            )
                            if not match:
                                continue

                        results.append(data)

                    return results[offset:offset + limit]
                else:
                    query = """
                        SELECT id, data, created_at, updated_at
                        FROM records
                        WHERE collection = %s AND is_deleted = FALSE
                        AND data LIKE %s
                    """
                    await cursor.execute(query, (collection, search_pattern))
                    rows = await cursor.fetchall()

                    results = []
                    for row in rows:
                        data = self._from_json(row["data"])
                        data["id"] = row["id"]
                        data["created_at"] = row["created_at"]
                        data["updated_at"] = row["updated_at"]

                        if filters:
                            match = all(
                                self._match_filter(data, k, v)
                                for k, v in filters.items()
                            )
                            if not match:
                                continue

                        results.append(data)

                    return results[offset:offset + limit]