"""
Task queue implementation with SQLite backend.
Provides reliable task storage with row-level locking for concurrent workers.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from .models import EvaluationTask, TaskStatus
from ..core.logging import logger
from ..core.exceptions import TaskError


class TaskQueue:
    """
    SQLite-backed task queue with row-level locking.

    Features:
    - Async operations
    - Row-level locking for concurrent workers
    - Heartbeat-based stale task detection
    - Priority-based task ordering
    """

    def __init__(self, db_path: str = "./data/scheduler.db"):
        self.db_path = Path(db_path)
        self._db: Optional[aiosqlite.Connection] = None
        self._lock: Optional[asyncio.Lock] = None  # Lazy-initialized lock
        self._worker_id: str = f"worker_{id(self)}"

    def _get_lock(self) -> asyncio.Lock:
        """Get or create the lock (lazy initialization)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock
    
    async def initialize(self) -> None:
        """Initialize database connection and create tables."""
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        
        # Enable WAL mode for better concurrency
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute("PRAGMA cache_size=-64000")  # 64MB cache
        
        await self._create_tables()
        logger.info(f"Initialized task queue database: {self.db_path}")
    
    async def _create_tables(self) -> None:
        """Create evaluation_tasks table with indexes."""
        # Main tasks table
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_tasks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending', 'claimed', 'running', 'completed', 'failed', 'cancelled')),
                priority INTEGER DEFAULT 2 CHECK(priority >= 0 AND priority <= 3),
                
                -- Task configuration
                annotation_source TEXT NOT NULL DEFAULT 'all',
                annotation_filter TEXT,
                metrics_config TEXT NOT NULL DEFAULT '[]',
                rag_interface TEXT,
                
                -- Worker assignment
                claimed_by TEXT,
                claimed_at TEXT,
                heartbeat_at TEXT,
                
                -- Scheduling reference
                schedule_id TEXT,
                
                -- Execution results
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                
                -- Metadata
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                is_deleted INTEGER DEFAULT 0
            )
        """)
        
        # Critical indexes for performance
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_status_priority
            ON evaluation_tasks(status, priority)
            WHERE is_deleted = 0
        """)
        
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_claimed
            ON evaluation_tasks(claimed_at)
            WHERE claimed_by IS NOT NULL
        """)
        
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_schedule
            ON evaluation_tasks(schedule_id)
            WHERE schedule_id IS NOT NULL
        """)
        
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_created
            ON evaluation_tasks(created_at)
            WHERE is_deleted = 0
        """)
        
        # Scheduled tasks table
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                cron_expression TEXT NOT NULL,
                timezone TEXT DEFAULT 'Asia/Shanghai',
                enabled INTEGER DEFAULT 1,
                task_config TEXT NOT NULL DEFAULT '{}',
                last_run_at TEXT,
                last_run_status TEXT,
                last_run_duration_ms INTEGER,
                next_run_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_scheduled_enabled
            ON scheduled_tasks(enabled)
            WHERE enabled = 1
        """)
        
        # Performance benchmarks table
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS performance_benchmarks (
                id TEXT PRIMARY KEY,
                test_name TEXT NOT NULL,
                test_type TEXT NOT NULL CHECK(test_type IN ('benchmark', 'stress', 'load')),
                version TEXT NOT NULL,
                avg_response_time_ms REAL,
                p95_response_time_ms REAL,
                p99_response_time_ms REAL,
                throughput_qps REAL,
                error_rate REAL,
                memory_peak_mb REAL,
                cpu_peak_percent REAL,
                environment_config TEXT NOT NULL,
                hardware_info TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT
            )
        """)
        
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_benchmarks_version
            ON performance_benchmarks(version)
        """)
        
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_benchmarks_test
            ON performance_benchmarks(test_name, test_type)
        """)
        
        await self._db.commit()
        logger.debug("Created database tables and indexes")
    
    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None
    
    # ==================== Task Operations ====================
    
    async def add_task(self, task: EvaluationTask) -> str:
        """Add a new task to the queue."""
        async with self._get_lock():
            await self._db.execute(
                """
                INSERT INTO evaluation_tasks (
                    id, name, status, priority, annotation_source, annotation_filter,
                    metrics_config, rag_interface, schedule_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.id,
                    task.name,
                    task.status.value,
                    task.priority,
                    task.annotation_source,
                    json.dumps(task.annotation_filter) if task.annotation_filter else None,
                    json.dumps(task.metrics_config),
                    task.rag_interface,
                    task.schedule_id,
                    task.created_at.isoformat(),
                    task.updated_at.isoformat(),
                )
            )
            await self._db.commit()
        
        logger.info(f"Added task {task.id}: {task.name}")
        return task.id
    
    async def claim_task(self, task_id: str) -> Optional[EvaluationTask]:
        """
        Claim a task for execution (row-level locking).
        
        Returns None if task cannot be claimed (already claimed or not pending).
        """
        async with self._get_lock():
            # Try to claim the task atomically
            cursor = await self._db.execute(
                """
                UPDATE evaluation_tasks
                SET status = 'claimed',
                    claimed_by = ?,
                    claimed_at = ?,
                    heartbeat_at = ?,
                    updated_at = ?
                WHERE id = ? AND status = 'pending' AND is_deleted = 0
                """,
                (
                    self._worker_id,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    task_id,
                )
            )
            
            if cursor.rowcount == 0:
                return None  # Task was already claimed or not pending
            
            await self._db.commit()
        
        # Fetch and return the claimed task
        return await self.get_task(task_id)
    
    async def get_next_pending_task(self) -> Optional[EvaluationTask]:
        """
        Get the next pending task ordered by priority and creation time.
        Does not claim the task - use claim_task() for that.
        """
        async with self._get_lock():
            cursor = await self._db.execute(
                """
                SELECT * FROM evaluation_tasks
                WHERE status = 'pending' AND is_deleted = 0
                ORDER BY priority ASC, created_at ASC
                LIMIT 1
                """,
            )
            row = await cursor.fetchone()
        
        if row:
            return self._row_to_task(row)
        return None
    
    async def start_task(self, task_id: str) -> bool:
        """Mark a claimed task as running."""
        async with self._get_lock():
            cursor = await self._db.execute(
                """
                UPDATE evaluation_tasks
                SET status = 'running',
                    started_at = ?,
                    updated_at = ?
                WHERE id = ? AND status = 'claimed' AND claimed_by = ?
                """,
                (
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    task_id,
                    self._worker_id,
                )
            )
            await self._db.commit()
        
        return cursor.rowcount > 0
    
    async def complete_task(self, task_id: str) -> bool:
        """Mark a running task as completed."""
        async with self._get_lock():
            cursor = await self._db.execute(
                """
                UPDATE evaluation_tasks
                SET status = 'completed',
                    completed_at = ?,
                    updated_at = ?
                WHERE id = ? AND claimed_by = ?
                """,
                (
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    task_id,
                    self._worker_id,
                )
            )
            await self._db.commit()
        
        logger.info(f"Task {task_id} completed")
        return cursor.rowcount > 0
    
    async def fail_task(self, task_id: str, error: str) -> bool:
        """Mark a running task as failed."""
        async with self._get_lock():
            cursor = await self._db.execute(
                """
                UPDATE evaluation_tasks
                SET status = 'failed',
                    error_message = ?,
                    completed_at = ?,
                    retry_count = retry_count + 1,
                    updated_at = ?
                WHERE id = ? AND claimed_by = ?
                """,
                (
                    error,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    task_id,
                    self._worker_id,
                )
            )
            await self._db.commit()
        
        logger.warning(f"Task {task_id} failed: {error}")
        return cursor.rowcount > 0
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task (any status)."""
        async with self._get_lock():
            cursor = await self._db.execute(
                """
                UPDATE evaluation_tasks
                SET status = 'cancelled',
                    completed_at = ?,
                    updated_at = ?
                WHERE id = ? AND status NOT IN ('completed', 'cancelled')
                """,
                (
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    task_id,
                )
            )
            await self._db.commit()
        
        logger.info(f"Task {task_id} cancelled")
        return cursor.rowcount > 0
    
    async def heartbeat(self, task_id: str) -> bool:
        """Update heartbeat timestamp for a running task."""
        async with self._get_lock():
            cursor = await self._db.execute(
                """
                UPDATE evaluation_tasks
                SET heartbeat_at = ?, updated_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (datetime.now().isoformat(), datetime.now().isoformat(), task_id)
            )
            await self._db.commit()
        
        return cursor.rowcount > 0
    
    async def release_stale_tasks(self, timeout_seconds: int = 300) -> int:
        """
        Release tasks that haven't sent heartbeat within timeout.
        
        Returns number of tasks released.
        """
        cutoff = (datetime.now() - timedelta(seconds=timeout_seconds)).isoformat()
        
        async with self._get_lock():
            cursor = await self._db.execute(
                """
                UPDATE evaluation_tasks
                SET status = 'pending',
                    claimed_by = NULL,
                    claimed_at = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE status IN ('claimed', 'running')
                  AND heartbeat_at < ?
                """,
                (cutoff,)
            )
            released = cursor.rowcount
            await self._db.commit()
        
        if released > 0:
            logger.info(f"Released {released} stale tasks")
        
        return released
    
    async def get_task(self, task_id: str) -> Optional[EvaluationTask]:
        """Get a task by ID."""
        async with self._get_lock():
            cursor = await self._db.execute(
                """
                SELECT * FROM evaluation_tasks
                WHERE id = ? AND is_deleted = 0
                """,
                (task_id,)
            )
            row = await cursor.fetchone()
        
        if row:
            return self._row_to_task(row)
        return None
    
    async def get_tasks_by_status(
        self,
        status: TaskStatus,
        limit: int = 100,
        offset: int = 0,
    ) -> list[EvaluationTask]:
        """Get tasks by status."""
        async with self._get_lock():
            cursor = await self._db.execute(
                """
                SELECT * FROM evaluation_tasks
                WHERE status = ? AND is_deleted = 0
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (status.value, limit, offset)
            )
            rows = await cursor.fetchall()
        
        return [self._row_to_task(row) for row in rows]
    
    async def get_pending_task_count(self) -> int:
        """Get count of pending tasks."""
        async with self._get_lock():
            cursor = await self._db.execute(
                """
                SELECT COUNT(*) as cnt FROM evaluation_tasks
                WHERE status = 'pending' AND is_deleted = 0
                """
            )
            row = await cursor.fetchone()
        
        return row["cnt"] if row else 0
    
    async def delete_task(self, task_id: str) -> bool:
        """Soft delete a task."""
        async with self._get_lock():
            cursor = await self._db.execute(
                """
                UPDATE evaluation_tasks
                SET is_deleted = 1, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (task_id,)
            )
            await self._db.commit()
        
        return cursor.rowcount > 0
    
    # ==================== Scheduled Task Operations ====================
    
    async def add_scheduled_task(self, scheduled: "ScheduledTask") -> str:
        """Add a scheduled task configuration."""
        async with self._get_lock():
            await self._db.execute(
                """
                INSERT INTO scheduled_tasks (
                    id, name, cron_expression, timezone, enabled, task_config,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scheduled.id,
                    scheduled.name,
                    scheduled.cron_expression,
                    scheduled.timezone,
                    1 if scheduled.enabled else 0,
                    json.dumps(scheduled.task_config),
                    scheduled.created_at.isoformat(),
                    scheduled.updated_at.isoformat(),
                )
            )
            await self._db.commit()
        
        logger.info(f"Added scheduled task {scheduled.id}: {scheduled.name}")
        return scheduled.id
    
    async def get_scheduled_task(self, task_id: str) -> Optional["ScheduledTask"]:
        """Get a scheduled task by ID."""
        # Import here to avoid circular dependency
        from .models import ScheduledTask
        
        async with self._get_lock():
            cursor = await self._db.execute(
                """
                SELECT * FROM scheduled_tasks
                WHERE id = ?
                """,
                (task_id,)
            )
            row = await cursor.fetchone()
        
        if row:
            data = dict(row)
            data["enabled"] = bool(data["enabled"])
            data["task_config"] = json.loads(data["task_config"])
            return ScheduledTask(**data)
        return None
    
    async def get_enabled_scheduled_tasks(self) -> list["ScheduledTask"]:
        """Get all enabled scheduled tasks."""
        from .models import ScheduledTask
        
        async with self._get_lock():
            cursor = await self._db.execute(
                """
                SELECT * FROM scheduled_tasks
                WHERE enabled = 1
                ORDER BY name
                """
            )
            rows = await cursor.fetchall()
        
        tasks = []
        for row in rows:
            data = dict(row)
            data["enabled"] = True
            data["task_config"] = json.loads(data["task_config"])
            tasks.append(ScheduledTask(**data))
        
        return tasks
    
    async def update_scheduled_task_run(
        self,
        task_id: str,
        status: str,
        duration_ms: int,
        next_run_at: Optional[datetime] = None,
    ) -> None:
        """Update scheduled task after a run."""
        async with self._get_lock():
            await self._db.execute(
                """
                UPDATE scheduled_tasks
                SET last_run_at = CURRENT_TIMESTAMP,
                    last_run_status = ?,
                    last_run_duration_ms = ?,
                    next_run_at = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    status,
                    duration_ms,
                    next_run_at.isoformat() if next_run_at else None,
                    task_id,
                )
            )
            await self._db.commit()
    
    async def delete_scheduled_task(self, task_id: str) -> bool:
        """Delete a scheduled task."""
        async with self._get_lock():
            cursor = await self._db.execute(
                "DELETE FROM scheduled_tasks WHERE id = ?",
                (task_id,)
            )
            await self._db.commit()
        
        return cursor.rowcount > 0
    
    # ==================== Benchmark Operations ====================
    
    async def save_benchmark(self, benchmark: "PerformanceBenchmark") -> str:
        """Save a performance benchmark result."""
        async with self._get_lock():
            await self._db.execute(
                """
                INSERT INTO performance_benchmarks (
                    id, test_name, test_type, version,
                    avg_response_time_ms, p95_response_time_ms, p99_response_time_ms,
                    throughput_qps, error_rate, memory_peak_mb, cpu_peak_percent,
                    environment_config, hardware_info, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    benchmark.id,
                    benchmark.test_name,
                    benchmark.test_type,
                    benchmark.version,
                    benchmark.avg_response_time_ms,
                    benchmark.p95_response_time_ms,
                    benchmark.p99_response_time_ms,
                    benchmark.throughput_qps,
                    benchmark.error_rate,
                    benchmark.memory_peak_mb,
                    benchmark.cpu_peak_percent,
                    json.dumps(benchmark.environment_config),
                    json.dumps(benchmark.hardware_info),
                    benchmark.created_by,
                )
            )
            await self._db.commit()
        
        logger.info(f"Saved benchmark {benchmark.id}: {benchmark.test_name}")
        return benchmark.id
    
    async def get_benchmarks(
        self,
        test_name: Optional[str] = None,
        version: Optional[str] = None,
        limit: int = 100,
    ) -> list["PerformanceBenchmark"]:
        """Get benchmark results with optional filtering."""
        from .models import PerformanceBenchmark
        
        query = "SELECT * FROM performance_benchmarks WHERE 1=1"
        params = []
        
        if test_name:
            query += " AND test_name = ?"
            params.append(test_name)
        
        if version:
            query += " AND version = ?"
            params.append(version)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        async with self._get_lock():
            cursor = await self._db.execute(query, params)
            rows = await cursor.fetchall()
        
        benchmarks = []
        for row in rows:
            data = dict(row)
            data["environment_config"] = json.loads(data["environment_config"])
            data["hardware_info"] = json.loads(data["hardware_info"])
            benchmarks.append(PerformanceBenchmark(**data))
        
        return benchmarks
    
    # ==================== Helper Methods ====================
    
    def _row_to_task(self, row: aiosqlite.Row) -> EvaluationTask:
        """Convert database row to EvaluationTask."""
        data = dict(row)
        
        # Convert status
        data["status"] = TaskStatus(data["status"])
        
        # Convert JSON fields
        if data.get("annotation_filter"):
            data["annotation_filter"] = json.loads(data["annotation_filter"])
        
        if data.get("metrics_config"):
            data["metrics_config"] = json.loads(data["metrics_config"])
        
        # Convert datetime fields
        for field in ["claimed_at", "heartbeat_at", "started_at", "completed_at", "created_at", "updated_at"]:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        
        # Convert is_deleted
        data["is_deleted"] = bool(data["is_deleted"])
        
        return EvaluationTask(**data)
    
    async def __aenter__(self) -> "TaskQueue":
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
