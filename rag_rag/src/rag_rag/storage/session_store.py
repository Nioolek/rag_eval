"""
Session Store Implementation.

SQLite-based storage for conversation history and session state.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from rag_rag.storage.base import KeyValueStore, StoreStatus
from rag_rag.core.exceptions import SessionStoreError
from rag_rag.core.logging import get_logger

logger = get_logger("rag_rag.storage.session")


class SessionStore(KeyValueStore):
    """
    Session Store with SQLite backend.

    Features:
    - Conversation history persistence
    - TTL-based expiration
    - Agent-specific sessions
    """

    def __init__(
        self,
        db_path: str | Path,
        max_history_turns: int = 5,
        session_timeout: int = 3600,
    ):
        super().__init__("session_store")
        self.db_path = Path(db_path)
        self.max_history_turns = max_history_turns
        self.session_timeout = session_timeout
        self._db: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the session database."""
        try:
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Connect to database
            self._db = await aiosqlite.connect(self.db_path)
            self._db.row_factory = aiosqlite.Row

            # Create tables
            await self._create_tables()

            self._set_ready()
            logger.info(f"Session Store initialized: {self.db_path}")

        except Exception as e:
            self._set_error(str(e))
            raise SessionStoreError(f"Failed to initialize session store: {e}")

    async def _create_tables(self) -> None:
        """Create database tables."""
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
        """)
        await self._db.commit()

    async def health_check(self) -> bool:
        """Check if session store is healthy."""
        try:
            if self._db is None:
                return False
            await self._db.execute("SELECT 1")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
        self._set_closed()

    # === Session Operations ===

    async def create_session(
        self,
        session_id: str,
        agent_id: str = "default",
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Create a new session."""
        async with self._lock:
            now = datetime.utcnow()
            expires_at = now + timedelta(seconds=self.session_timeout)

            await self._db.execute(
                """
                INSERT OR REPLACE INTO sessions (id, agent_id, metadata, created_at, updated_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    agent_id,
                    json.dumps(metadata or {}),
                    now.isoformat(),
                    now.isoformat(),
                    expires_at.isoformat(),
                ),
            )
            await self._db.commit()

            return {
                "id": session_id,
                "agent_id": agent_id,
                "metadata": metadata or {},
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "expires_at": expires_at.isoformat(),
            }

    async def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Get session info."""
        cursor = await self._db.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if row:
            return {
                "id": row["id"],
                "agent_id": row["agent_id"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "expires_at": row["expires_at"],
            }
        return None

    async def update_session(
        self,
        session_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Update session metadata and expiration."""
        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=self.session_timeout)

        async with self._lock:
            cursor = await self._db.execute(
                """
                UPDATE sessions SET
                    metadata = COALESCE(?, metadata),
                    updated_at = ?,
                    expires_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(metadata) if metadata else None,
                    now.isoformat(),
                    expires_at.isoformat(),
                    session_id,
                ),
            )
            await self._db.commit()
            return cursor.rowcount > 0

    # === Message Operations ===

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Add a message to the conversation history."""
        now = datetime.utcnow()

        async with self._lock:
            # Ensure session exists
            session = await self.get_session(session_id)
            if not session:
                await self.create_session(session_id)

            # Add message
            cursor = await self._db.execute(
                """
                INSERT INTO messages (session_id, role, content, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    now.isoformat(),
                    json.dumps(metadata or {}),
                ),
            )
            await self._db.commit()

            # Update session
            await self.update_session(session_id)

            return {
                "id": cursor.lastrowid,
                "session_id": session_id,
                "role": role,
                "content": content,
                "timestamp": now.isoformat(),
                "metadata": metadata or {},
            }

    async def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Get conversation history for a session."""
        limit = limit or self.max_history_turns * 2  # Each turn = 2 messages

        cursor = await self._db.execute(
            """
            SELECT * FROM messages
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (session_id, limit),
        )
        rows = await cursor.fetchall()

        messages = []
        for row in reversed(rows):  # Return in chronological order
            messages.append({
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
                "metadata": json.loads(row["metadata"]),
            })

        return messages

    async def clear_history(self, session_id: str) -> int:
        """Clear conversation history for a session."""
        async with self._lock:
            cursor = await self._db.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            await self._db.commit()
            return cursor.rowcount

    # === KeyValueStore Interface ===

    async def get(self, key: str) -> Optional[Any]:
        """Get session data (key = session_id)."""
        session = await self.get_session(key)
        if session:
            return {
                "session": session,
                "history": await self.get_history(key),
            }
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set session data."""
        if isinstance(value, dict):
            agent_id = value.get("agent_id", "default")
            metadata = value.get("metadata", {})
            await self.create_session(key, agent_id, metadata)

            if "messages" in value:
                for msg in value["messages"]:
                    await self.add_message(
                        key,
                        msg.get("role", "user"),
                        msg.get("content", ""),
                        msg.get("metadata"),
                    )

    async def delete(self, key: str) -> bool:
        """Delete a session and its history."""
        async with self._lock:
            cursor = await self._db.execute(
                "DELETE FROM sessions WHERE id = ?", (key,)
            )
            await self._db.commit()
            return cursor.rowcount > 0

    async def exists(self, key: str) -> bool:
        """Check if session exists."""
        cursor = await self._db.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (key,)
        )
        return await cursor.fetchone() is not None

    async def keys(self, pattern: Optional[str] = None) -> list[str]:
        """List session IDs."""
        if pattern:
            cursor = await self._db.execute(
                "SELECT id FROM sessions WHERE id LIKE ?",
                (pattern.replace("*", "%"),),
            )
        else:
            cursor = await self._db.execute("SELECT id FROM sessions")
        rows = await cursor.fetchall()
        return [row["id"] for row in rows]

    # === Maintenance ===

    async def cleanup_expired(self) -> int:
        """Remove expired sessions."""
        now = datetime.utcnow().isoformat()

        async with self._lock:
            cursor = await self._db.execute(
                "DELETE FROM sessions WHERE expires_at < ?", (now,)
            )
            await self._db.commit()
            count = cursor.rowcount

            if count > 0:
                logger.info(f"Cleaned up {count} expired sessions")

            return count

    async def get_stats(self) -> dict[str, Any]:
        """Get session statistics."""
        cursor = await self._db.execute("SELECT COUNT(*) as count FROM sessions")
        session_count = (await cursor.fetchone())["count"]

        cursor = await self._db.execute("SELECT COUNT(*) as count FROM messages")
        message_count = (await cursor.fetchone())["count"]

        return {
            "total_sessions": session_count,
            "total_messages": message_count,
        }