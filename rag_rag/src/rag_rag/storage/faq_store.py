"""
FAQ Store Implementation.

SQLite-based storage for FAQ Q&A pairs with exact and semantic matching.
"""

import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from rag_rag.storage.base import SearchableKeyValueStore, StoreStatus
from rag_rag.core.exceptions import FAQStoreError
from rag_rag.core.logging import get_logger

logger = get_logger("rag_rag.storage.faq")


class FAQStore(SearchableKeyValueStore):
    """
    FAQ Store with SQLite backend.

    Features:
    - Exact text matching
    - Semantic matching via embeddings (requires embedding service)
    - Hybrid search combining exact and semantic
    """

    def __init__(
        self,
        db_path: str | Path,
        match_threshold: float = 0.85,
        embedding_service: Optional[Any] = None,
    ):
        super().__init__("faq_store")
        self.db_path = Path(db_path)
        self.match_threshold = match_threshold
        self._embedding_service = embedding_service
        self._db: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the FAQ database."""
        try:
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Connect to database
            self._db = await aiosqlite.connect(self.db_path)
            self._db.row_factory = aiosqlite.Row

            # Create tables
            await self._create_tables()

            self._set_ready()
            logger.info(f"FAQ Store initialized: {self.db_path}")

        except Exception as e:
            self._set_error(str(e))
            raise FAQStoreError(f"Failed to initialize FAQ store: {e}")

    async def _create_tables(self) -> None:
        """Create database tables."""
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS faqs (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                category TEXT DEFAULT '',
                keywords TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                embedding BLOB,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_faqs_question ON faqs(question);
            CREATE INDEX IF NOT EXISTS idx_faqs_category ON faqs(category);
        """)
        await self._db.commit()

    async def health_check(self) -> bool:
        """Check if FAQ store is healthy."""
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

    # === CRUD Operations ===

    async def get(self, key: str) -> Optional[dict[str, Any]]:
        """Get FAQ by ID."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT * FROM faqs WHERE id = ?", (key,)
            )
            row = await cursor.fetchone()
            if row:
                return self._row_to_dict(row)
            return None

    async def set(
        self,
        key: str,
        value: dict[str, Any],
        ttl: Optional[int] = None,
    ) -> None:
        """Add or update an FAQ."""
        async with self._lock:
            now = datetime.utcnow().isoformat()

            # Check if exists
            cursor = await self._db.execute(
                "SELECT id FROM faqs WHERE id = ?", (key,)
            )
            exists = await cursor.fetchone()

            import json

            if exists:
                # Update
                await self._db.execute(
                    """
                    UPDATE faqs SET
                        question = ?,
                        answer = ?,
                        category = ?,
                        keywords = ?,
                        metadata = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        value.get("question", ""),
                        value.get("answer", ""),
                        value.get("category", ""),
                        json.dumps(value.get("keywords", [])),
                        json.dumps(value.get("metadata", {})),
                        now,
                        key,
                    ),
                )
            else:
                # Insert
                await self._db.execute(
                    """
                    INSERT INTO faqs (id, question, answer, category, keywords, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key,
                        value.get("question", ""),
                        value.get("answer", ""),
                        value.get("category", ""),
                        json.dumps(value.get("keywords", [])),
                        json.dumps(value.get("metadata", {})),
                        now,
                        now,
                    ),
                )

            await self._db.commit()

    async def delete(self, key: str) -> bool:
        """Delete FAQ by ID."""
        async with self._lock:
            cursor = await self._db.execute(
                "DELETE FROM faqs WHERE id = ?", (key,)
            )
            await self._db.commit()
            return cursor.rowcount > 0

    async def exists(self, key: str) -> bool:
        """Check if FAQ exists."""
        cursor = await self._db.execute(
            "SELECT 1 FROM faqs WHERE id = ?", (key,)
        )
        return await cursor.fetchone() is not None

    async def keys(self, pattern: Optional[str] = None) -> list[str]:
        """List FAQ IDs."""
        if pattern:
            cursor = await self._db.execute(
                "SELECT id FROM faqs WHERE id LIKE ?", (pattern.replace("*", "%"),)
            )
        else:
            cursor = await self._db.execute("SELECT id FROM faqs")
        rows = await cursor.fetchall()
        return [row["id"] for row in rows]

    # === Search Operations ===

    async def search(
        self,
        query: str,
        top_k: int = 10,
        search_type: str = "hybrid",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Search for FAQs.

        Args:
            query: Search query
            top_k: Maximum results
            search_type: "exact", "semantic", or "hybrid"

        Returns:
            List of matching FAQs with scores
        """
        if search_type == "exact":
            return await self._exact_search(query, top_k)
        elif search_type == "semantic":
            return await self._semantic_search(query, top_k)
        else:  # hybrid
            return await self._hybrid_search(query, top_k)

    async def _exact_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Exact text matching search."""
        query_lower = query.lower().strip()

        cursor = await self._db.execute(
            """
            SELECT * FROM faqs
            WHERE LOWER(question) LIKE ? OR LOWER(keywords) LIKE ?
            LIMIT ?
            """,
            (f"%{query_lower}%", f"%{query_lower}%", top_k),
        )
        rows = await cursor.fetchall()

        results = []
        for row in rows:
            faq = self._row_to_dict(row)
            # Calculate simple similarity score
            question_lower = faq["question"].lower()
            if query_lower == question_lower:
                score = 1.0
            elif query_lower in question_lower or question_lower in query_lower:
                score = 0.8
            else:
                score = 0.5

            if score >= self.match_threshold:
                results.append({
                    **faq,
                    "score": score,
                    "match_type": "exact",
                })

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    async def _semantic_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Semantic search using embeddings."""
        if not self._embedding_service:
            logger.warning("No embedding service, falling back to exact search")
            return await self._exact_search(query, top_k)

        try:
            # Get query embedding
            query_embedding = await self._embedding_service.embed_single(query)

            # Get all FAQs with embeddings
            cursor = await self._db.execute(
                "SELECT * FROM faqs WHERE embedding IS NOT NULL"
            )
            rows = await cursor.fetchall()

            import numpy as np

            results = []
            for row in rows:
                faq = self._row_to_dict(row)
                if row["embedding"]:
                    stored_embedding = np.frombuffer(row["embedding"], dtype=np.float32)
                    score = self._cosine_similarity(query_embedding, stored_embedding)

                    if score >= self.match_threshold:
                        results.append({
                            **faq,
                            "score": float(score),
                            "match_type": "semantic",
                        })

            return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return await self._exact_search(query, top_k)

    async def _hybrid_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Hybrid search combining exact and semantic."""
        # Get results from both
        exact_results = await self._exact_search(query, top_k * 2)
        semantic_results = await self._semantic_search(query, top_k * 2)

        # Merge and deduplicate
        merged = {}
        for result in exact_results:
            merged[result["id"]] = {
                **result,
                "exact_score": result["score"],
                "semantic_score": 0.0,
            }

        for result in semantic_results:
            if result["id"] in merged:
                merged[result["id"]]["semantic_score"] = result["score"]
            else:
                merged[result["id"]] = {
                    **result,
                    "exact_score": 0.0,
                    "semantic_score": result["score"],
                }

        # Calculate combined score (weighted)
        for item in merged.values():
            item["score"] = 0.3 * item["exact_score"] + 0.7 * item["semantic_score"]

        # Sort and return top_k
        results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def add_faqs(
        self,
        faqs: list[dict[str, Any]],
        generate_embeddings: bool = True,
    ) -> int:
        """
        Add multiple FAQs.

        Args:
            faqs: List of FAQ dicts with 'id', 'question', 'answer'
            generate_embeddings: Whether to generate embeddings

        Returns:
            Number of FAQs added
        """
        count = 0
        for faq in faqs:
            await self.set(faq["id"], faq)
            count += 1

        if generate_embeddings and self._embedding_service:
            await self._generate_embeddings_batch(faqs)

        return count

    async def _generate_embeddings_batch(
        self,
        faqs: list[dict[str, Any]],
    ) -> None:
        """Generate and store embeddings for FAQs."""
        if not self._embedding_service:
            return

        import numpy as np

        questions = [faq["question"] for faq in faqs]
        embeddings = await self._embedding_service.embed(questions)

        async with self._lock:
            for faq, embedding in zip(faqs, embeddings):
                embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                await self._db.execute(
                    "UPDATE faqs SET embedding = ? WHERE id = ?",
                    (embedding_bytes, faq["id"]),
                )
            await self._db.commit()

    @staticmethod
    def _cosine_similarity(a: list[float], b: Any) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np

        a_np = np.array(a)
        b_np = np.array(b)
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))

    @staticmethod
    def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
        """Convert database row to dict."""
        import json

        return {
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "category": row["category"],
            "keywords": json.loads(row["keywords"]),
            "metadata": json.loads(row["metadata"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }