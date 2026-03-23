"""
Fulltext Store Implementation.

Whoosh-based fulltext search with BM25 ranking.
"""

import asyncio
import shutil
from pathlib import Path
from typing import Any, Optional

from rag_rag.storage.base import RetrievableStore, StoreStatus
from rag_rag.core.exceptions import FulltextStoreError
from rag_rag.core.logging import get_logger

logger = get_logger("rag_rag.storage.fulltext")


class FulltextStore(RetrievableStore):
    """
    Fulltext Store with Whoosh backend.

    Features:
    - BM25 ranking
    - Chinese text support
    - Field-based search
    """

    def __init__(
        self,
        index_dir: str | Path,
        index_name: str = "documents",
        analyzer: Optional[str] = None,
    ):
        super().__init__("fulltext_store")
        self.index_dir = Path(index_dir)
        self.index_name = index_name
        self.analyzer_name = analyzer or "chinese"
        self._index: Optional[Any] = None
        self._writer: Optional[Any] = None

    async def initialize(self) -> None:
        """Initialize Whoosh index."""
        try:
            from whoosh.index import create_in, exists_in, open_dir
            from whoosh.fields import Schema, TEXT, ID, STORED
            from whoosh.analysis import StemmingAnalyzer, SimpleAnalyzer

            # Ensure directory exists
            self.index_dir.mkdir(parents=True, exist_ok=True)

            # Define schema
            # Use SimpleAnalyzer for Chinese support (no stemming)
            schema = Schema(
                document_id=ID(stored=True, unique=True),
                content=TEXT(stored=True, analyzer=SimpleAnalyzer()),
                title=TEXT(stored=True),
                category=ID(stored=True),
                metadata=STORED,
            )

            # Create or open index
            if exists_in(self.index_dir, self.index_name):
                self._index = open_dir(self.index_dir, self.index_name)
            else:
                self._index = create_in(self.index_dir, schema, self.index_name)

            self._set_ready()
            logger.info(
                f"Fulltext Store initialized: {self.index_dir} "
                f"({self._index.doc_count()} documents)"
            )

        except ImportError:
            self._set_error("whoosh not installed")
            raise FulltextStoreError(
                "whoosh not installed. Install with: pip install whoosh"
            )
        except Exception as e:
            self._set_error(str(e))
            raise FulltextStoreError(f"Failed to initialize fulltext store: {e}")

    async def health_check(self) -> bool:
        """Check if fulltext store is healthy."""
        try:
            if self._index is None:
                return False
            self._index.doc_count()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the fulltext store."""
        if self._writer:
            self._writer.commit()
            self._writer = None
        self._index = None
        self._set_closed()

    # === Document Operations ===

    async def add(
        self,
        documents: list[dict[str, Any]],
        **kwargs: Any,
    ) -> int:
        """
        Add documents to the fulltext index.

        Args:
            documents: List of documents with 'id', 'content', 'metadata'

        Returns:
            Number of documents added
        """
        if not documents:
            return 0

        if self._index is None:
            raise FulltextStoreError("Store not initialized")

        import json

        writer = self._index.writer()

        for doc in documents:
            doc_id = doc.get("id", "")
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            title = metadata.get("title", "")
            category = metadata.get("category", "")

            writer.add_document(
                document_id=doc_id,
                content=content,
                title=title,
                category=category,
                metadata=json.dumps(metadata),
            )

        writer.commit()
        logger.debug(f"Added {len(documents)} documents to fulltext index")
        return len(documents)

    async def search(
        self,
        query: str,
        top_k: int = 10,
        fields: Optional[list[str]] = None,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Search for documents using BM25.

        Args:
            query: Search query
            top_k: Maximum results
            fields: Fields to search (default: content)
            filter: Optional filter

        Returns:
            List of search results
        """
        if self._index is None:
            raise FulltextStoreError("Store not initialized")

        from whoosh.qparser import QueryParser, MultifieldParser
        import json

        # Create searcher
        searcher = self._index.searcher()

        try:
            # Parse query
            search_fields = fields or ["content", "title"]
            if len(search_fields) == 1:
                parser = QueryParser(search_fields[0], self._index.schema)
            else:
                parser = MultifieldParser(search_fields, self._index.schema)

            parsed_query = parser.parse(query)

            # Execute search
            results = searcher.search(parsed_query, limit=top_k)

            # Format results
            formatted_results = []
            for hit in results:
                formatted_results.append({
                    "document_id": hit["document_id"],
                    "content": hit.get("content", ""),
                    "score": hit.score,
                    "source": "fulltext",
                    "metadata": json.loads(hit.get("metadata", "{}")) if hit.get("metadata") else {},
                })

            return formatted_results

        finally:
            searcher.close()

    async def delete(self, document_ids: list[str]) -> int:
        """Delete documents by IDs."""
        if self._index is None:
            raise FulltextStoreError("Store not initialized")

        from whoosh.query import Term

        writer = self._index.writer()
        count = 0

        for doc_id in document_ids:
            writer.delete_by_term("document_id", doc_id)
            count += 1

        writer.commit()
        return count

    async def get(self, document_id: str) -> Optional[dict[str, Any]]:
        """Get a document by ID."""
        if self._index is None:
            raise FulltextStoreError("Store not initialized")

        from whoosh.query import Term
        import json

        searcher = self._index.searcher()
        try:
            results = searcher.search(Term("document_id", document_id), limit=1)

            if results:
                hit = results[0]
                return {
                    "document_id": hit["document_id"],
                    "content": hit.get("content", ""),
                    "metadata": json.loads(hit.get("metadata", "{}")) if hit.get("metadata") else {},
                }
            return None
        finally:
            searcher.close()

    async def count(self) -> int:
        """Get total document count."""
        if self._index is None:
            return 0
        return self._index.doc_count()

    async def clear(self) -> int:
        """Clear all documents."""
        if self._index is None:
            return 0

        count = self._index.doc_count()

        # Delete all documents
        writer = self._index.writer()
        writer.commit(mergetype="clear")

        return count

    async def update(
        self,
        document_id: str,
        document: dict[str, Any],
    ) -> bool:
        """Update a document."""
        if self._index is None:
            raise FulltextStoreError("Store not initialized")

        import json

        writer = self._index.writer()

        # Delete old version
        writer.delete_by_term("document_id", document_id)

        # Add new version
        metadata = document.get("metadata", {})
        writer.add_document(
            document_id=document_id,
            content=document.get("content", ""),
            title=metadata.get("title", ""),
            category=metadata.get("category", ""),
            metadata=json.dumps(metadata),
        )

        writer.commit()
        return True

    def get_info(self) -> dict[str, Any]:
        """Get store information."""
        return {
            "name": self.name,
            "status": self.status.value,
            "index_dir": str(self.index_dir),
            "index_name": self.index_name,
            "document_count": self._index.doc_count() if self._index else 0,
        }