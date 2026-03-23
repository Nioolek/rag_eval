"""
Vector Store Implementation.

Chroma-based vector storage for semantic search.
"""

import asyncio
from pathlib import Path
from typing import Any, Optional

from rag_rag.storage.base import RetrievableStore, StoreStatus
from rag_rag.core.exceptions import VectorStoreError
from rag_rag.core.logging import get_logger

logger = get_logger("rag_rag.storage.vector")


class VectorStore(RetrievableStore):
    """
    Vector Store with Chroma backend.

    Features:
    - Semantic similarity search
    - Metadata filtering
    - Batch operations
    """

    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str = "documents",
        embedding_dimension: int = 1024,
        embedding_service: Optional[Any] = None,
    ):
        super().__init__("vector_store")
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self._embedding_service = embedding_service
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None

    async def initialize(self) -> None:
        """Initialize Chroma client and collection."""
        try:
            import chromadb

            # Ensure directory exists
            self.persist_dir.mkdir(parents=True, exist_ok=True)

            # Create persistent client
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            self._set_ready()
            logger.info(
                f"Vector Store initialized: {self.persist_dir} "
                f"({self._collection.count()} documents)"
            )

        except ImportError:
            self._set_error("chromadb not installed")
            raise VectorStoreError(
                "chromadb not installed. Install with: pip install chromadb"
            )
        except Exception as e:
            self._set_error(str(e))
            raise VectorStoreError(f"Failed to initialize vector store: {e}")

    async def health_check(self) -> bool:
        """Check if vector store is healthy."""
        try:
            if self._client is None or self._collection is None:
                return False
            # Try to count documents
            self._collection.count()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the vector store."""
        # Chroma client doesn't need explicit close
        self._client = None
        self._collection = None
        self._set_closed()

    # === Document Operations ===

    async def add(
        self,
        documents: list[dict[str, Any]],
        embeddings: Optional[list[list[float]]] = None,
        **kwargs: Any,
    ) -> int:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents with 'id', 'content', 'metadata'
            embeddings: Optional pre-computed embeddings

        Returns:
            Number of documents added
        """
        if not documents:
            return 0

        if self._collection is None:
            raise VectorStoreError("Store not initialized")

        # Generate embeddings if not provided
        if embeddings is None:
            if self._embedding_service is None:
                raise VectorStoreError(
                    "No embedding service and no embeddings provided"
                )
            contents = [doc.get("content", "") for doc in documents]
            embeddings = await self._embedding_service.embed(contents)

        # Prepare data
        ids = [doc.get("id", str(i)) for i, doc in enumerate(documents)]
        contents = [doc.get("content", "") for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        # Add to collection
        self._collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.debug(f"Added {len(ids)} documents to vector store")
        return len(ids)

    async def search(
        self,
        query: str | list[float],
        top_k: int = 10,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query: Query string or embedding vector
            top_k: Maximum results
            filter: Optional metadata filter

        Returns:
            List of search results
        """
        if self._collection is None:
            raise VectorStoreError("Store not initialized")

        # Get query embedding if string provided
        if isinstance(query, str):
            if self._embedding_service is None:
                raise VectorStoreError("No embedding service for text query")
            query_embedding = await self._embedding_service.embed_single(query)
        else:
            query_embedding = query

        # Search
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                formatted_results.append({
                    "document_id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "score": 1 - results["distances"][0][i] if results["distances"] else 0.0,
                    "source": "vector",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })

        return formatted_results

    async def delete(self, document_ids: list[str]) -> int:
        """Delete documents by IDs."""
        if self._collection is None:
            raise VectorStoreError("Store not initialized")

        try:
            self._collection.delete(ids=document_ids)
            return len(document_ids)
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return 0

    async def get(self, document_id: str) -> Optional[dict[str, Any]]:
        """Get a document by ID."""
        if self._collection is None:
            raise VectorStoreError("Store not initialized")

        try:
            results = self._collection.get(
                ids=[document_id],
                include=["documents", "metadatas", "embeddings"],
            )

            if results["ids"]:
                return {
                    "document_id": results["ids"][0],
                    "content": results["documents"][0] if results["documents"] else "",
                    "metadata": results["metadatas"][0] if results["metadatas"] else {},
                    "embedding": results["embeddings"][0] if results["embeddings"] else None,
                }
            return None
        except Exception:
            return None

    async def count(self) -> int:
        """Get total document count."""
        if self._collection is None:
            return 0
        return self._collection.count()

    async def clear(self) -> int:
        """Clear all documents."""
        if self._collection is None:
            return 0

        count = self._collection.count()

        # Delete all documents
        all_ids = self._collection.get()["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)

        return count

    async def update(
        self,
        document_id: str,
        document: dict[str, Any],
        embedding: Optional[list[float]] = None,
    ) -> bool:
        """Update a document."""
        if self._collection is None:
            raise VectorStoreError("Store not initialized")

        # Generate embedding if needed
        if embedding is None and self._embedding_service:
            embedding = await self._embedding_service.embed_single(
                document.get("content", "")
            )

        # Update in collection
        self._collection.update(
            ids=[document_id],
            documents=[document.get("content", "")],
            embeddings=[embedding] if embedding else None,
            metadatas=[document.get("metadata", {})],
        )

        return True

    def get_info(self) -> dict[str, Any]:
        """Get store information."""
        return {
            "name": self.name,
            "status": self.status.value,
            "persist_dir": str(self.persist_dir),
            "collection_name": self.collection_name,
            "document_count": self._collection.count() if self._collection else 0,
            "embedding_dimension": self.embedding_dimension,
        }