"""
Fallback Handlers Implementation.

Provides fallback strategies for different services.
"""

import asyncio
from typing import Any, Optional

from rag_rag.core.logging import get_logger

logger = get_logger("rag_rag.degradation.fallback")


class FallbackHandler:
    """Base class for fallback handlers."""

    def __init__(self, name: str):
        self.name = name

    async def handle(self, *args: Any, **kwargs: Any) -> Any:
        """Handle fallback."""
        raise NotImplementedError


class EmbeddingFallback(FallbackHandler):
    """
    Fallback handler for embedding service.

    Strategies:
    1. Return cached embedding if available
    2. Return pre-computed embedding
    3. Return zero vector (disables vector search)
    """

    def __init__(
        self,
        dimension: int = 1024,
        cache: Optional[dict[str, list[float]]] = None,
    ):
        super().__init__("embedding_fallback")
        self.dimension = dimension
        self._cache = cache or {}

    async def handle(self, text: str) -> list[float]:
        """Get embedding with fallback."""
        # Check cache
        if text in self._cache:
            logger.debug(f"Using cached embedding for: {text[:50]}...")
            return self._cache[text]

        # Return zero vector (disables vector search but doesn't break pipeline)
        logger.warning(f"Using zero vector fallback for: {text[:50]}...")
        return [0.0] * self.dimension

    async def handle_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for batch with fallback."""
        return [await self.handle(text) for text in texts]


class RerankFallback(FallbackHandler):
    """
    Fallback handler for rerank service.

    Strategies:
    1. Local BM25 reranking
    2. Keep original order
    """

    def __init__(self, top_k: int = 5):
        super().__init__("rerank_fallback")
        self.top_k = top_k

    async def handle(
        self,
        query: str,
        documents: list[str],
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Rerank with BM25 fallback."""
        top_k = top_k or self.top_k

        try:
            from rank_bm25 import BM25Okapi

            tokenized_docs = [doc.split() for doc in documents]
            bm25 = BM25Okapi(tokenized_docs)
            scores = bm25.get_scores(query.split())

            indexed_scores = sorted(
                enumerate(scores),
                key=lambda x: x[1],
                reverse=True,
            )

            return [
                {
                    "index": idx,
                    "document": documents[idx],
                    "relevance_score": float(score),
                }
                for idx, score in indexed_scores[:top_k]
            ]

        except ImportError:
            # Keep original order
            logger.warning("rank-bm25 not installed, keeping original order")
            return [
                {
                    "index": i,
                    "document": doc,
                    "relevance_score": 1.0 - i * 0.1,
                }
                for i, doc in enumerate(documents[:top_k])
            ]


class LLMFallback(FallbackHandler):
    """
    Fallback handler for LLM service.

    Strategies:
    1. Use backup model (qwen-turbo instead of qwen-plus)
    2. Use template response
    """

    def __init__(
        self,
        backup_model: str = "qwen-turbo",
        template_responses: Optional[dict[str, str]] = None,
    ):
        super().__init__("llm_fallback")
        self.backup_model = backup_model
        self._template_responses = template_responses or {
            "default": "抱歉，我暂时无法处理您的请求，请稍后再试。",
            "refusal": "抱歉，这个问题我无法回答。",
            "error": "服务暂时不可用，请稍后再试。",
        }

    async def handle(
        self,
        query: str,
        error_type: str = "default",
    ) -> str:
        """Get LLM response with fallback."""
        # Try backup model first
        # (This would be implemented with actual LLM call in production)
        logger.warning(f"Using template fallback for LLM: {error_type}")

        return self._template_responses.get(
            error_type,
            self._template_responses["default"],
        )


class VectorStoreFallback(FallbackHandler):
    """Fallback handler for vector store - returns empty results."""

    async def handle(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Return empty results."""
        logger.warning("Vector store unavailable, returning empty results")
        return []


class FulltextStoreFallback(FallbackHandler):
    """Fallback handler for fulltext store - returns empty results."""

    async def handle(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Return empty results."""
        logger.warning("Fulltext store unavailable, returning empty results")
        return []


class GraphStoreFallback(FallbackHandler):
    """Fallback handler for graph store - returns empty results."""

    async def handle(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Return empty results."""
        logger.warning("Graph store unavailable, returning empty results")
        return []


class QueryRewriteFallback(FallbackHandler):
    """
    Fallback handler for query rewrite.

    Uses simple rules instead of LLM.
    """

    async def handle(
        self,
        query: str,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> dict[str, Any]:
        """Simple rule-based query rewrite."""
        # Simple strategy: append context from history
        if conversation_history:
            last_user_msg = ""
            for msg in reversed(conversation_history):
                if msg.get("role") == "user":
                    last_user_msg = msg.get("content", "")
                    break

            if last_user_msg:
                # Combine with previous context
                rewritten = f"{query} (相关上下文: {last_user_msg[:50]})"
                return {
                    "original_query": query,
                    "rewritten_query": rewritten,
                    "rewrite_type": "simple_context",
                    "confidence": 0.5,
                }

        # No history - return original query
        return {
            "original_query": query,
            "rewritten_query": query,
            "rewrite_type": "none",
            "confidence": 0.0,
        }


# Fallback registry
FALLBACK_HANDLERS: dict[str, FallbackHandler] = {}


def register_fallback(name: str, handler: FallbackHandler) -> None:
    """Register a fallback handler."""
    FALLBACK_HANDLERS[name] = handler


def get_fallback(name: str) -> Optional[FallbackHandler]:
    """Get a fallback handler by name."""
    return FALLBACK_HANDLERS.get(name)


def initialize_default_fallbacks(
    embedding_dimension: int = 1024,
    rerank_top_k: int = 5,
) -> None:
    """Initialize default fallback handlers."""
    register_fallback("embedding", EmbeddingFallback(dimension=embedding_dimension))
    register_fallback("rerank", RerankFallback(top_k=rerank_top_k))
    register_fallback("llm", LLMFallback())
    register_fallback("vector_store", VectorStoreFallback())
    register_fallback("fulltext_store", FulltextStoreFallback())
    register_fallback("graph_store", GraphStoreFallback())
    register_fallback("query_rewrite", QueryRewriteFallback())