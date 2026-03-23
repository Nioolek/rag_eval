"""
Rerank Service Implementation.

Alibaba gte-rerank integration for document re-ranking.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from rag_rag.core.exceptions import RerankServiceError, RateLimitError
from rag_rag.core.logging import get_logger

logger = get_logger("rag_rag.services.rerank")


@dataclass
class RerankConfig:
    """Rerank service configuration."""

    api_key: str = ""
    model: str = "gte-rerank"
    top_k: int = 5
    timeout: int = 30
    max_retries: int = 2
    retry_delay: float = 0.5


@dataclass
class RerankResult:
    """Rerank result for a single document."""

    index: int
    document: str
    relevance_score: float


class RerankService:
    """
    Rerank Service with Alibaba gte-rerank backend.

    Features:
    - Document re-ranking
    - Batch processing
    - Retry with exponential backoff
    - Local BM25 fallback
    """

    def __init__(self, config: RerankConfig):
        self.config = config
        self._client: Optional[Any] = None

    async def initialize(self) -> None:
        """Initialize the rerank client."""
        try:
            import dashscope
            from dashscope import TextReRank

            dashscope.api_key = self.config.api_key
            self._client = TextReRank
            logger.info(f"Rerank Service initialized: {self.config.model}")

        except ImportError:
            raise RerankServiceError(
                "dashscope not installed. Install with: pip install dashscope"
            )

    async def close(self) -> None:
        """Close the rerank client."""
        self._client = None

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        reraise=True,
    )
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: Optional[int] = None,
    ) -> list[RerankResult]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Query string
            documents: List of document texts
            top_k: Number of top results (default from config)

        Returns:
            List of RerankResult sorted by relevance
        """
        if not documents:
            return []

        top_k = top_k or self.config.top_k

        if self._client is None:
            await self.initialize()

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.call(
                    model=self.config.model,
                    query=query,
                    documents=documents,
                    top_n=min(top_k, len(documents)),
                    return_documents=True,
                ),
            )

            if response.status_code != 200:
                if response.code == "RateLimit":
                    raise RateLimitError(
                        f"Rate limit exceeded: {response.message}",
                        retry_after=60,
                    )
                raise RerankServiceError(
                    f"Rerank API error: {response.code} - {response.message}"
                )

            # Parse results
            results = []
            for item in response.output["results"]:
                results.append(
                    RerankResult(
                        index=item["index"],
                        document=item["document"],
                        relevance_score=item["relevance_score"],
                    )
                )

            return results

        except RateLimitError:
            # Fallback to local BM25
            logger.warning("Rate limit exceeded, using local BM25 fallback")
            return await self._local_rerank(query, documents, top_k)

        except Exception as e:
            logger.error(f"Rerank failed: {e}")
            # Fallback to local BM25
            logger.warning("Using local BM25 fallback")
            return await self._local_rerank(query, documents, top_k)

    async def _local_rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int,
    ) -> list[RerankResult]:
        """
        Local BM25-based reranking as fallback.

        This is different from initial BM25 retrieval:
        - Initial: searches entire corpus
        - Fallback: re-ranks already retrieved documents
        """
        try:
            from rank_bm25 import BM25Okapi

            # Tokenize documents
            tokenized_docs = [doc.split() for doc in documents]
            bm25 = BM25Okapi(tokenized_docs)

            # Get scores
            tokenized_query = query.split()
            scores = bm25.get_scores(tokenized_query)

            # Sort by score
            indexed_scores = sorted(
                enumerate(scores),
                key=lambda x: x[1],
                reverse=True,
            )

            # Return top_k results
            results = []
            for idx, score in indexed_scores[:top_k]:
                results.append(
                    RerankResult(
                        index=idx,
                        document=documents[idx],
                        relevance_score=float(score),
                    )
                )

            return results

        except ImportError:
            logger.warning("rank-bm25 not installed, returning original order")
            return [
                RerankResult(index=i, document=doc, relevance_score=1.0 - i * 0.1)
                for i, doc in enumerate(documents[:top_k])
            ]

    async def rerank_with_metadata(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank documents with metadata.

        Args:
            query: Query string
            documents: List of document dicts with 'content' and 'metadata'
            top_k: Number of top results

        Returns:
            List of document dicts sorted by relevance with rerank_score
        """
        if not documents:
            return []

        contents = [doc.get("content", "") for doc in documents]
        results = await self.rerank(query, contents, top_k)

        # Map back to original documents with scores
        reranked_docs = []
        for result in results:
            original_doc = documents[result.index]
            reranked_docs.append({
                **original_doc,
                "rerank_score": result.relevance_score,
                "rank": len(reranked_docs) + 1,
            })

        return reranked_docs