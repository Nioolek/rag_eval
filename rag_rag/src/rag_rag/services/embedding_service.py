"""
Embedding Service Implementation.

Alibaba text-embedding-v3 integration for semantic embeddings.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from rag_rag.core.exceptions import EmbeddingServiceError, RateLimitError
from rag_rag.core.logging import get_logger

logger = get_logger("rag_rag.services.embedding")


@dataclass
class EmbeddingConfig:
    """Embedding service configuration."""

    api_key: str = ""
    model: str = "text-embedding-v3"
    dimension: int = 1024
    batch_size: int = 20
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 0.5


class EmbeddingService:
    """
    Embedding Service with Alibaba text-embedding-v3 backend.

    Features:
    - Batch embedding support
    - Automatic batching
    - Retry with exponential backoff
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._client: Optional[Any] = None

    async def initialize(self) -> None:
        """Initialize the embedding client."""
        try:
            import dashscope
            from dashscope import TextEmbedding

            dashscope.api_key = self.config.api_key
            self._client = TextEmbedding
            logger.info(
                f"Embedding Service initialized: {self.config.model} "
                f"(dim={self.config.dimension})"
            )

        except ImportError:
            raise EmbeddingServiceError(
                "dashscope not installed. Install with: pip install dashscope"
            )

    async def close(self) -> None:
        """Close the embedding client."""
        self._client = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        reraise=True,
    )
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if self._client is None:
            await self.initialize()

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]

            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.call(
                        model=self.config.model,
                        input=batch,
                        dimension=self.config.dimension,
                    ),
                )

                if response.status_code != 200:
                    if response.code == "RateLimit":
                        raise RateLimitError(
                            f"Rate limit exceeded: {response.message}",
                            retry_after=60,
                        )
                    raise EmbeddingServiceError(
                        f"Embedding API error: {response.code} - {response.message}"
                    )

                # Extract embeddings
                batch_embeddings = [
                    item.embedding for item in response.output["embeddings"]
                ]
                all_embeddings.extend(batch_embeddings)

            except RateLimitError:
                raise
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                raise EmbeddingServiceError(f"Embedding generation failed: {e}")

        return all_embeddings

    async def embed_single(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        embeddings = await self.embed([text])
        return embeddings[0] if embeddings else []

    async def similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """
        Calculate semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        import numpy as np

        embeddings = await self.embed([text1, text2])

        if len(embeddings) < 2:
            return 0.0

        vec1 = np.array(embeddings[0])
        vec2 = np.array(embeddings[1])

        similarity = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        return similarity

    async def batch_similarity(
        self,
        query: str,
        texts: list[str],
    ) -> list[float]:
        """
        Calculate similarity between query and multiple texts.

        Args:
            query: Query text
            texts: List of texts to compare

        Returns:
            List of similarity scores
        """
        import numpy as np

        all_texts = [query] + texts
        embeddings = await self.embed(all_texts)

        if len(embeddings) < len(all_texts):
            return [0.0] * len(texts)

        query_embedding = np.array(embeddings[0])
        text_embeddings = np.array(embeddings[1:])

        # Calculate cosine similarities
        similarities = []
        for text_emb in text_embeddings:
            sim = float(
                np.dot(query_embedding, text_emb)
                / (np.linalg.norm(query_embedding) * np.linalg.norm(text_emb))
            )
            similarities.append(sim)

        return similarities