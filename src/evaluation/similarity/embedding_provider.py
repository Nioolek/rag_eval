"""
Embedding provider base class and implementations.
Supports multiple embedding backends for semantic similarity.
"""

import asyncio
import hashlib
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from ...core.config import get_config
from ...core.logging import logger


class EmbeddingCache:
    """
    Simple in-memory cache for embeddings.
    Uses content hash as key for efficient retrieval.
    """

    def __init__(self, max_size: int = 10000):
        self._cache: dict[str, list[float]] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def _hash(self, text: str) -> str:
        """Generate hash for text content."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[list[float]]:
        """Get embedding from cache."""
        key = self._hash(text)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def set(self, text: str, embedding: list[float]) -> None:
        """Store embedding in cache."""
        if len(self._cache) >= self._max_size:
            # Simple LRU: clear half the cache
            keys = list(self._cache.keys())[:self._max_size // 2]
            for k in keys:
                del self._cache[k]

        key = self._hash(text)
        self._cache[key] = embedding

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
        }

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    """

    def __init__(self, cache: Optional[EmbeddingCache] = None):
        self._cache = cache or EmbeddingCache()
        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Get provider name."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass

    @abstractmethod
    async def _embed_single(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        pass

    async def initialize(self) -> None:
        """Initialize the provider."""
        self._initialized = True

    async def embed(self, text: str) -> list[float]:
        """
        Embed text with caching.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Check cache first
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        # Generate embedding
        if not self._initialized:
            await self.initialize()

        embedding = await self._embed_single(text)

        # Cache the result
        self._cache.set(text, embedding)

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            emb = await self.embed(text)
            embeddings.append(emb)
        return embeddings

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI embedding provider using the OpenAI API.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        cache: Optional[EmbeddingCache] = None,
    ):
        super().__init__(cache)
        self._model = model
        self._api_key = api_key
        self._api_base = api_base

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    @property
    def dimension(self) -> int:
        # OpenAI embedding dimensions
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(self._model, 1536)

    async def _embed_single(self, text: str) -> list[float]:
        """Embed using OpenAI API."""
        try:
            from openai import AsyncOpenAI

            # Get API key from config if not provided
            if not self._api_key:
                config = get_config()
                self._api_key = config.llm.api_key

            client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._api_base,
            )

            response = await client.embeddings.create(
                model=self._model,
                input=text,
            )

            return list(response.data[0].embedding)

        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache: Optional[EmbeddingCache] = None,
    ):
        super().__init__(cache)
        self._model_name = model_name
        self._model = None

    @property
    def name(self) -> str:
        return f"sentence-transformers:{self._model_name}"

    @property
    def dimension(self) -> int:
        # Common model dimensions
        dimensions = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "paraphrase-multilingual-MiniLM-L12-v2": 384,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(self._model_name, 768)

    async def initialize(self) -> None:
        """Load the model."""
        try:
            from sentence_transformers import SentenceTransformer

            # Run in thread pool since model loading is synchronous
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(self._model_name)
            )

            self._initialized = True
            logger.info(f"Loaded sentence-transformers model: {self._model_name}")

        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    async def _embed_single(self, text: str) -> list[float]:
        """Embed using sentence-transformers."""
        if self._model is None:
            await self.initialize()

        # Run encoding in thread pool
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._model.encode(text, convert_to_numpy=True)
        )

        return embedding.tolist()


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Mock embedding provider for testing.
    Generates deterministic embeddings based on text hash.
    """

    def __init__(self, dimension: int = 384, cache: Optional[EmbeddingCache] = None):
        super().__init__(cache)
        self._dimension = dimension

    @property
    def name(self) -> str:
        return "mock"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def _embed_single(self, text: str) -> list[float]:
        """Generate mock embedding based on text hash."""
        # Use text hash as seed for reproducibility
        np.random.seed(hash(text) % (2**32))

        # Generate random embedding
        embedding = np.random.randn(self._dimension).astype(np.float32)

        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()


# Provider factory
def create_embedding_provider(
    provider_type: str = "mock",
    **kwargs: Any,
) -> BaseEmbeddingProvider:
    """
    Create an embedding provider.

    Args:
        provider_type: Type of provider ("openai", "sentence-transformers", "mock")
        **kwargs: Provider-specific arguments

    Returns:
        Embedding provider instance
    """
    providers = {
        "openai": OpenAIEmbeddingProvider,
        "sentence-transformers": SentenceTransformerProvider,
        "mock": MockEmbeddingProvider,
    }

    if provider_type not in providers:
        raise ValueError(f"Unknown provider type: {provider_type}")

    return providers[provider_type](**kwargs)


# Global provider instance
_provider: Optional[BaseEmbeddingProvider] = None


async def get_embedding_provider(
    provider_type: Optional[str] = None,
    **kwargs: Any,
) -> BaseEmbeddingProvider:
    """
    Get the global embedding provider.

    Args:
        provider_type: Optional provider type override
        **kwargs: Provider-specific arguments

    Returns:
        Embedding provider instance
    """
    global _provider

    if _provider is None or provider_type:
        config = get_config()
        provider_type = provider_type or getattr(config, 'embedding_provider', 'mock')

        _provider = create_embedding_provider(provider_type, **kwargs)
        await _provider.initialize()

    return _provider