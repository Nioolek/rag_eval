"""
Semantic similarity module for RAG evaluation.

Provides embedding-based similarity calculations and metrics.
"""

from .embedding_provider import (
    BaseEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SentenceTransformerProvider,
    MockEmbeddingProvider,
    EmbeddingCache,
    create_embedding_provider,
    get_embedding_provider,
)

from .semantic_similarity import (
    compute_cosine_similarity,
    compute_semantic_similarity,
    compute_semantic_similarity_batch,
    find_most_similar,
    compute_semantic_coverage,
)

from .semantic_metrics import (
    SemanticAnswerSimilarityMetric,
    SemanticRelevanceMetric,
    SemanticCoverageMetric,
    RetrievalSemanticRelevanceMetric,
    GroundTruthSimilarityMetric,
    register_semantic_metrics,
)

__all__ = [
    # Providers
    "BaseEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "SentenceTransformerProvider",
    "MockEmbeddingProvider",
    "EmbeddingCache",
    "create_embedding_provider",
    "get_embedding_provider",
    # Similarity functions
    "compute_cosine_similarity",
    "compute_semantic_similarity",
    "compute_semantic_similarity_batch",
    "find_most_similar",
    "compute_semantic_coverage",
    # Metrics
    "SemanticAnswerSimilarityMetric",
    "SemanticRelevanceMetric",
    "SemanticCoverageMetric",
    "RetrievalSemanticRelevanceMetric",
    "GroundTruthSimilarityMetric",
    "register_semantic_metrics",
]