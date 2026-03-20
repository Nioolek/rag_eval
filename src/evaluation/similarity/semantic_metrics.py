"""
Semantic similarity-based metrics using embeddings.
"""

import time
from typing import Any, Optional

from ..metrics.base import BaseMetric, MetricContext
from ..metrics.metric_registry import get_registry
from ...models.metric_result import MetricCategory, MetricResult
from .embedding_provider import get_embedding_provider, BaseEmbeddingProvider
from .semantic_similarity import (
    compute_semantic_similarity,
    compute_semantic_similarity_batch,
    compute_semantic_coverage,
)
from ...core.logging import logger


class SemanticAnswerSimilarityMetric(BaseMetric):
    """
    Semantic similarity between generated answer and standard answers.
    Uses embedding-based similarity instead of word overlap.
    """

    name = "semantic_answer_similarity"
    category = MetricCategory.GENERATION
    description = "Semantic similarity between generated answer and standard answers using embeddings"
    requires_llm = False
    threshold = 0.6

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        provider: Optional[BaseEmbeddingProvider] = None,
    ):
        super().__init__(config)
        self._provider = provider

    async def calculate(self, context: MetricContext) -> MetricResult:
        """Calculate semantic answer similarity."""
        answer = context.answer
        standard_answers = context.standard_answers

        if not answer:
            return self._create_result(
                score=0.0,
                details={"error": "No answer to evaluate"},
            )

        if not standard_answers:
            return self._create_result(
                score=0.0,
                details={"error": "No standard answers provided"},
            )

        try:
            # Get provider
            if self._provider is None:
                self._provider = await get_embedding_provider()

            # Compute similarities to all standard answers
            similarities = await compute_semantic_similarity_batch(
                answer,
                standard_answers,
                self._provider,
            )

            # Take maximum similarity (best match)
            max_similarity = max(similarities) if similarities else 0.0
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

            return self._create_result(
                score=max_similarity,
                raw_score=max_similarity,
                details={
                    "max_similarity": max_similarity,
                    "avg_similarity": avg_similarity,
                    "per_answer_similarity": {
                        f"standard_{i}": sim
                        for i, sim in enumerate(similarities)
                    },
                    "provider": self._provider.name,
                },
            )

        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return self._error_result(str(e))


class SemanticRelevanceMetric(BaseMetric):
    """
    Semantic relevance between query and generated answer.
    Measures how well the answer addresses the query semantically.
    """

    name = "semantic_relevance"
    category = MetricCategory.GENERATION
    description = "Semantic relevance between query and generated answer"
    requires_llm = False
    threshold = 0.5

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        provider: Optional[BaseEmbeddingProvider] = None,
    ):
        super().__init__(config)
        self._provider = provider

    async def calculate(self, context: MetricContext) -> MetricResult:
        """Calculate query-answer semantic relevance."""
        query = context.query
        answer = context.answer

        if not answer:
            return self._create_result(
                score=0.0,
                details={"error": "No answer to evaluate"},
            )

        if not query:
            return self._create_result(
                score=0.0,
                details={"error": "No query provided"},
            )

        try:
            if self._provider is None:
                self._provider = await get_embedding_provider()

            similarity = await compute_semantic_similarity(
                query,
                answer,
                self._provider,
            )

            return self._create_result(
                score=similarity,
                raw_score=similarity,
                details={
                    "query_answer_similarity": similarity,
                    "provider": self._provider.name,
                },
            )

        except Exception as e:
            logger.error(f"Semantic relevance calculation failed: {e}")
            return self._error_result(str(e))


class SemanticCoverageMetric(BaseMetric):
    """
    Semantic coverage of reference content by generated answer.
    Measures how well the answer covers key information from references.
    """

    name = "semantic_coverage"
    category = MetricCategory.GENERATION
    description = "Semantic coverage of reference documents by generated answer"
    requires_llm = False
    threshold = 0.5

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        provider: Optional[BaseEmbeddingProvider] = None,
    ):
        super().__init__(config)
        self._provider = provider

    async def calculate(self, context: MetricContext) -> MetricResult:
        """Calculate semantic coverage of retrieved documents."""
        answer = context.answer
        retrieved_docs = context.retrieved_documents

        if not answer:
            return self._create_result(
                score=0.0,
                details={"error": "No answer to evaluate"},
            )

        if not retrieved_docs:
            return self._create_result(
                score=0.0,
                details={"error": "No retrieved documents"},
            )

        try:
            if self._provider is None:
                self._provider = await get_embedding_provider()

            # Compute coverage for each document
            coverage_scores = []
            for doc in retrieved_docs[:5]:  # Limit to top 5 docs
                coverage = await compute_semantic_coverage(
                    answer,
                    doc,
                    self._provider,
                    sentence_level=False,
                )
                coverage_scores.append(coverage["coverage_score"])

            # Average coverage across documents
            avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0
            max_coverage = max(coverage_scores) if coverage_scores else 0.0

            return self._create_result(
                score=avg_coverage,
                raw_score=avg_coverage,
                details={
                    "avg_coverage": avg_coverage,
                    "max_coverage": max_coverage,
                    "per_doc_coverage": coverage_scores,
                    "doc_count": len(retrieved_docs),
                    "provider": self._provider.name,
                },
            )

        except Exception as e:
            logger.error(f"Semantic coverage calculation failed: {e}")
            return self._error_result(str(e))


class RetrievalSemanticRelevanceMetric(BaseMetric):
    """
    Semantic relevance of retrieved documents to the query.
    Uses embeddings to measure document-query relevance.
    """

    name = "retrieval_semantic_relevance"
    category = MetricCategory.RETRIEVAL
    description = "Semantic relevance of retrieved documents to query using embeddings"
    requires_llm = False
    threshold = 0.5

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        provider: Optional[BaseEmbeddingProvider] = None,
        top_k: int = 5,
    ):
        super().__init__(config)
        self._provider = provider
        self._top_k = top_k

    async def calculate(self, context: MetricContext) -> MetricResult:
        """Calculate semantic relevance of retrieved documents."""
        query = context.query
        retrieved_docs = context.retrieved_documents

        if not retrieved_docs:
            return self._create_result(
                score=0.0,
                details={"error": "No retrieved documents"},
            )

        if not query:
            return self._create_result(
                score=0.0,
                details={"error": "No query provided"},
            )

        try:
            if self._provider is None:
                self._provider = await get_embedding_provider()

            # Compute relevance for top-k documents
            docs_to_eval = retrieved_docs[:self._top_k]
            similarities = await compute_semantic_similarity_batch(
                query,
                docs_to_eval,
                self._provider,
            )

            # Compute metrics
            avg_relevance = sum(similarities) / len(similarities) if similarities else 0.0
            max_relevance = max(similarities) if similarities else 0.0

            # Count documents above threshold
            relevant_count = sum(1 for s in similarities if s >= self.threshold)
            relevant_ratio = relevant_count / len(similarities) if similarities else 0.0

            return self._create_result(
                score=avg_relevance,
                raw_score=avg_relevance,
                details={
                    "avg_relevance": avg_relevance,
                    "max_relevance": max_relevance,
                    "relevant_count": relevant_count,
                    "relevant_ratio": relevant_ratio,
                    "per_doc_relevance": similarities,
                    "doc_count": len(docs_to_eval),
                    "provider": self._provider.name,
                },
            )

        except Exception as e:
            logger.error(f"Retrieval semantic relevance calculation failed: {e}")
            return self._error_result(str(e))


class GroundTruthSimilarityMetric(BaseMetric):
    """
    Similarity between generated answer and ground truth documents.
    Measures alignment with expected reference content.
    """

    name = "ground_truth_similarity"
    category = MetricCategory.COMPREHENSIVE
    description = "Semantic similarity between answer and ground truth documents"
    requires_llm = False
    threshold = 0.5

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        provider: Optional[BaseEmbeddingProvider] = None,
    ):
        super().__init__(config)
        self._provider = provider

    async def calculate(self, context: MetricContext) -> MetricResult:
        """Calculate similarity to ground truth documents."""
        answer = context.answer
        gt_documents = context.gt_documents

        if not answer:
            return self._create_result(
                score=0.0,
                details={"error": "No answer to evaluate"},
            )

        if not gt_documents:
            return self._create_result(
                score=0.0,
                details={"error": "No ground truth documents provided"},
            )

        try:
            if self._provider is None:
                self._provider = await get_embedding_provider()

            # Compute similarity to each ground truth document
            similarities = await compute_semantic_similarity_batch(
                answer,
                gt_documents,
                self._provider,
            )

            max_sim = max(similarities) if similarities else 0.0
            avg_sim = sum(similarities) / len(similarities) if similarities else 0.0

            return self._create_result(
                score=max_sim,
                raw_score=max_sim,
                details={
                    "max_similarity": max_sim,
                    "avg_similarity": avg_sim,
                    "per_doc_similarity": similarities,
                    "provider": self._provider.name,
                },
            )

        except Exception as e:
            logger.error(f"Ground truth similarity calculation failed: {e}")
            return self._error_result(str(e))


# Register semantic metrics
def register_semantic_metrics():
    """Register all semantic similarity metrics."""
    registry = get_registry()

    registry.register(SemanticAnswerSimilarityMetric)
    registry.register(SemanticRelevanceMetric)
    registry.register(SemanticCoverageMetric)
    registry.register(RetrievalSemanticRelevanceMetric)
    registry.register(GroundTruthSimilarityMetric)

    logger.info("Registered semantic similarity metrics")