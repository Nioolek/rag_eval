"""
Retrieval metrics for RAG evaluation.
Metrics: Precision, Recall, MRR, Hit Rate, Relevance
"""

import math
from typing import Any, Optional, Set

from .base import BaseMetric, MetricContext
from ...models.metric_result import MetricCategory, MetricResult
from ...core.logging import logger


class RetrievalPrecisionMetric(BaseMetric):
    """
    检索精确率 (Retrieval Precision)
    计算检索结果中相关文档的比例。

    Precision = |Retrieved ∩ Relevant| / |Retrieved|
    """

    name = "retrieval_precision"
    category = MetricCategory.RETRIEVAL
    description = "检索精确率：检索结果中相关文档的比例"
    requires_llm = False
    threshold = 0.7

    async def calculate(self, context: MetricContext) -> MetricResult:
        retrieved = context.retrieved_documents
        gt_docs = context.gt_documents

        if not retrieved:
            return self._create_result(
                score=0.0,
                details={"reason": "No documents retrieved"}
            )

        if not gt_docs:
            return self._create_result(
                score=0.0,
                details={"reason": "No ground truth documents"}
            )

        # Calculate overlap using content similarity
        relevant_count = 0
        for ret_doc in retrieved:
            for gt_doc in gt_docs:
                if self._is_similar(ret_doc, gt_doc):
                    relevant_count += 1
                    break

        precision = relevant_count / len(retrieved)

        return self._create_result(
            score=precision,
            raw_score=precision,
            details={
                "retrieved_count": len(retrieved),
                "relevant_count": relevant_count,
                "precision": precision,
            }
        )

    def _is_similar(self, text1: str, text2: str, threshold: float = 0.5) -> bool:
        """Check if two texts are similar using word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        similarity = overlap / min(len(words1), len(words2))

        return similarity >= threshold


class RetrievalRecallMetric(BaseMetric):
    """
    检索召回率 (Retrieval Recall)
    计算相关文档被检索到的比例。

    Recall = |Retrieved ∩ Relevant| / |Relevant|
    """

    name = "retrieval_recall"
    category = MetricCategory.RETRIEVAL
    description = "检索召回率：相关文档被检索到的比例"
    requires_llm = False
    threshold = 0.7

    async def calculate(self, context: MetricContext) -> MetricResult:
        retrieved = context.retrieved_documents
        gt_docs = context.gt_documents

        if not gt_docs:
            return self._create_result(
                score=0.0,
                details={"reason": "No ground truth documents"}
            )

        if not retrieved:
            return self._create_result(
                score=0.0,
                details={"reason": "No documents retrieved"}
            )

        # Calculate how many GT docs were found
        found_count = 0
        for gt_doc in gt_docs:
            for ret_doc in retrieved:
                if self._is_similar(ret_doc, gt_doc):
                    found_count += 1
                    break

        recall = found_count / len(gt_docs)

        return self._create_result(
            score=recall,
            raw_score=recall,
            details={
                "gt_count": len(gt_docs),
                "found_count": found_count,
                "recall": recall,
            }
        )

    def _is_similar(self, text1: str, text2: str, threshold: float = 0.5) -> bool:
        """Check text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        similarity = overlap / min(len(words1), len(words2))

        return similarity >= threshold


class MRRMetric(BaseMetric):
    """
    平均倒数排名 (Mean Reciprocal Rank)
    计算第一个相关文档的排名倒数。

    MRR = 1 / rank(first relevant doc)
    """

    name = "mrr"
    category = MetricCategory.RETRIEVAL
    description = "平均倒数排名：第一个相关文档的排名倒数"
    requires_llm = False
    threshold = 0.5

    async def calculate(self, context: MetricContext) -> MetricResult:
        retrieved = context.retrieved_documents
        gt_docs = context.gt_documents

        if not gt_docs:
            return self._create_result(
                score=0.0,
                details={"reason": "No ground truth documents"}
            )

        if not retrieved:
            return self._create_result(
                score=0.0,
                details={"reason": "No documents retrieved"}
            )

        # Find first relevant document
        for rank, ret_doc in enumerate(retrieved, 1):
            for gt_doc in gt_docs:
                if self._is_similar(ret_doc, gt_doc):
                    mrr = 1.0 / rank
                    return self._create_result(
                        score=mrr,
                        raw_score=mrr,
                        details={
                            "first_relevant_rank": rank,
                            "mrr": mrr,
                        }
                    )

        # No relevant document found
        return self._create_result(
            score=0.0,
            details={
                "reason": "No relevant document found in results",
                "first_relevant_rank": None,
            }
        )

    def _is_similar(self, text1: str, text2: str, threshold: float = 0.5) -> bool:
        """Check text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        similarity = overlap / min(len(words1), len(words2))

        return similarity >= threshold


class HitRateMetric(BaseMetric):
    """
    Hit Rate@k
    计算前k个结果中是否包含相关文档。

    Hit Rate@k = 1 if relevant doc in top-k, else 0
    """

    name = "hit_rate"
    category = MetricCategory.RETRIEVAL
    description = "Hit Rate@k：前k个结果中是否包含相关文档"
    requires_llm = False
    threshold = 0.5

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        self.k = self.config.get("k", 5)

    async def calculate(self, context: MetricContext) -> MetricResult:
        retrieved = context.retrieved_documents
        gt_docs = context.gt_documents

        if not gt_docs:
            return self._create_result(
                score=0.0,
                details={"reason": "No ground truth documents"}
            )

        # Check top-k
        top_k = retrieved[:self.k]

        if not top_k:
            return self._create_result(
                score=0.0,
                details={"reason": f"No documents in top-{self.k}"}
            )

        # Check if any GT doc is in top-k
        hit = False
        for ret_doc in top_k:
            for gt_doc in gt_docs:
                if self._is_similar(ret_doc, gt_doc):
                    hit = True
                    break
            if hit:
                break

        hit_rate = 1.0 if hit else 0.0

        return self._create_result(
            score=hit_rate,
            raw_score=hit_rate,
            details={
                "k": self.k,
                "hit": hit,
                "top_k_count": len(top_k),
            }
        )

    @classmethod
    def get_info(cls) -> dict[str, Any]:
        info = super().get_info()
        info["default_k"] = 5
        return info

    def _is_similar(self, text1: str, text2: str, threshold: float = 0.5) -> bool:
        """Check text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        similarity = overlap / min(len(words1), len(words2))

        return similarity >= threshold


class RetrievalRelevanceMetric(BaseMetric):
    """
    检索片段与问题的相关性得分
    计算检索文档与查询的相关性。

    Uses simple keyword overlap and TF-IDF-like scoring.
    """

    name = "retrieval_relevance"
    category = MetricCategory.RETRIEVAL
    description = "检索片段与问题的相关性得分"
    requires_llm = False
    threshold = 0.5

    async def calculate(self, context: MetricContext) -> MetricResult:
        query = context.query
        retrieved = context.retrieved_documents

        if not retrieved:
            return self._create_result(
                score=0.0,
                details={"reason": "No documents retrieved"}
            )

        # Calculate relevance for each document
        relevance_scores = []
        query_words = set(query.lower().split())

        for doc in retrieved:
            # Use list for proper word counting (set.count() doesn't work)
            doc_words_list = doc.lower().split()
            doc_words = set(doc_words_list)

            if not doc_words:
                relevance_scores.append(0.0)
                continue

            # Keyword overlap
            overlap = len(query_words & doc_words)
            overlap_score = overlap / len(query_words) if query_words else 0

            # IDF-like weighting (simplified) - use list for counting
            idf_score = sum(
                1.0 / (doc_words_list.count(w) + 1)
                for w in query_words
                if w in doc_words
            ) / len(query_words) if query_words else 0

            # Combined score
            score = (overlap_score + idf_score) / 2
            relevance_scores.append(score)

        # Average relevance
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0

        return self._create_result(
            score=avg_relevance,
            raw_score=avg_relevance,
            details={
                "document_count": len(retrieved),
                "relevance_scores": relevance_scores[:10],  # Top 10 only
                "average_relevance": avg_relevance,
            }
        )