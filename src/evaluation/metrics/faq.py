"""
FAQ-specific metrics for RAG evaluation.
Metrics: FAQ Match Accuracy, FAQ Recall, FAQ Answer Consistency
"""

from typing import Any, Optional

from .base import BaseMetric, MetricContext
from ...models.metric_result import MetricCategory, MetricResult
from ...core.logging import logger


class FAQMatchAccuracyMetric(BaseMetric):
    """
    FAQ 匹配准确率
    评估FAQ匹配是否正确。
    """

    name = "faq_match_accuracy"
    category = MetricCategory.FAQ
    description = "FAQ匹配准确率：评估FAQ匹配是否正确"
    requires_llm = False
    threshold = 0.8

    async def calculate(self, context: MetricContext) -> MetricResult:
        faq_matched_expected = context.annotation.faq_matched
        faq_response = context.rag_response.faq_match

        # Check if FAQ matching is correct
        faq_matched_actual = faq_response.matched if faq_response else False

        # Calculate accuracy
        if faq_matched_expected and faq_matched_actual:
            # True Positive: should match and did match
            outcome = "true_positive"
            score = 1.0
        elif not faq_matched_expected and not faq_matched_actual:
            # True Negative: should not match and did not match
            outcome = "true_negative"
            score = 1.0
        elif not faq_matched_expected and faq_matched_actual:
            # False Positive: incorrectly matched
            outcome = "false_positive"
            score = 0.0
        else:
            # False Negative: should have matched but didn't
            outcome = "false_negative"
            score = 0.0

        details = {
            "expected_matched": faq_matched_expected,
            "actual_matched": faq_matched_actual,
            "outcome": outcome,
        }

        # If matched, include confidence
        if faq_response and faq_matched_actual:
            details["confidence"] = faq_response.confidence
            details["similarity_score"] = faq_response.similarity_score

        return self._create_result(score=score, details=details)


class FAQRecallMetric(BaseMetric):
    """
    FAQ 召回率
    评估应该匹配FAQ的情况是否被正确识别。
    """

    name = "faq_recall"
    category = MetricCategory.FAQ
    description = "FAQ召回率：应该匹配FAQ时是否被正确识别"
    requires_llm = False
    threshold = 0.8

    async def calculate(self, context: MetricContext) -> MetricResult:
        faq_matched_expected = context.annotation.faq_matched
        faq_response = context.rag_response.faq_match

        # Only consider cases where FAQ should be matched
        if not faq_matched_expected:
            return self._create_result(
                score=1.0,  # N/A, treated as success
                details={
                    "reason": "FAQ match not expected for this query",
                    "applicable": False,
                }
            )

        # Check if FAQ was correctly matched
        faq_matched_actual = faq_response.matched if faq_response else False

        if faq_matched_actual:
            score = 1.0
            recall_status = "recall_success"
        else:
            score = 0.0
            recall_status = "recall_failure"

        details = {
            "applicable": True,
            "expected_matched": faq_matched_expected,
            "actual_matched": faq_matched_actual,
            "recall_status": recall_status,
        }

        if faq_response and faq_matched_actual:
            details["matched_faq_id"] = faq_response.faq_id
            details["confidence"] = faq_response.confidence

        return self._create_result(score=score, details=details)


class FAQAnswerConsistencyMetric(BaseMetric):
    """
    FAQ 答案与标准答案的一致性
    评估FAQ答案与标准答案的匹配程度。
    """

    name = "faq_answer_consistency"
    category = MetricCategory.FAQ
    description = "FAQ答案与标准答案的一致性"
    requires_llm = False
    threshold = 0.7

    async def calculate(self, context: MetricContext) -> MetricResult:
        faq_response = context.rag_response.faq_match
        standard_answers = context.annotation.standard_answers

        # Check if FAQ was matched
        if not faq_response or not faq_response.matched:
            return self._create_result(
                score=0.0,
                details={
                    "reason": "No FAQ match in response",
                    "applicable": False,
                }
            )

        # If no standard answers, can't compare
        if not standard_answers:
            return self._create_result(
                score=1.0,  # Assume correct if no standard to compare
                details={
                    "reason": "No standard answers to compare",
                    "applicable": False,
                }
            )

        # Compare FAQ answer with standard answers
        faq_answer = faq_response.faq_answer
        faq_words = set(faq_answer.lower().split())

        best_match_score = 0.0
        best_match_idx = -1

        for idx, std_answer in enumerate(standard_answers):
            std_words = set(std_answer.lower().split())

            if not std_words:
                continue

            # Calculate word overlap
            overlap = len(faq_words & std_words)
            union = len(faq_words | std_words)

            if union > 0:
                jaccard = overlap / union
            else:
                jaccard = 0

            # Also calculate coverage
            coverage = overlap / len(std_words) if std_words else 0

            # Combined score
            match_score = (jaccard + coverage) / 2

            if match_score > best_match_score:
                best_match_score = match_score
                best_match_idx = idx

        details = {
            "applicable": True,
            "faq_answer": faq_answer[:200],  # Truncate for storage
            "best_match_index": best_match_idx,
            "best_match_score": best_match_score,
            "confidence": faq_response.confidence,
        }

        return self._create_result(
            score=best_match_score,
            details=details
        )