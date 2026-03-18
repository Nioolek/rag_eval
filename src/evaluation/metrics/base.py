"""
Base metric class for all evaluation metrics.
Implements Strategy pattern for extensible metric calculations.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from ...models.annotation import Annotation
from ...models.rag_response import RAGResponse
from ...models.metric_result import MetricResult, MetricCategory
from ...core.exceptions import MetricCalculationError
from ...core.logging import logger


@dataclass
class MetricContext:
    """
    Context for metric calculation.
    Contains all data needed for evaluation.
    """
    annotation: Annotation
    rag_response: RAGResponse
    llm_client: Optional[Any] = None  # LangChain LLM client for LLM-based metrics
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def query(self) -> str:
        """Get the query."""
        return self.annotation.query

    @property
    def answer(self) -> str:
        """Get the RAG answer."""
        return self.rag_response.final_answer

    @property
    def gt_documents(self) -> list[str]:
        """Get ground truth documents."""
        return self.annotation.gt_documents

    @property
    def retrieved_documents(self) -> list[str]:
        """Get retrieved document contents."""
        return self.rag_response.get_retrieved_contents()

    @property
    def standard_answers(self) -> list[str]:
        """Get standard answers."""
        return self.annotation.standard_answers

    @property
    def should_refuse(self) -> bool:
        """Check if should refuse to answer."""
        return self.annotation.should_refuse

    @property
    def is_refused(self) -> bool:
        """Check if RAG refused to answer."""
        return self.rag_response.is_refused


class BaseMetric(ABC):
    """
    Abstract base class for all evaluation metrics.
    Implements the Strategy pattern.

    Each metric is an independent strategy that can be:
    - Registered in the metric registry
    - Created by the metric factory
    - Applied to annotation-response pairs

    To add a new metric:
    1. Subclass BaseMetric
    2. Implement calculate() method
    3. Register in MetricRegistry
    """

    # Class-level metadata
    name: str = "base_metric"
    category: MetricCategory = MetricCategory.COMPREHENSIVE
    description: str = "Base metric class"
    requires_llm: bool = False
    threshold: float = 0.5

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """
        Initialize metric with optional configuration.

        Args:
            config: Metric-specific configuration
        """
        self.config = config or {}
        self.threshold = self.config.get("threshold", self.threshold)

    @abstractmethod
    async def calculate(self, context: MetricContext) -> MetricResult:
        """
        Calculate the metric score.

        Args:
            context: Metric calculation context

        Returns:
            MetricResult with score and details
        """
        pass

    def _create_result(
        self,
        score: float,
        raw_score: Optional[float] = None,
        details: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
        computation_time_ms: float = 0.0,
    ) -> MetricResult:
        """
        Create a standardized metric result.

        Args:
            score: Normalized score (0-1)
            raw_score: Raw score before normalization
            details: Additional details
            error: Error message if failed
            computation_time_ms: Time taken to compute

        Returns:
            MetricResult
        """
        passed = score >= self.threshold

        return MetricResult(
            metric_name=self.name,
            metric_category=self.category,
            description=self.description,
            score=score,
            raw_score=raw_score,
            passed=passed,
            threshold=self.threshold,
            details=details or {},
            error=error,
            requires_llm=self.requires_llm,
            computation_time_ms=computation_time_ms,
        )

    def _error_result(self, error: str) -> MetricResult:
        """Create an error result."""
        return MetricResult.error_result(
            metric_name=self.name,
            metric_category=self.category,
            error=error,
        )

    async def evaluate(self, context: MetricContext) -> MetricResult:
        """
        Evaluate the metric with timing and error handling.

        Args:
            context: Metric calculation context

        Returns:
            MetricResult
        """
        start_time = time.time()

        try:
            result = await self.calculate(context)
            result.computation_time_ms = (time.time() - start_time) * 1000
            return result

        except Exception as e:
            logger.error(f"Metric {self.name} calculation failed: {e}")
            return self._error_result(str(e))

    @classmethod
    def get_info(cls) -> dict[str, Any]:
        """Get metric information."""
        return {
            "name": cls.name,
            "category": cls.category.value,
            "description": cls.description,
            "requires_llm": cls.requires_llm,
            "default_threshold": cls.threshold,
        }


class SimpleMetric(BaseMetric):
    """
    Simple metric for basic calculations without LLM.
    Useful for rule-based metrics.
    """

    def __init__(
        self,
        name: str,
        calculate_func: callable,
        category: MetricCategory = MetricCategory.COMPREHENSIVE,
        description: str = "",
        threshold: float = 0.5,
    ):
        """
        Initialize simple metric with a calculation function.

        Args:
            name: Metric name
            calculate_func: Function(context) -> float (0-1)
            category: Metric category
            description: Metric description
            threshold: Pass threshold
        """
        self._name = name
        self._calculate_func = calculate_func
        self._category = category
        self._description = description
        self.threshold = threshold

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> MetricCategory:
        return self._category

    @property
    def description(self) -> str:
        return self._description

    async def calculate(self, context: MetricContext) -> MetricResult:
        """Calculate using the provided function."""
        score = self._calculate_func(context)
        return self._create_result(score=score)