"""
Metric result model.
Standardized output format for all evaluation metrics.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class MetricCategory(str, Enum):
    """Categories of evaluation metrics."""
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    FAQ = "faq"
    COMPREHENSIVE = "comprehensive"
    PERFORMANCE = "performance"


class MetricResult(BaseModel):
    """
    Standardized metric calculation result.
    All metrics must return this format.
    """
    # Metric identification
    metric_name: str
    metric_category: MetricCategory
    description: str = ""

    # Score
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score between 0 and 1")
    raw_score: Optional[float] = Field(None, description="Raw score before normalization")

    # Interpretation
    passed: bool = True  # Whether the evaluation passed a threshold
    threshold: float = 0.5  # Threshold for pass/fail

    # Details
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details")
    error: Optional[str] = Field(None, description="Error message if calculation failed")

    # Metadata
    requires_llm: bool = Field(default=False, description="Whether LLM was used")
    computation_time_ms: float = Field(default=0.0, description="Time taken to compute")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "metric_name": self.metric_name,
            "metric_category": self.metric_category.value,
            "description": self.description,
            "score": self.score,
            "raw_score": self.raw_score,
            "passed": self.passed,
            "threshold": self.threshold,
            "details": self.details,
            "error": self.error,
            "requires_llm": self.requires_llm,
            "computation_time_ms": self.computation_time_ms,
        }

    @classmethod
    def error_result(
        cls,
        metric_name: str,
        metric_category: MetricCategory,
        error: str
    ) -> "MetricResult":
        """Create an error result."""
        return cls(
            metric_name=metric_name,
            metric_category=metric_category,
            score=0.0,
            passed=False,
            error=error,
        )


class MetricSummary(BaseModel):
    """Summary of multiple metric results."""
    total_metrics: int = 0
    passed_metrics: int = 0
    failed_metrics: int = 0
    average_score: float = 0.0
    category_scores: dict[str, float] = Field(default_factory=dict)
    metrics: list[MetricResult] = Field(default_factory=list)

    def add_result(self, result: MetricResult) -> None:
        """Add a metric result to the summary."""
        self.metrics.append(result)
        self.total_metrics += 1
        if result.passed:
            self.passed_metrics += 1
        else:
            self.failed_metrics += 1
        self._recalculate()

    def _recalculate(self) -> None:
        """Recalculate summary statistics."""
        if self.metrics:
            self.average_score = sum(m.score for m in self.metrics) / len(self.metrics)
            # Group by category
            category_totals: dict[str, list[float]] = {}
            for m in self.metrics:
                cat = m.metric_category.value
                if cat not in category_totals:
                    category_totals[cat] = []
                category_totals[cat].append(m.score)
            self.category_scores = {
                cat: sum(scores) / len(scores)
                for cat, scores in category_totals.items()
            }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_metrics": self.total_metrics,
            "passed_metrics": self.passed_metrics,
            "failed_metrics": self.failed_metrics,
            "average_score": self.average_score,
            "category_scores": self.category_scores,
            "metrics": [m.to_dict() for m in self.metrics],
        }