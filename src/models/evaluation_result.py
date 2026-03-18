"""
Evaluation result model.
Stores complete evaluation run results.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from .annotation import Annotation
from .rag_response import RAGResponse
from .metric_result import MetricSummary, MetricResult


class EvaluationResult(BaseModel):
    """
    Single evaluation result for one annotation.
    Contains RAG response and all metric results.
    """
    # Identification
    id: str = Field(default_factory=lambda: str(uuid4()))
    annotation_id: str
    run_id: str = ""

    # RAG Interface info
    rag_interface: str = "default"  # Interface name for dual-RAG comparison

    # Results
    annotation: Optional[Annotation] = None
    rag_response: Optional[RAGResponse] = None
    metrics: MetricSummary = Field(default_factory=MetricSummary)

    # Status
    success: bool = True
    error_message: str = ""
    duration_ms: float = 0.0

    # Timestamps
    evaluated_at: datetime = Field(default_factory=datetime.now)

    def add_metric(self, result: MetricResult) -> None:
        """Add a metric result."""
        self.metrics.add_result(result)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "annotation_id": self.annotation_id,
            "run_id": self.run_id,
            "rag_interface": self.rag_interface,
            "annotation": self.annotation.to_dict() if self.annotation else None,
            "rag_response": self.rag_response.to_dict() if self.rag_response else None,
            "metrics": self.metrics.to_dict(),
            "success": self.success,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "evaluated_at": self.evaluated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationResult":
        """Create from dictionary."""
        if isinstance(data.get("evaluated_at"), str):
            data["evaluated_at"] = datetime.fromisoformat(data["evaluated_at"])
        if data.get("annotation"):
            data["annotation"] = Annotation.from_dict(data["annotation"])
        if data.get("rag_response"):
            data["rag_response"] = RAGResponse.from_dict(data["rag_response"])
        if data.get("metrics"):
            data["metrics"] = MetricSummary(**data["metrics"])
        return cls(**data)


class EvaluationRun(BaseModel):
    """
    Complete evaluation run with multiple results.
    Supports dual-RAG comparison mode.
    """
    # Identification
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = ""

    # Configuration
    rag_interfaces: list[str] = Field(default_factory=list)  # ["default"] or ["left", "right"]
    selected_metrics: list[str] = Field(default_factory=list)
    concurrent_workers: int = 10

    # Results
    results: list[EvaluationResult] = Field(default_factory=list)

    # Statistics
    total_annotations: int = 0
    completed_count: int = 0
    failed_count: int = 0

    # Timing
    started_at: datetime = Field(default_factory=datetime.now)
    finished_at: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Status
    status: str = "pending"  # pending, running, completed, cancelled, failed

    # Summary statistics per interface
    summary_by_interface: dict[str, dict[str, float]] = Field(default_factory=dict)

    def add_result(self, result: EvaluationResult) -> None:
        """Add an evaluation result."""
        result.run_id = self.id
        self.results.append(result)
        self.completed_count += 1
        if not result.success:
            self.failed_count += 1
        self._update_summary()

    def _update_summary(self) -> None:
        """Update summary statistics."""
        interface_results: dict[str, list[EvaluationResult]] = {}
        for r in self.results:
            iface = r.rag_interface
            if iface not in interface_results:
                interface_results[iface] = []
            interface_results[iface].append(r)

        for iface, results in interface_results.items():
            if results:
                avg_score = sum(r.metrics.average_score for r in results if r.success) / len([r for r in results if r.success]) if any(r.success for r in results) else 0
                success_rate = len([r for r in results if r.success]) / len(results) if results else 0
                self.summary_by_interface[iface] = {
                    "average_score": avg_score,
                    "success_rate": success_rate,
                    "total": len(results),
                    "successful": len([r for r in results if r.success]),
                }

    def finish(self) -> None:
        """Mark the run as completed."""
        self.finished_at = datetime.now()
        self.status = "completed"
        if self.started_at and self.finished_at:
            self.duration_seconds = (self.finished_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "rag_interfaces": self.rag_interfaces,
            "selected_metrics": self.selected_metrics,
            "concurrent_workers": self.concurrent_workers,
            "results": [r.to_dict() for r in self.results],
            "total_annotations": self.total_annotations,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "summary_by_interface": self.summary_by_interface,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationRun":
        """Create from dictionary."""
        if isinstance(data.get("started_at"), str):
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        if isinstance(data.get("finished_at"), str):
            data["finished_at"] = datetime.fromisoformat(data["finished_at"])
        if data.get("results"):
            data["results"] = [EvaluationResult.from_dict(r) for r in data["results"]]
        return cls(**data)