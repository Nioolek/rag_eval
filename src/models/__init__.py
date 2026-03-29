"""Models module: data models for the RAG evaluation system."""

from .annotation import Annotation, AnnotationField, AnnotationList
from .dataset import Dataset, DatasetList, DatasetStatus, DatasetSummary
from .rag_response import RAGResponse, RetrievalResult, FAQMatch, QueryRewrite
from .metric_result import MetricResult, MetricCategory
from .evaluation_result import EvaluationResult, EvaluationRun

__all__ = [
    "Annotation",
    "AnnotationField",
    "AnnotationList",
    "Dataset",
    "DatasetList",
    "DatasetStatus",
    "DatasetSummary",
    "RAGResponse",
    "RetrievalResult",
    "FAQMatch",
    "QueryRewrite",
    "MetricResult",
    "MetricCategory",
    "EvaluationResult",
    "EvaluationRun",
]