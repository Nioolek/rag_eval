"""Models module: data models for the RAG evaluation system."""

from .annotation import Annotation, AnnotationField, AnnotationList
from .rag_response import RAGResponse, RetrievalResult, FAQMatch, QueryRewrite
from .metric_result import MetricResult, MetricCategory
from .evaluation_result import EvaluationResult, EvaluationRun

__all__ = [
    "Annotation",
    "AnnotationField",
    "AnnotationList",
    "RAGResponse",
    "RetrievalResult",
    "FAQMatch",
    "QueryRewrite",
    "MetricResult",
    "MetricCategory",
    "EvaluationResult",
    "EvaluationRun",
]