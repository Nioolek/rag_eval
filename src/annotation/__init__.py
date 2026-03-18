"""
Annotation module: annotation management, statistics, and iteration.
"""

from .annotation_handler import AnnotationHandler, get_annotation_handler
from .iterator import AnnotationIterator
from .statistics import AnnotationStatistics, get_statistics

__all__ = [
    "AnnotationHandler",
    "get_annotation_handler",
    "AnnotationIterator",
    "AnnotationStatistics",
    "get_statistics",
]