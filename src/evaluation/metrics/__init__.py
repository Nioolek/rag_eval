"""
Evaluation metrics module.
Implements Strategy pattern for extensible metric calculations.
"""

from .base import BaseMetric, MetricContext
from .metric_registry import MetricRegistry, get_registry
from .metric_factory import MetricFactory

# Retrieval metrics
from .retrieval import (
    RetrievalPrecisionMetric,
    RetrievalRecallMetric,
    MRRMetric,
    HitRateMetric,
    RetrievalRelevanceMetric,
)

# Generation quality metrics
from .generation import (
    FactualConsistencyMetric,
    AnswerRelevanceMetric,
    AnswerCompletenessMetric,
    AnswerFluencyMetric,
    RefusalAccuracyMetric,
    HallucinationDetectionMetric,
)

# FAQ metrics
from .faq import (
    FAQMatchAccuracyMetric,
    FAQRecallMetric,
    FAQAnswerConsistencyMetric,
)

# Comprehensive metrics
from .comprehensive import (
    MultiAnswerMatchMetric,
    StyleMatchMetric,
    ConversationConsistencyMetric,
    ContextUtilizationMetric,
    AnswerRepetitionMetric,
)

__all__ = [
    # Base
    "BaseMetric",
    "MetricContext",
    "MetricRegistry",
    "get_registry",
    "MetricFactory",
    # Retrieval
    "RetrievalPrecisionMetric",
    "RetrievalRecallMetric",
    "MRRMetric",
    "HitRateMetric",
    "RetrievalRelevanceMetric",
    # Generation
    "FactualConsistencyMetric",
    "AnswerRelevanceMetric",
    "AnswerCompletenessMetric",
    "AnswerFluencyMetric",
    "RefusalAccuracyMetric",
    "HallucinationDetectionMetric",
    # FAQ
    "FAQMatchAccuracyMetric",
    "FAQRecallMetric",
    "FAQAnswerConsistencyMetric",
    # Comprehensive
    "MultiAnswerMatchMetric",
    "StyleMatchMetric",
    "ConversationConsistencyMetric",
    "ContextUtilizationMetric",
    "AnswerRepetitionMetric",
]