"""
Metric registry for managing available metrics.
Implements centralized metric registration and discovery.
"""

from typing import Any, Optional, Type

from .base import BaseMetric
from ...models.metric_result import MetricCategory
from ...core.exceptions import ConfigurationError
from ...core.logging import logger


class MetricRegistry:
    """
    Registry for all available metrics.
    Supports registration, discovery, and listing of metrics.
    """

    def __init__(self):
        self._metrics: dict[str, Type[BaseMetric]] = {}
        self._categories: dict[MetricCategory, list[str]] = {
            cat: [] for cat in MetricCategory
        }

    def register(self, metric_class: Type[BaseMetric]) -> None:
        """
        Register a metric class.

        Args:
            metric_class: Metric class to register
        """
        name = metric_class.name
        if name in self._metrics:
            logger.warning(f"Overriding existing metric: {name}")

        self._metrics[name] = metric_class
        self._categories[metric_class.category].append(name)

        logger.debug(f"Registered metric: {name} ({metric_class.category.value})")

    def unregister(self, name: str) -> bool:
        """
        Unregister a metric.

        Args:
            name: Metric name to unregister

        Returns:
            True if unregistered successfully
        """
        if name not in self._metrics:
            return False

        metric_class = self._metrics.pop(name)
        self._categories[metric_class.category].remove(name)
        return True

    def get(self, name: str) -> Optional[Type[BaseMetric]]:
        """
        Get a metric class by name.

        Args:
            name: Metric name

        Returns:
            Metric class or None if not found
        """
        return self._metrics.get(name)

    def list_all(self) -> list[str]:
        """Get list of all registered metric names."""
        return list(self._metrics.keys())

    def list_by_category(self, category: MetricCategory) -> list[str]:
        """
        Get metrics by category.

        Args:
            category: Metric category

        Returns:
            List of metric names in the category
        """
        return self._categories.get(category, [])

    def get_info(self, name: str) -> Optional[dict[str, Any]]:
        """
        Get metric information.

        Args:
            name: Metric name

        Returns:
            Metric info dict or None
        """
        metric_class = self.get(name)
        if metric_class:
            return metric_class.get_info()
        return None

    def get_all_info(self) -> dict[str, dict[str, Any]]:
        """Get information for all registered metrics."""
        return {
            name: cls.get_info()
            for name, cls in self._metrics.items()
        }

    def __contains__(self, name: str) -> bool:
        """Check if metric is registered."""
        return name in self._metrics

    def __len__(self) -> int:
        """Get number of registered metrics."""
        return len(self._metrics)


# Global registry instance
_registry: Optional[MetricRegistry] = None


def get_registry() -> MetricRegistry:
    """
    Get the global metric registry.
    Initializes with default metrics on first call.
    """
    global _registry

    if _registry is None:
        _registry = MetricRegistry()
        _register_default_metrics(_registry)

    return _registry


def _register_default_metrics(registry: MetricRegistry) -> None:
    """Register all default metrics."""
    # Import and register all metric classes
    # This is done here to avoid circular imports

    try:
        from .retrieval import (
            RetrievalPrecisionMetric,
            RetrievalRecallMetric,
            MRRMetric,
            HitRateMetric,
            RetrievalRelevanceMetric,
        )
        for cls in [
            RetrievalPrecisionMetric,
            RetrievalRecallMetric,
            MRRMetric,
            HitRateMetric,
            RetrievalRelevanceMetric,
        ]:
            registry.register(cls)
    except ImportError:
        pass

    try:
        from .generation import (
            FactualConsistencyMetric,
            AnswerRelevanceMetric,
            AnswerCompletenessMetric,
            AnswerFluencyMetric,
            RefusalAccuracyMetric,
            HallucinationDetectionMetric,
        )
        for cls in [
            FactualConsistencyMetric,
            AnswerRelevanceMetric,
            AnswerCompletenessMetric,
            AnswerFluencyMetric,
            RefusalAccuracyMetric,
            HallucinationDetectionMetric,
        ]:
            registry.register(cls)
    except ImportError:
        pass

    try:
        from .faq import (
            FAQMatchAccuracyMetric,
            FAQRecallMetric,
            FAQAnswerConsistencyMetric,
        )
        for cls in [
            FAQMatchAccuracyMetric,
            FAQRecallMetric,
            FAQAnswerConsistencyMetric,
        ]:
            registry.register(cls)
    except ImportError:
        pass

    try:
        from .comprehensive import (
            MultiAnswerMatchMetric,
            StyleMatchMetric,
            ConversationConsistencyMetric,
            ContextUtilizationMetric,
            AnswerRepetitionMetric,
        )
        for cls in [
            MultiAnswerMatchMetric,
            StyleMatchMetric,
            ConversationConsistencyMetric,
            ContextUtilizationMetric,
            AnswerRepetitionMetric,
        ]:
            registry.register(cls)
    except ImportError:
        pass

    # Register semantic similarity metrics
    try:
        from ..similarity.semantic_metrics import register_semantic_metrics
        register_semantic_metrics()
    except ImportError:
        pass

    # Register performance metrics
    try:
        from .performance import (
            StageLatencyMetric,
            TotalLatencyMetric,
            LatencyDistributionMetric,
            PerformanceComparisonMetric,
            StageEfficiencyMetric,
            ThroughputMetric,
        )
        for cls in [
            StageLatencyMetric,
            TotalLatencyMetric,
            LatencyDistributionMetric,
            PerformanceComparisonMetric,
            StageEfficiencyMetric,
            ThroughputMetric,
        ]:
            registry.register(cls)
    except ImportError:
        pass

    logger.info(f"Registered {len(registry)} default metrics")