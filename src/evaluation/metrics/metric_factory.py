"""
Metric factory for creating metric instances.
Implements Factory pattern for metric instantiation.
"""

from typing import Any, Optional

from .base import BaseMetric
from .metric_registry import get_registry
from ...core.exceptions import ConfigurationError
from ...core.logging import logger


class MetricFactory:
    """
    Factory for creating metric instances.
    Implements the Factory pattern.
    """

    @staticmethod
    def create(
        metric_name: str,
        config: Optional[dict[str, Any]] = None,
    ) -> BaseMetric:
        """
        Create a metric instance by name.

        Args:
            metric_name: Name of the metric to create
            config: Optional configuration for the metric

        Returns:
            Metric instance

        Raises:
            ConfigurationError: If metric not found
        """
        registry = get_registry()
        metric_class = registry.get(metric_name)

        if metric_class is None:
            available = registry.list_all()
            raise ConfigurationError(
                f"Unknown metric: {metric_name}. "
                f"Available metrics: {available}"
            )

        instance = metric_class(config=config)
        logger.debug(f"Created metric instance: {metric_name}")

        return instance

    @staticmethod
    def create_all(
        metric_names: Optional[list[str]] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> list[BaseMetric]:
        """
        Create multiple metric instances.

        Args:
            metric_names: List of metric names to create.
                         If None, creates all registered metrics.
            config: Optional configuration for all metrics

        Returns:
            List of metric instances
        """
        registry = get_registry()

        if metric_names is None:
            metric_names = registry.list_all()

        metrics = []
        for name in metric_names:
            try:
                metric = MetricFactory.create(name, config)
                metrics.append(metric)
            except ConfigurationError as e:
                logger.warning(f"Skipping metric {name}: {e}")

        return metrics

    @staticmethod
    def create_by_category(
        category: str,
        config: Optional[dict[str, Any]] = None,
    ) -> list[BaseMetric]:
        """
        Create all metrics in a category.

        Args:
            category: Category name (retrieval, generation, faq, comprehensive)
            config: Optional configuration

        Returns:
            List of metric instances in the category
        """
        from ...models.metric_result import MetricCategory

        try:
            cat = MetricCategory(category)
        except ValueError:
            raise ConfigurationError(
                f"Unknown category: {category}. "
                f"Valid categories: {[c.value for c in MetricCategory]}"
            )

        registry = get_registry()
        metric_names = registry.list_by_category(cat)

        return MetricFactory.create_all(metric_names, config)