"""
Annotation statistics calculator.
Provides visualization-ready statistics for annotation data.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from ..models.annotation import Annotation, Language
from ..storage.base import StorageBackend
from ..core.logging import logger


@dataclass
class AnnotationStats:
    """Statistics for annotation data."""
    # Basic counts
    total_count: int = 0
    active_count: int = 0
    deleted_count: int = 0

    # Language distribution
    language_distribution: dict[str, int] = field(default_factory=dict)

    # Agent distribution
    agent_distribution: dict[str, int] = field(default_factory=dict)

    # FAQ matching
    faq_matched_count: int = 0
    faq_matched_rate: float = 0.0

    # Refusal statistics
    should_refuse_count: int = 0
    refusal_rate: float = 0.0

    # Conversation statistics
    multi_turn_count: int = 0
    single_turn_count: int = 0
    avg_history_length: float = 0.0

    # Ground truth statistics
    with_gt_documents_count: int = 0
    avg_gt_documents: float = 0.0
    with_standard_answers_count: int = 0
    avg_standard_answers: float = 0.0

    # Time statistics
    created_today: int = 0
    created_this_week: int = 0
    created_this_month: int = 0

    # Custom field statistics
    custom_field_usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_count": self.total_count,
            "active_count": self.active_count,
            "deleted_count": self.deleted_count,
            "language_distribution": self.language_distribution,
            "agent_distribution": self.agent_distribution,
            "faq_matched_count": self.faq_matched_count,
            "faq_matched_rate": self.faq_matched_rate,
            "should_refuse_count": self.should_refuse_count,
            "refusal_rate": self.refusal_rate,
            "multi_turn_count": self.multi_turn_count,
            "single_turn_count": self.single_turn_count,
            "avg_history_length": self.avg_history_length,
            "with_gt_documents_count": self.with_gt_documents_count,
            "avg_gt_documents": self.avg_gt_documents,
            "with_standard_answers_count": self.with_standard_answers_count,
            "avg_standard_answers": self.avg_standard_answers,
            "created_today": self.created_today,
            "created_this_week": self.created_this_week,
            "created_this_month": self.created_this_month,
            "custom_field_usage": self.custom_field_usage,
        }


class AnnotationStatistics:
    """
    Calculator for annotation statistics.
    Provides caching for frequently accessed stats.
    """

    def __init__(self, storage: StorageBackend, cache_ttl_seconds: int = 60):
        self.storage = storage
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Optional[AnnotationStats] = None
        self._cache_time: Optional[datetime] = None

    async def calculate(self, force_refresh: bool = False) -> AnnotationStats:
        """
        Calculate annotation statistics.

        Args:
            force_refresh: Force cache refresh

        Returns:
            AnnotationStats object
        """
        # Check cache
        if not force_refresh and self._cache and self._cache_time:
            elapsed = (datetime.now() - self._cache_time).total_seconds()
            if elapsed < self.cache_ttl_seconds:
                logger.debug("Returning cached statistics")
                return self._cache

        logger.info("Calculating annotation statistics")
        stats = AnnotationStats()

        # Initialize counters
        language_counter: Counter = Counter()
        agent_counter: Counter = Counter()
        history_lengths: list[int] = []
        gt_doc_counts: list[int] = []
        std_answer_counts: list[int] = []
        custom_fields: Counter = Counter()

        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Iterate through all annotations
        async for data in self.storage.iterate("annotations"):
            stats.total_count += 1

            if data.get("is_deleted"):
                stats.deleted_count += 1
                continue

            stats.active_count += 1

            # Language distribution
            lang = data.get("language", "auto")
            language_counter[lang] += 1

            # Agent distribution
            agent = data.get("agent_id", "default")
            agent_counter[agent] += 1

            # FAQ matching
            if data.get("faq_matched"):
                stats.faq_matched_count += 1

            # Refusal
            if data.get("should_refuse"):
                stats.should_refuse_count += 1

            # Conversation history
            history = data.get("conversation_history", [])
            history_lengths.append(len(history))
            if len(history) > 0:
                stats.multi_turn_count += 1
            else:
                stats.single_turn_count += 1

            # Ground truth documents
            gt_docs = data.get("gt_documents", [])
            if gt_docs:
                stats.with_gt_documents_count += 1
                gt_doc_counts.append(len(gt_docs))

            # Standard answers
            std_answers = data.get("standard_answers", [])
            if std_answers:
                stats.with_standard_answers_count += 1
                std_answer_counts.append(len(std_answers))

            # Custom fields
            custom = data.get("custom_fields", {})
            for key in custom.keys():
                custom_fields[key] += 1

            # Time-based statistics
            created_at_str = data.get("created_at")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                    if created_at >= today_start:
                        stats.created_today += 1
                    if (now - created_at).days < 7:
                        stats.created_this_week += 1
                    if (now - created_at).days < 30:
                        stats.created_this_month += 1
                except ValueError:
                    pass

        # Calculate rates and averages
        if stats.active_count > 0:
            stats.faq_matched_rate = stats.faq_matched_count / stats.active_count
            stats.refusal_rate = stats.should_refuse_count / stats.active_count

        if history_lengths:
            stats.avg_history_length = sum(history_lengths) / len(history_lengths)

        if gt_doc_counts:
            stats.avg_gt_documents = sum(gt_doc_counts) / len(gt_doc_counts)

        if std_answer_counts:
            stats.avg_standard_answers = sum(std_answer_counts) / len(std_answer_counts)

        # Assign counters
        stats.language_distribution = dict(language_counter)
        stats.agent_distribution = dict(agent_counter)
        stats.custom_field_usage = dict(custom_fields)

        # Update cache
        self._cache = stats
        self._cache_time = datetime.now()

        logger.info(f"Statistics calculated: {stats.active_count} active annotations")
        return stats

    async def get_distribution_by_field(self, field_name: str) -> dict[str, int]:
        """
        Get distribution of a specific field.

        Args:
            field_name: Field name to analyze

        Returns:
            Dictionary mapping field values to counts
        """
        counter: Counter = Counter()

        async for data in self.storage.iterate("annotations"):
            if data.get("is_deleted"):
                continue

            value = data.get(field_name)
            if value is not None:
                key = str(value)
                counter[key] += 1

        return dict(counter)

    async def get_time_series(
        self,
        days: int = 30,
        granularity: str = "day"
    ) -> list[dict[str, Any]]:
        """
        Get time series of annotation creation.

        Args:
            days: Number of days to include
            granularity: "day" or "hour"

        Returns:
            List of time-series data points
        """
        from datetime import timedelta

        now = datetime.now()
        start = now - timedelta(days=days)

        counter: Counter = Counter()

        async for data in self.storage.iterate("annotations"):
            created_at_str = data.get("created_at")
            if not created_at_str:
                continue

            try:
                created_at = datetime.fromisoformat(created_at_str)
                if created_at < start:
                    continue

                if granularity == "day":
                    key = created_at.strftime("%Y-%m-%d")
                else:
                    key = created_at.strftime("%Y-%m-%d %H:00")

                counter[key] += 1
            except ValueError:
                continue

        # Sort by key
        result = [
            {"date": k, "count": v}
            for k, v in sorted(counter.items())
        ]

        return result

    def clear_cache(self) -> None:
        """Clear the statistics cache."""
        self._cache = None
        self._cache_time = None


# Singleton statistics instance
_stats_instance: Optional[AnnotationStatistics] = None


async def get_statistics() -> AnnotationStatistics:
    """Get singleton statistics instance."""
    from ..storage.storage_factory import get_storage

    global _stats_instance

    if _stats_instance is None:
        storage = await get_storage()
        _stats_instance = AnnotationStatistics(storage)

    return _stats_instance