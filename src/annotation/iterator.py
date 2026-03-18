"""
Annotation iterator with lazy loading.
Implements Iterator pattern for memory-efficient traversal.
"""

from typing import Any, AsyncIterator, Optional

from ..models.annotation import Annotation
from ..storage.base import StorageBackend
from ..core.logging import logger


class AnnotationIterator:
    """
    Iterator for annotation data with lazy loading.
    Implements the Iterator pattern for efficient traversal.
    """

    def __init__(
        self,
        storage: StorageBackend,
        batch_size: int = 100,
        filters: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize iterator.

        Args:
            storage: Storage backend
            batch_size: Number of items to load per batch
            filters: Optional filters to apply
        """
        self.storage = storage
        self.batch_size = batch_size
        self.filters = filters
        self._current_batch: list[Annotation] = []
        self._batch_index: int = 0
        self._total_yielded: int = 0
        self._exhausted: bool = False

    def __aiter__(self) -> "AnnotationIterator":
        """Return async iterator."""
        return self

    async def __anext__(self) -> Annotation:
        """Get next annotation."""
        if self._exhausted:
            raise StopAsyncIteration

        # Load next batch if current is exhausted
        if self._batch_index >= len(self._current_batch):
            await self._load_next_batch()

            # Check if exhausted after loading
            if not self._current_batch:
                self._exhausted = True
                raise StopAsyncIteration

        # Get current item
        item = self._current_batch[self._batch_index]
        self._batch_index += 1
        self._total_yielded += 1

        return item

    async def _load_next_batch(self) -> None:
        """Load the next batch of annotations."""
        offset = self._total_yielded

        items_data = await self.storage.get_all(
            "annotations",
            filters=self.filters,
            limit=self.batch_size,
            offset=offset,
        )

        self._current_batch = [Annotation.from_dict(d) for d in items_data]
        self._batch_index = 0

        logger.debug(
            f"Loaded batch: {len(self._current_batch)} items, "
            f"offset: {offset}, total yielded: {self._total_yielded}"
        )

    async def to_list(self, limit: Optional[int] = None) -> list[Annotation]:
        """
        Convert iterator to list.

        Args:
            limit: Maximum number of items to return

        Returns:
            List of annotations
        """
        result = []
        count = 0

        async for annotation in self:
            result.append(annotation)
            count += 1
            if limit and count >= limit:
                break

        return result

    async def count(self) -> int:
        """Get total count of matching annotations."""
        return await self.storage.count("annotations", filters=self.filters)

    def reset(self) -> None:
        """Reset the iterator to start from beginning."""
        self._current_batch = []
        self._batch_index = 0
        self._total_yielded = 0
        self._exhausted = False


class FilteredAnnotationIterator(AnnotationIterator):
    """
    Filtered iterator with custom filter function.
    """

    def __init__(
        self,
        storage: StorageBackend,
        filter_func: callable,
        batch_size: int = 100,
        base_filters: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize filtered iterator.

        Args:
            storage: Storage backend
            filter_func: Custom filter function (annotation -> bool)
            batch_size: Batch size
            base_filters: Base filters to apply
        """
        super().__init__(storage, batch_size, base_filters)
        self.filter_func = filter_func

    async def __anext__(self) -> Annotation:
        """Get next annotation that passes the filter."""
        while True:
            annotation = await super().__anext__()
            if self.filter_func(annotation):
                return annotation


class BatchAnnotationIterator(AnnotationIterator):
    """
    Iterator that yields batches instead of individual items.
    """

    async def __anext__(self) -> list[Annotation]:
        """Get next batch of annotations."""
        if self._exhausted:
            raise StopAsyncIteration

        await self._load_next_batch()

        if not self._current_batch:
            self._exhausted = True
            raise StopAsyncIteration

        batch = self._current_batch.copy()
        self._total_yielded += len(batch)

        return batch