"""
Annotation handler with CRUD operations.
Implements Template Method pattern for standardized operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from ..models.annotation import Annotation, AnnotationList
from ..storage.base import StorageBackend
from ..storage.storage_factory import get_storage
from ..core.exceptions import AnnotationError
from ..core.logging import logger


class BaseAnnotationHandler(ABC):
    """
    Abstract base class for annotation handlers.
    Implements Template Method pattern.
    """

    @abstractmethod
    async def create(self, annotation: Annotation) -> str:
        """Create a new annotation."""
        pass

    @abstractmethod
    async def get(self, annotation_id: str) -> Optional[Annotation]:
        """Get an annotation by ID."""
        pass

    @abstractmethod
    async def update(self, annotation_id: str, data: dict[str, Any]) -> Annotation:
        """Update an annotation."""
        pass

    @abstractmethod
    async def delete(self, annotation_id: str) -> bool:
        """Delete an annotation."""
        pass

    @abstractmethod
    async def list(
        self,
        page: int = 1,
        page_size: int = 50,
        filters: Optional[dict[str, Any]] = None,
    ) -> AnnotationList:
        """List annotations with pagination."""
        pass


class AnnotationHandler(BaseAnnotationHandler):
    """
    Main annotation handler with full CRUD support.
    Supports versioning and extensible fields.
    """

    COLLECTION = "annotations"

    def __init__(self, storage: StorageBackend):
        self.storage = storage

    async def create(self, annotation: Annotation) -> str:
        """
        Create a new annotation.

        Args:
            annotation: Annotation data to create

        Returns:
            Created annotation ID
        """
        try:
            # Validate annotation
            annotation.id  # Ensure ID exists

            # Save to storage
            data = annotation.to_dict()
            await self.storage.save(self.COLLECTION, data)

            # Save initial version
            await self.storage.save_version(
                self.COLLECTION,
                annotation.id,
                {"data": data, "action": "create"}
            )

            logger.info(f"Created annotation {annotation.id}")
            return annotation.id

        except Exception as e:
            logger.error(f"Failed to create annotation: {e}")
            raise AnnotationError(f"Failed to create annotation: {e}")

    async def get(self, annotation_id: str) -> Optional[Annotation]:
        """
        Get an annotation by ID.

        Args:
            annotation_id: Annotation ID

        Returns:
            Annotation or None if not found
        """
        data = await self.storage.get(self.COLLECTION, annotation_id)
        if data:
            return Annotation.from_dict(data)
        return None

    async def update(
        self,
        annotation_id: str,
        data: dict[str, Any],
        create_version: bool = True
    ) -> Annotation:
        """
        Update an annotation.

        Args:
            annotation_id: Annotation ID
            data: Fields to update
            create_version: Whether to create a version snapshot

        Returns:
            Updated annotation
        """
        # Get existing annotation
        annotation = await self.get(annotation_id)
        if not annotation:
            raise AnnotationError(f"Annotation not found: {annotation_id}")

        # Update fields
        old_data = annotation.to_dict()
        for key, value in data.items():
            if hasattr(annotation, key):
                setattr(annotation, key, value)
        annotation.updated_at = datetime.now()
        annotation.version += 1

        # Save to storage
        await self.storage.update(
            self.COLLECTION,
            annotation_id,
            annotation.to_dict()
        )

        # Save version
        if create_version:
            await self.storage.save_version(
                self.COLLECTION,
                annotation_id,
                {
                    "previous_data": old_data,
                    "new_data": annotation.to_dict(),
                    "action": "update",
                }
            )

        logger.info(f"Updated annotation {annotation_id} to version {annotation.version}")
        return annotation

    async def delete(self, annotation_id: str, soft_delete: bool = True) -> bool:
        """
        Delete an annotation.

        Args:
            annotation_id: Annotation ID
            soft_delete: Whether to soft delete (default True)

        Returns:
            True if deleted successfully
        """
        annotation = await self.get(annotation_id)
        if not annotation:
            raise AnnotationError(f"Annotation not found: {annotation_id}")

        if soft_delete:
            await self.storage.update(
                self.COLLECTION,
                annotation_id,
                {"is_deleted": True, "updated_at": datetime.now().isoformat()}
            )
        else:
            # Hard delete - not recommended
            await self.storage.delete(self.COLLECTION, annotation_id)

        logger.info(f"Deleted annotation {annotation_id}")
        return True

    async def list(
        self,
        page: int = 1,
        page_size: int = 50,
        filters: Optional[dict[str, Any]] = None,
    ) -> AnnotationList:
        """
        List annotations with pagination.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page
            filters: Optional filters

        Returns:
            AnnotationList with items and pagination info
        """
        offset = (page - 1) * page_size

        items_data = await self.storage.get_all(
            self.COLLECTION,
            filters=filters,
            limit=page_size,
            offset=offset,
        )

        total = await self.storage.count(self.COLLECTION, filters=filters)

        items = [Annotation.from_dict(d) for d in items_data]

        return AnnotationList(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
        )

    async def batch_create(self, annotations: list[Annotation]) -> list[str]:
        """
        Create multiple annotations in batch.

        Args:
            annotations: List of annotations to create

        Returns:
            List of created annotation IDs
        """
        ids = []
        for annotation in annotations:
            aid = await self.create(annotation)
            ids.append(aid)
        return ids

    async def search(
        self,
        query: str,
        page: int = 1,
        page_size: int = 50,
    ) -> AnnotationList:
        """
        Search annotations by query text.

        Args:
            query: Search query
            page: Page number
            page_size: Items per page

        Returns:
            AnnotationList with matching items
        """
        # Simple text search in query field
        # For production, consider using full-text search
        all_items = await self.storage.get_all(
            self.COLLECTION,
            limit=10000,  # Limit for performance
        )

        matching = [
            item for item in all_items
            if query.lower() in item.get("query", "").lower()
        ]

        total = len(matching)
        offset = (page - 1) * page_size
        page_items = matching[offset:offset + page_size]

        items = [Annotation.from_dict(d) for d in page_items]

        return AnnotationList(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
        )

    async def get_versions(self, annotation_id: str) -> list[dict[str, Any]]:
        """
        Get version history of an annotation.

        Args:
            annotation_id: Annotation ID

        Returns:
            List of versions
        """
        return await self.storage.get_versions(self.COLLECTION, annotation_id)

    async def restore_version(
        self,
        annotation_id: str,
        version_number: int
    ) -> Annotation:
        """
        Restore annotation to a specific version.

        Args:
            annotation_id: Annotation ID
            version_number: Version number to restore

        Returns:
            Restored annotation
        """
        versions = await self.get_versions(annotation_id)
        target_version = next(
            (v for v in versions if v.get("version_number") == version_number),
            None
        )

        if not target_version:
            raise AnnotationError(f"Version {version_number} not found")

        # Get the data from the version
        version_data = target_version.get("new_data") or target_version.get("data")
        if not version_data:
            raise AnnotationError("Invalid version data")

        # Restore
        annotation = Annotation.from_dict(version_data)
        return await self.update(annotation_id, annotation.to_dict())


# Singleton handler instance
_handler_instance: Optional[AnnotationHandler] = None


async def get_annotation_handler() -> AnnotationHandler:
    """Get singleton annotation handler instance."""
    global _handler_instance

    if _handler_instance is None:
        storage = await get_storage()
        _handler_instance = AnnotationHandler(storage)

    return _handler_instance