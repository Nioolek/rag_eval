"""
Dataset handler for managing annotation datasets.
Implements CRUD operations, import/export, and statistics.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import aiofiles

from ..models.dataset import Dataset, DatasetList, DatasetStatus, DatasetSummary
from ..models.annotation import Annotation
from ..storage.base import StorageBackend
from ..storage.storage_factory import get_storage
from ..core.exceptions import DatasetError
from ..core.logging import logger


class DatasetHandler:
    """
    Handler for dataset management operations.
    Follows the same patterns as AnnotationHandler.
    """

    COLLECTION = "datasets"

    def __init__(self, storage: StorageBackend):
        self.storage = storage

    async def create(self, dataset: Dataset) -> str:
        """
        Create a new dataset.

        Args:
            dataset: Dataset data to create

        Returns:
            Created dataset ID
        """
        try:
            # Validate dataset
            dataset.id  # Ensure ID exists

            # Check for duplicate name
            existing = await self.get_by_name(dataset.name)
            if existing and not existing.is_deleted:
                raise DatasetError(f"Dataset with name '{dataset.name}' already exists")

            # If this is set as default, unset other defaults
            if dataset.is_default:
                await self._unset_default_datasets()

            # Save to storage
            data = dataset.to_dict()
            await self.storage.save(self.COLLECTION, data)

            logger.info(f"Created dataset {dataset.id}: {dataset.name}")
            return dataset.id

        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            if isinstance(e, DatasetError):
                raise
            raise DatasetError(f"Failed to create dataset: {e}")

    async def get(self, dataset_id: str) -> Optional[Dataset]:
        """
        Get a dataset by ID.

        Args:
            dataset_id: Dataset ID

        Returns:
            Dataset or None if not found
        """
        data = await self.storage.get(self.COLLECTION, dataset_id)
        if data:
            return Dataset.from_dict(data)
        return None

    async def get_by_name(self, name: str) -> Optional[Dataset]:
        """
        Get a dataset by name.

        Args:
            name: Dataset name

        Returns:
            Dataset or None if not found
        """
        datasets = await self.storage.get_all(
            self.COLLECTION,
            filters={"name": name},
            limit=1,
        )
        if datasets:
            return Dataset.from_dict(datasets[0])
        return None

    async def get_default(self) -> Dataset:
        """
        Get the default dataset. Creates one if none exists.

        Returns:
            Default dataset
        """
        datasets = await self.storage.get_all(
            self.COLLECTION,
            filters={"is_default": True, "is_deleted": False},
            limit=1,
        )
        if datasets:
            return Dataset.from_dict(datasets[0])

        # If no default exists, create one
        default_dataset = Dataset(
            name="Default Dataset",
            description="Default dataset for annotations",
            is_default=True,
            status=DatasetStatus.ACTIVE,
        )
        await self.create(default_dataset)
        return default_dataset

    async def update(self, dataset_id: str, data: dict[str, Any]) -> Dataset:
        """
        Update a dataset.

        Args:
            dataset_id: Dataset ID
            data: Fields to update

        Returns:
            Updated dataset
        """
        dataset = await self.get(dataset_id)
        if not dataset:
            raise DatasetError(f"Dataset not found: {dataset_id}")

        # Handle is_default change
        if data.get("is_default") and not dataset.is_default:
            await self._unset_default_datasets()

        # Handle name change - check for duplicates
        new_name = data.get("name")
        if new_name and new_name != dataset.name:
            existing = await self.get_by_name(new_name)
            if existing and existing.id != dataset_id and not existing.is_deleted:
                raise DatasetError(f"Dataset with name '{new_name}' already exists")

        # Update fields
        for key, value in data.items():
            if hasattr(dataset, key):
                setattr(dataset, key, value)
        dataset.updated_at = datetime.now()

        # Save to storage
        await self.storage.update(self.COLLECTION, dataset_id, dataset.to_dict())

        logger.info(f"Updated dataset {dataset_id}")
        return dataset

    async def delete(self, dataset_id: str, soft_delete: bool = True) -> bool:
        """
        Delete a dataset.

        Args:
            dataset_id: Dataset ID
            soft_delete: Whether to soft delete (default True)

        Returns:
            True if deleted successfully
        """
        dataset = await self.get(dataset_id)
        if not dataset:
            raise DatasetError(f"Dataset not found: {dataset_id}")

        if dataset.is_default:
            raise DatasetError("Cannot delete the default dataset")

        if soft_delete:
            await self.storage.update(
                self.COLLECTION,
                dataset_id,
                {"is_deleted": True, "updated_at": datetime.now().isoformat()}
            )
        else:
            await self.storage.delete(self.COLLECTION, dataset_id)

        logger.info(f"Deleted dataset {dataset_id}")
        return True

    async def list(
        self,
        page: int = 1,
        page_size: int = 50,
        filters: Optional[dict[str, Any]] = None,
        include_deleted: bool = False,
    ) -> DatasetList:
        """
        List datasets with pagination.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page
            filters: Optional filters
            include_deleted: Whether to include deleted datasets

        Returns:
            DatasetList with items and pagination info
        """
        # Add is_deleted filter if not including deleted
        combined_filters = filters or {}
        if not include_deleted:
            combined_filters["is_deleted"] = False

        offset = (page - 1) * page_size

        items_data = await self.storage.get_all(
            self.COLLECTION,
            filters=combined_filters,
            limit=page_size,
            offset=offset,
        )

        total = await self.storage.count(self.COLLECTION, filters=combined_filters)

        items = [Dataset.from_dict(d) for d in items_data]

        return DatasetList(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
        )

    async def set_active(self, dataset_id: str) -> bool:
        """
        Set a dataset as the active (default) dataset.

        Args:
            dataset_id: Dataset ID

        Returns:
            True if successful
        """
        dataset = await self.get(dataset_id)
        if not dataset:
            raise DatasetError(f"Dataset not found: {dataset_id}")

        await self._unset_default_datasets()
        await self.update(dataset_id, {"is_default": True})

        logger.info(f"Set dataset {dataset_id} as active")
        return True

    async def _unset_default_datasets(self) -> None:
        """Unset is_default on all datasets."""
        datasets = await self.storage.get_all(
            self.COLLECTION,
            filters={"is_default": True},
            limit=1000,
        )
        for d in datasets:
            await self.storage.update(
                self.COLLECTION,
                d["id"],
                {"is_default": False}
            )

    async def update_annotation_count(self, dataset_id: str) -> None:
        """
        Update the cached annotation count for a dataset.

        Args:
            dataset_id: Dataset ID
        """
        count = await self._count_annotations(dataset_id)
        last_annotation = await self._get_last_annotation_time(dataset_id)

        await self.storage.update(
            self.COLLECTION,
            dataset_id,
            {
                "annotation_count": count,
                "last_annotation_at": last_annotation.isoformat() if last_annotation else None,
                "updated_at": datetime.now().isoformat(),
            }
        )

    async def _count_annotations(self, dataset_id: str) -> int:
        """Count annotations in a dataset."""
        return await self.storage.count(
            "annotations",
            filters={"dataset_id": dataset_id}
        )

    async def _get_last_annotation_time(self, dataset_id: str) -> Optional[datetime]:
        """Get the most recent annotation timestamp for a dataset."""
        annotations = await self.storage.query_with_sort(
            "annotations",
            filters={"dataset_id": dataset_id},
            sort_by="created_at",
            sort_desc=True,
            limit=1,
        )
        if annotations:
            created_at = annotations[0].get("created_at")
            if created_at:
                if isinstance(created_at, str):
                    return datetime.fromisoformat(created_at)
                return created_at
        return None

    # ===== Import/Export =====

    async def export_dataset(
        self,
        dataset_id: str,
        output_path: Optional[Path] = None,
        include_annotations: bool = True,
    ) -> Path:
        """
        Export a dataset to a JSON file.

        Args:
            dataset_id: Dataset ID to export
            output_path: Output file path (optional)
            include_annotations: Whether to include annotations

        Returns:
            Path to exported file
        """
        dataset = await self.get(dataset_id)
        if not dataset:
            raise DatasetError(f"Dataset not found: {dataset_id}")

        export_data = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "dataset": dataset.to_dict(),
        }

        if include_annotations:
            annotations = await self.storage.get_all(
                "annotations",
                filters={"dataset_id": dataset_id},
                limit=10000,
            )
            # Filter out deleted annotations
            active_annotations = [a for a in annotations if not a.get("is_deleted")]
            export_data["annotations"] = active_annotations

        if output_path is None:
            safe_name = dataset.name.replace(" ", "_").replace("/", "_")
            output_path = Path(f"dataset_{safe_name}_{dataset_id[:8]}.json")

        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(export_data, ensure_ascii=False, indent=2))

        logger.info(f"Exported dataset {dataset_id} to {output_path}")
        return output_path

    async def import_dataset(
        self,
        file_path: Path,
        name: Optional[str] = None,
        merge: bool = False,
    ) -> tuple[Dataset, int]:
        """
        Import a dataset from a JSON file.

        Args:
            file_path: Path to import file
            name: Optional new name for imported dataset
            merge: If True, merge into existing dataset with same name

        Returns:
            Tuple of (imported Dataset, number of annotations imported)
        """
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            import_data = json.loads(content)

        # Validate format
        if import_data.get("version") != "1.0":
            raise DatasetError("Unsupported import file version. Expected version 1.0")

        dataset_data = import_data.get("dataset")
        if not dataset_data:
            raise DatasetError("Invalid import file: missing dataset data")

        # Handle name
        if name:
            dataset_data["name"] = name

        # Check for existing dataset
        existing = await self.get_by_name(dataset_data["name"])
        dataset: Dataset

        if merge and existing and not existing.is_deleted:
            dataset = existing
        else:
            # Create new dataset
            dataset_data["id"] = str(uuid4())  # Generate new ID
            dataset_data["is_default"] = False  # Never import as default
            dataset_data["created_at"] = datetime.now().isoformat()
            dataset_data["updated_at"] = datetime.now().isoformat()
            dataset = Dataset.from_dict(dataset_data)
            await self.create(dataset)

        # Import annotations
        annotations_data = import_data.get("annotations", [])
        imported_count = 0

        if annotations_data:
            for ann_data in annotations_data:
                ann_data["id"] = str(uuid4())  # Generate new ID
                ann_data["dataset_id"] = dataset.id
                ann_data["created_at"] = ann_data.get("created_at") or datetime.now().isoformat()
                ann_data["updated_at"] = datetime.now().isoformat()
                annotation = Annotation.from_dict(ann_data)
                await self.storage.save("annotations", annotation.to_dict())
                imported_count += 1

        # Update count
        await self.update_annotation_count(dataset.id)

        logger.info(f"Imported dataset {dataset.id} with {imported_count} annotations")
        return dataset, imported_count

    async def create_version(
        self,
        dataset_id: str,
        version_note: str = "",
    ) -> Dataset:
        """
        Create a new version of a dataset.

        Args:
            dataset_id: Dataset ID to version
            version_note: Note describing this version

        Returns:
            New dataset version
        """
        dataset = await self.get(dataset_id)
        if not dataset:
            raise DatasetError(f"Dataset not found: {dataset_id}")

        # Create new dataset as a version
        new_version = Dataset(
            name=f"{dataset.name} (v{dataset.version + 1})",
            description=dataset.description,
            version=dataset.version + 1,
            status=DatasetStatus.DRAFT,
            tags=dataset.tags.copy(),
            metadata=dataset.metadata.copy(),
            parent_id=dataset.id,
            version_note=version_note,
            created_by=dataset.created_by,
        )

        await self.create(new_version)

        # Copy annotations
        annotations = await self.storage.get_all(
            "annotations",
            filters={"dataset_id": dataset_id},
            limit=10000,
        )

        for ann_data in annotations:
            if ann_data.get("is_deleted"):
                continue
            new_ann_data = ann_data.copy()
            new_ann_data["id"] = str(uuid4())  # New ID
            new_ann_data["dataset_id"] = new_version.id
            new_ann_data["created_at"] = datetime.now().isoformat()
            new_ann_data["updated_at"] = datetime.now().isoformat()
            annotation = Annotation.from_dict(new_ann_data)
            await self.storage.save("annotations", annotation.to_dict())

        await self.update_annotation_count(new_version.id)

        logger.info(f"Created version {new_version.version} of dataset {dataset_id}")
        return new_version

    async def get_statistics(self, dataset_id: str) -> DatasetSummary:
        """
        Get statistics for a dataset.

        Args:
            dataset_id: Dataset ID

        Returns:
            DatasetSummary with statistics
        """
        dataset = await self.get(dataset_id)
        if not dataset:
            raise DatasetError(f"Dataset not found: {dataset_id}")

        summary = DatasetSummary(
            dataset_id=dataset_id,
            dataset_name=dataset.name,
        )

        language_dist: dict[str, int] = {}
        agent_dist: dict[str, int] = {}

        annotations = await self.storage.get_all(
            "annotations",
            filters={"dataset_id": dataset_id},
            limit=10000,
        )

        for ann_data in annotations:
            if ann_data.get("is_deleted"):
                continue

            summary.total_annotations += 1
            summary.active_annotations += 1

            if ann_data.get("gt_documents"):
                summary.with_gt_documents += 1
            if ann_data.get("standard_answers"):
                summary.with_standard_answers += 1
            if ann_data.get("faq_matched"):
                summary.faq_matched += 1
            if ann_data.get("should_refuse"):
                summary.should_refuse += 1
            if ann_data.get("conversation_history"):
                summary.multi_turn += 1

            lang = ann_data.get("language", "auto")
            language_dist[lang] = language_dist.get(lang, 0) + 1

            agent = ann_data.get("agent_id", "default")
            agent_dist[agent] = agent_dist.get(agent, 0) + 1

        summary.language_distribution = language_dist
        summary.agent_distribution = agent_dist

        return summary

    async def get_choices_for_ui(self) -> list[tuple[str, str]]:
        """
        Get dataset choices for UI dropdown.

        Returns:
            List of (display_name, dataset_id) tuples
        """
        datasets = await self.list(page=1, page_size=100)

        choices = []
        for d in datasets.items:
            display = f"{d.name} ({d.annotation_count} 条标注)"
            if d.is_default:
                display = f"[默认] {display}"
            choices.append((display, d.id))

        return choices


# Singleton handler instance
_handler_instance: Optional[DatasetHandler] = None
_handler_lock: Optional[asyncio.Lock] = None


def _get_handler_lock() -> asyncio.Lock:
    """Get or create the handler lock (lazy initialization)."""
    global _handler_lock
    if _handler_lock is None:
        _handler_lock = asyncio.Lock()
    return _handler_lock


async def get_dataset_handler() -> DatasetHandler:
    """
    Get singleton dataset handler instance.
    Thread-safe for concurrent access.
    """
    global _handler_instance

    if _handler_instance is not None:
        return _handler_instance

    # Use lock for thread-safe singleton initialization
    async with _get_handler_lock():
        # Double-check after acquiring lock
        if _handler_instance is not None:
            return _handler_instance

        storage = await get_storage()
        _handler_instance = DatasetHandler(storage)

    return _handler_instance