"""
Dataset data model for managing collections of annotations.
Supports versioning, tagging, and metadata.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class DatasetStatus(str, Enum):
    """Dataset status enum."""
    DRAFT = "draft"        # Still being annotated
    ACTIVE = "active"      # Ready for evaluation
    ARCHIVED = "archived"  # No longer in use
    LOCKED = "locked"      # Read-only, cannot be modified


class Dataset(BaseModel):
    """
    Dataset model representing a collection of annotations.
    Supports versioning, tagging, and metadata.
    """
    # Unique identifier
    id: str = Field(default_factory=lambda: str(uuid4()))

    # Basic info
    name: str = Field(..., description="Dataset name", min_length=1, max_length=100)
    description: str = Field(default="", description="Dataset description")
    version: int = Field(default=1, description="Dataset version number")

    # Status
    status: DatasetStatus = Field(default=DatasetStatus.DRAFT)
    is_default: bool = Field(default=False, description="Whether this is the default dataset")

    # Statistics (cached)
    annotation_count: int = Field(default=0)
    last_annotation_at: Optional[datetime] = Field(default=None)

    # Tags and metadata
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Version management
    parent_id: Optional[str] = Field(default=None, description="Parent dataset ID for versioning")
    version_note: str = Field(default="", description="Note for this version")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: str = Field(default="", description="Creator identifier")

    # Soft delete
    is_deleted: bool = Field(default=False)

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate dataset name."""
        if not v or not v.strip():
            raise ValueError("Dataset name cannot be empty")
        return v.strip()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "status": self.status.value,
            "is_default": self.is_default,
            "annotation_count": self.annotation_count,
            "last_annotation_at": self.last_annotation_at.isoformat() if self.last_annotation_at else None,
            "tags": self.tags,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "version_note": self.version_note,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "is_deleted": self.is_deleted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Dataset":
        """Create dataset from dictionary."""
        # Handle datetime conversion
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        if isinstance(data.get("last_annotation_at"), str):
            data["last_annotation_at"] = datetime.fromisoformat(data["last_annotation_at"])
        # Handle enum conversion
        if isinstance(data.get("status"), str):
            data["status"] = DatasetStatus(data["status"])
        return cls(**data)


class DatasetList(BaseModel):
    """List of datasets with pagination support."""
    items: list[Dataset] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 50

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dataset:
        return self.items[index]


class DatasetSummary(BaseModel):
    """Summary statistics for a dataset."""
    dataset_id: str
    dataset_name: str
    total_annotations: int = 0
    active_annotations: int = 0
    with_gt_documents: int = 0
    with_standard_answers: int = 0
    faq_matched: int = 0
    should_refuse: int = 0
    multi_turn: int = 0
    language_distribution: dict[str, int] = Field(default_factory=dict)
    agent_distribution: dict[str, int] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "total_annotations": self.total_annotations,
            "active_annotations": self.active_annotations,
            "with_gt_documents": self.with_gt_documents,
            "with_standard_answers": self.with_standard_answers,
            "faq_matched": self.faq_matched,
            "should_refuse": self.should_refuse,
            "multi_turn": self.multi_turn,
            "language_distribution": self.language_distribution,
            "agent_distribution": self.agent_distribution,
        }