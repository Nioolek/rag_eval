"""
Annotation data model with extensible fields.
Supports dynamic field addition for future requirements.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class Language(str, Enum):
    """Supported languages."""
    ZH = "zh"
    EN = "en"
    AUTO = "auto"


class AnnotationField(BaseModel):
    """
    Extensible annotation field.
    Supports custom fields with type validation.
    """
    name: str
    value: Any
    field_type: str = "text"  # text, number, boolean, list, json
    required: bool = False
    description: str = ""

    model_config = {"extra": "allow"}


class Annotation(BaseModel):
    """
    Annotation data model with RAG input and labeling fields.
    Designed for high extensibility - new fields can be added without schema changes.
    """
    # Unique identifier
    id: str = Field(default_factory=lambda: str(uuid4()))

    # RAG Input Fields
    query: str = Field(..., description="User query")
    conversation_history: list[str] = Field(
        default_factory=list,
        description="Multi-turn conversation history"
    )
    agent_id: str = Field(default="default", description="Agent identifier")
    dataset_id: str = Field(default="default", description="Dataset this annotation belongs to")
    language: Language = Field(default=Language.AUTO, description="Query language")
    enable_thinking: bool = Field(default=False, description="Enable thinking mode")

    # Annotation Result Fields
    gt_documents: list[str] = Field(
        default_factory=list,
        description="Ground truth documents"
    )
    faq_matched: bool = Field(default=False, description="Whether FAQ is matched")
    should_refuse: bool = Field(default=False, description="Whether should refuse to answer")
    standard_answers: list[str] = Field(
        default_factory=list,
        description="Multiple standard answers"
    )
    answer_style: str = Field(default="", description="Required answer style")
    notes: str = Field(default="", description="Additional notes")

    # Custom extensible fields
    custom_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom extension fields"
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    version: int = Field(default=1)
    is_deleted: bool = Field(default=False)

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

    def update(self, **kwargs: Any) -> "Annotation":
        """Update annotation fields and increment version."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now()
        self.version += 1
        return self

    def add_custom_field(self, name: str, value: Any) -> None:
        """Add a custom field."""
        self.custom_fields[name] = value
        self.updated_at = datetime.now()
        self.version += 1

    def soft_delete(self) -> None:
        """Soft delete the annotation."""
        self.is_deleted = True
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "query": self.query,
            "conversation_history": self.conversation_history,
            "agent_id": self.agent_id,
            "dataset_id": self.dataset_id,
            "language": self.language.value,
            "enable_thinking": self.enable_thinking,
            "gt_documents": self.gt_documents,
            "faq_matched": self.faq_matched,
            "should_refuse": self.should_refuse,
            "standard_answers": self.standard_answers,
            "answer_style": self.answer_style,
            "notes": self.notes,
            "custom_fields": self.custom_fields,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "is_deleted": self.is_deleted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Annotation":
        """Create annotation from dictionary."""
        # Handle datetime conversion
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        if isinstance(data.get("language"), str):
            data["language"] = Language(data["language"])
        return cls(**data)


class AnnotationList(BaseModel):
    """List of annotations with pagination support."""
    items: list[Annotation] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 50

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Annotation:
        return self.items[index]