"""Tests for annotation models and handler."""

import pytest
import pytest_asyncio
from datetime import datetime

from src.models.annotation import Annotation, AnnotationList, Language
from src.annotation.annotation_handler import AnnotationHandler
from src.storage.local_storage import LocalStorage
from pathlib import Path


class TestAnnotationModel:
    """Tests for Annotation model."""

    def test_annotation_creation(self, sample_annotation_data):
        """Test creating an annotation."""
        annotation = Annotation(**sample_annotation_data)

        assert annotation.query == sample_annotation_data["query"]
        assert annotation.language == Language.ZH
        assert len(annotation.conversation_history) == 2

    def test_annotation_validation(self):
        """Test annotation validation."""
        with pytest.raises(ValueError):
            Annotation(query="")  # Empty query should fail

    def test_annotation_update(self, sample_annotation_data):
        """Test updating annotation."""
        annotation = Annotation(**sample_annotation_data)
        original_version = annotation.version

        annotation.update(faq_matched=True)

        assert annotation.faq_matched is True
        assert annotation.version == original_version + 1

    def test_annotation_soft_delete(self, sample_annotation_data):
        """Test soft delete."""
        annotation = Annotation(**sample_annotation_data)

        annotation.soft_delete()

        assert annotation.is_deleted is True

    def test_annotation_custom_fields(self, sample_annotation_data):
        """Test custom fields."""
        annotation = Annotation(**sample_annotation_data)

        annotation.add_custom_field("domain", "finance")

        assert annotation.custom_fields["domain"] == "finance"

    def test_annotation_serialization(self, sample_annotation_data):
        """Test serialization and deserialization."""
        annotation = Annotation(**sample_annotation_data)

        data = annotation.to_dict()
        restored = Annotation.from_dict(data)

        assert restored.query == annotation.query
        assert restored.id == annotation.id


class TestAnnotationHandler:
    """Tests for AnnotationHandler."""

    @pytest_asyncio.fixture
    async def handler(self, temp_storage):
        """Create handler with temp storage."""
        return AnnotationHandler(temp_storage)

    @pytest.mark.asyncio
    async def test_create_annotation(self, handler, sample_annotation_data):
        """Test creating annotation."""
        annotation = Annotation(**sample_annotation_data)

        annotation_id = await handler.create(annotation)

        assert annotation_id is not None
        assert annotation_id == annotation.id

    @pytest.mark.asyncio
    async def test_get_annotation(self, handler, sample_annotation_data):
        """Test getting annotation."""
        annotation = Annotation(**sample_annotation_data)
        await handler.create(annotation)

        retrieved = await handler.get(annotation.id)

        assert retrieved is not None
        assert retrieved.query == annotation.query

    @pytest.mark.asyncio
    async def test_update_annotation(self, handler, sample_annotation_data):
        """Test updating annotation."""
        annotation = Annotation(**sample_annotation_data)
        await handler.create(annotation)

        updated = await handler.update(
            annotation.id,
            {"faq_matched": True, "notes": "Updated"}
        )

        assert updated.faq_matched is True
        assert updated.notes == "Updated"

    @pytest.mark.asyncio
    async def test_delete_annotation(self, handler, sample_annotation_data):
        """Test deleting annotation."""
        annotation = Annotation(**sample_annotation_data)
        await handler.create(annotation)

        await handler.delete(annotation.id)

        # Should be soft deleted
        result = await handler.get(annotation.id)
        # Note: get() should return None for deleted items

    @pytest.mark.asyncio
    async def test_list_annotations(self, handler):
        """Test listing annotations."""
        # Create multiple annotations
        for i in range(5):
            ann = Annotation(query=f"Test query {i}")
            await handler.create(ann)

        result = await handler.list(page=1, page_size=10)

        assert len(result.items) == 5
        assert result.total == 5

    @pytest.mark.asyncio
    async def test_search_annotations(self, handler):
        """Test searching annotations."""
        ann1 = Annotation(query="机器学习是什么")
        ann2 = Annotation(query="深度学习是什么")
        await handler.create(ann1)
        await handler.create(ann2)

        result = await handler.search("机器学习")

        assert len(result.items) >= 1
        assert any("机器学习" in ann.query for ann in result.items)


class TestAnnotationList:
    """Tests for AnnotationList."""

    def test_annotation_list(self):
        """Test annotation list operations."""
        items = [
            Annotation(query=f"Query {i}")
            for i in range(3)
        ]

        ann_list = AnnotationList(items=items, total=3)

        assert len(ann_list) == 3
        assert ann_list[0].query == "Query 0"

        for ann in ann_list:
            assert isinstance(ann, Annotation)