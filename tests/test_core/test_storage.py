"""Tests for storage backends."""

import pytest
import pytest_asyncio
from pathlib import Path
import tempfile
import shutil

from src.storage.local_storage import LocalStorage
from src.storage.sqlite_storage import SQLiteStorage


class TestLocalStorage:
    """Tests for LocalStorage backend."""

    @pytest_asyncio.fixture
    async def storage(self):
        """Create temp storage."""
        temp_dir = Path(tempfile.mkdtemp())
        storage = LocalStorage(temp_dir)
        await storage.initialize()

        yield storage

        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_save_and_get(self, storage):
        """Test save and get operations."""
        data = {"id": "test-1", "name": "Test Item"}

        await storage.save("test_collection", data)
        result = await storage.get("test_collection", "test-1")

        assert result is not None
        assert result["name"] == "Test Item"

    @pytest.mark.asyncio
    async def test_update(self, storage):
        """Test update operation."""
        data = {"id": "test-2", "value": 10}
        await storage.save("test_collection", data)

        await storage.update("test_collection", "test-2", {"value": 20})

        result = await storage.get("test_collection", "test-2")
        assert result["value"] == 20

    @pytest.mark.asyncio
    async def test_delete(self, storage):
        """Test delete operation."""
        data = {"id": "test-3", "name": "To Delete"}
        await storage.save("test_collection", data)

        await storage.delete("test_collection", "test-3")

        # Soft delete - should still exist but marked
        result = await storage.get("test_collection", "test-3")
        # After soft delete, get should return None
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all(self, storage):
        """Test get_all with pagination."""
        for i in range(10):
            await storage.save("test_collection", {"id": f"item-{i}", "index": i})

        results = await storage.get_all("test_collection", limit=5)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_count(self, storage):
        """Test count operation."""
        for i in range(5):
            await storage.save("test_collection", {"id": f"count-{i}"})

        count = await storage.count("test_collection")

        assert count == 5

    @pytest.mark.asyncio
    async def test_version_management(self, storage):
        """Test version saving."""
        data = {"id": "version-test", "version": 1}
        await storage.save("test_collection", data)

        version = await storage.save_version(
            "test_collection",
            "version-test",
            {"data": data, "action": "create"}
        )

        assert version == 1

        versions = await storage.get_versions("test_collection", "version-test")
        assert len(versions) == 1

    @pytest.mark.asyncio
    async def test_path_traversal_protection(self, storage):
        """Test path traversal attack protection."""
        from src.core.exceptions import PathTraversalError

        with pytest.raises(PathTraversalError):
            # Attempt path traversal
            await storage.get("../../../etc/passwd", "test")


class TestSQLiteStorage:
    """Tests for SQLiteStorage backend."""

    @pytest_asyncio.fixture
    async def storage(self):
        """Create temp SQLite storage."""
        temp_dir = Path(tempfile.mkdtemp())
        db_path = temp_dir / "test.db"

        storage = SQLiteStorage(f"sqlite:///{db_path}")
        await storage.initialize()

        yield storage

        await storage.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_save_and_get(self, storage):
        """Test SQLite save and get."""
        data = {"id": "sql-1", "name": "SQLite Test"}

        await storage.save("test_table", data)
        result = await storage.get("test_table", "sql-1")

        assert result is not None
        assert result["name"] == "SQLite Test"

    @pytest.mark.asyncio
    async def test_update(self, storage):
        """Test SQLite update."""
        data = {"id": "sql-2", "value": 100}
        await storage.save("test_table", data)

        await storage.update("test_table", "sql-2", {"value": 200})

        result = await storage.get("test_table", "sql-2")
        assert result["value"] == 200

    @pytest.mark.asyncio
    async def test_count(self, storage):
        """Test SQLite count."""
        for i in range(3):
            await storage.save("test_table", {"id": f"count-{i}"})

        count = await storage.count("test_table")

        assert count == 3