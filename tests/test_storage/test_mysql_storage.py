"""Tests for MySQL storage backend."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import pytest
import pytest_asyncio

# Check if MySQL is available
MYSQL_AVAILABLE = False
try:
    import aiomysql
    MYSQL_AVAILABLE = True
except ImportError:
    pass

# Skip all tests if MySQL is not available
pytestmark = pytest.mark.skipif(
    not MYSQL_AVAILABLE,
    reason="aiomysql not installed or MySQL not available"
)


def get_mysql_url() -> Optional[str]:
    """Get MySQL URL from environment or return None."""
    return os.getenv("MYSQL_TEST_URL") or os.getenv("DATABASE_URL")


@pytest_asyncio.fixture
async def mysql_storage():
    """Create MySQL storage for testing."""
    mysql_url = get_mysql_url()
    if not mysql_url:
        pytest.skip("MySQL test URL not configured (set MYSQL_TEST_URL or DATABASE_URL)")

    from src.storage.mysql_storage import MySQLStorage

    # Parse URL and create test database
    storage = MySQLStorage(
        database_url=mysql_url,
        pool_size=5,
        pool_recycle=3600,
    )

    try:
        await storage.initialize()
        yield storage
    finally:
        # Cleanup: close connection
        await storage.close()


@pytest_asyncio.fixture
async def clean_mysql_storage(mysql_storage):
    """Create MySQL storage with clean tables for each test."""
    # Clean up tables before test
    async with mysql_storage._pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("DELETE FROM annotations")
            await cursor.execute("DELETE FROM evaluation_results")
            await cursor.execute("DELETE FROM evaluation_runs")
            await cursor.execute("DELETE FROM records")
            await cursor.execute("DELETE FROM record_versions")

    yield mysql_storage


class TestMySQLStorageBasic:
    """Basic tests for MySQLStorage backend."""

    @pytest.mark.asyncio
    async def test_initialize(self, mysql_storage):
        """Test MySQL storage initialization."""
        assert mysql_storage._pool is not None

    @pytest.mark.asyncio
    async def test_save_and_get_dedicated_table(self, clean_mysql_storage):
        """Test save and get operations for annotations table."""
        storage = clean_mysql_storage

        data = {
            "id": "test-anno-1",
            "query": "What is machine learning?",
            "agent_id": "test-agent",
            "language": "en",
            "gt_documents": ["Document 1", "Document 2"],
            "faq_matched": False,
            "should_refuse": False,
        }

        await storage.save("annotations", data)
        result = await storage.get("annotations", "test-anno-1")

        assert result is not None
        assert result["query"] == "What is machine learning?"
        assert result["agent_id"] == "test-agent"
        assert result["gt_documents"] == ["Document 1", "Document 2"]

    @pytest.mark.asyncio
    async def test_save_and_get_generic_table(self, clean_mysql_storage):
        """Test save and get operations for generic records table."""
        storage = clean_mysql_storage

        data = {
            "id": "test-generic-1",
            "name": "Test Item",
            "value": 42,
            "nested": {"key": "value"},
        }

        await storage.save("custom_collection", data)
        result = await storage.get("custom_collection", "test-generic-1")

        assert result is not None
        assert result["name"] == "Test Item"
        assert result["value"] == 42
        assert result["nested"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_update(self, clean_mysql_storage):
        """Test update operation."""
        storage = clean_mysql_storage

        data = {"id": "test-update-1", "value": 10}
        await storage.save("annotations", data)

        await storage.update("annotations", "test-update-1", {"value": 20})

        result = await storage.get("annotations", "test-update-1")
        assert result["value"] == 20

    @pytest.mark.asyncio
    async def test_delete(self, clean_mysql_storage):
        """Test soft delete operation."""
        storage = clean_mysql_storage

        data = {"id": "test-delete-1", "name": "To Delete"}
        await storage.save("annotations", data)

        await storage.delete("annotations", "test-delete-1")

        # Soft delete - should return None
        result = await storage.get("annotations", "test-delete-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_with_pagination(self, clean_mysql_storage):
        """Test get_all with pagination."""
        storage = clean_mysql_storage

        for i in range(10):
            await storage.save("annotations", {
                "id": f"page-test-{i}",
                "query": f"Query {i}",
            })

        results = await storage.get_all("annotations", limit=5)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_count(self, clean_mysql_storage):
        """Test count operation."""
        storage = clean_mysql_storage

        for i in range(5):
            await storage.save("annotations", {
                "id": f"count-test-{i}",
                "query": f"Query {i}",
            })

        count = await storage.count("annotations")

        assert count == 5


class TestMySQLStorageAdvanced:
    """Advanced tests for MySQLStorage backend."""

    @pytest.mark.asyncio
    async def test_filters(self, clean_mysql_storage):
        """Test filtering by agent_id."""
        storage = clean_mysql_storage

        for i in range(5):
            await storage.save("annotations", {
                "id": f"filter-test-{i}",
                "query": f"Query {i}",
                "agent_id": "agent-a" if i < 3 else "agent-b",
            })

        results = await storage.get_all(
            "annotations",
            filters={"agent_id": "agent-a"},
        )

        assert len(results) == 3
        for r in results:
            assert r["agent_id"] == "agent-a"

    @pytest.mark.asyncio
    async def test_query_with_sort(self, clean_mysql_storage):
        """Test sorting results."""
        storage = clean_mysql_storage

        for i in range(5):
            await storage.save("annotations", {
                "id": f"sort-test-{i}",
                "query": f"Query {i}",
            })

        # Sort descending
        results_desc = await storage.query_with_sort(
            "annotations",
            sort_by="created_at",
            sort_desc=True,
            limit=10,
        )

        # Sort ascending
        results_asc = await storage.query_with_sort(
            "annotations",
            sort_by="created_at",
            sort_desc=False,
            limit=10,
        )

        assert len(results_desc) == 5
        assert len(results_asc) == 5

    @pytest.mark.asyncio
    async def test_search(self, clean_mysql_storage):
        """Test text search."""
        storage = clean_mysql_storage

        await storage.save("annotations", {
            "id": "search-1",
            "query": "What is machine learning?",
            "notes": "Important question about AI",
        })

        await storage.save("annotations", {
            "id": "search-2",
            "query": "How to cook pasta?",
            "notes": "Recipe question",
        })

        results = await storage.search(
            "annotations",
            search_query="machine learning",
            limit=10,
        )

        assert len(results) >= 1
        # Should find the machine learning question
        found_ml = any("machine learning" in r["query"].lower() for r in results)
        assert found_ml

    @pytest.mark.asyncio
    async def test_version_management(self, clean_mysql_storage):
        """Test version saving and retrieval."""
        storage = clean_mysql_storage

        data = {"id": "version-test", "query": "Original query"}
        await storage.save("annotations", data)

        # Save a version
        version = await storage.save_version(
            "annotations",
            "version-test",
            {"query": "Original query", "action": "create"},
        )

        assert version == 1

        # Get versions
        versions = await storage.get_versions("annotations", "version-test")
        assert len(versions) == 1
        assert versions[0]["version_number"] == 1

    @pytest.mark.asyncio
    async def test_iterate(self, clean_mysql_storage):
        """Test iteration over records."""
        storage = clean_mysql_storage

        for i in range(5):
            await storage.save("annotations", {
                "id": f"iter-test-{i}",
                "query": f"Query {i}",
            })

        records = []
        async for record in storage.iterate("annotations", batch_size=2):
            records.append(record)

        assert len(records) == 5

    @pytest.mark.asyncio
    async def test_evaluation_results(self, clean_mysql_storage):
        """Test saving evaluation results."""
        storage = clean_mysql_storage

        data = {
            "id": "eval-result-1",
            "annotation_id": "anno-1",
            "run_id": "run-1",
            "rag_interface": "test-interface",
            "metrics": {"accuracy": 0.95, "latency_ms": 100},
            "success": True,
        }

        await storage.save("evaluation_results", data)
        result = await storage.get("evaluation_results", "eval-result-1")

        assert result is not None
        assert result["annotation_id"] == "anno-1"
        assert result["metrics"]["accuracy"] == 0.95

    @pytest.mark.asyncio
    async def test_evaluation_runs(self, clean_mysql_storage):
        """Test saving evaluation runs."""
        storage = clean_mysql_storage

        data = {
            "id": "eval-run-1",
            "name": "Test Run",
            "rag_interfaces": ["interface-1", "interface-2"],
            "selected_metrics": ["accuracy", "latency"],
            "status": "completed",
            "total_annotations": 10,
            "completed_count": 10,
        }

        await storage.save("evaluation_runs", data)
        result = await storage.get("evaluation_runs", "eval-run-1")

        assert result is not None
        assert result["name"] == "Test Run"
        assert result["rag_interfaces"] == ["interface-1", "interface-2"]
        assert result["status"] == "completed"


class TestMySQLStorageEdgeCases:
    """Edge case tests for MySQLStorage backend."""

    @pytest.mark.asyncio
    async def test_missing_record(self, clean_mysql_storage):
        """Test getting non-existent record."""
        storage = clean_mysql_storage

        result = await storage.get("annotations", "non-existent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, clean_mysql_storage):
        """Test updating non-existent record."""
        storage = clean_mysql_storage

        result = await storage.update("annotations", "non-existent-id", {"value": 1})
        assert result is False

    @pytest.mark.asyncio
    async def test_save_with_existing_id(self, clean_mysql_storage):
        """Test save with existing ID (upsert)."""
        storage = clean_mysql_storage

        data = {"id": "upsert-test", "query": "Original", "value": 1}
        await storage.save("annotations", data)

        # Save again with same ID
        data2 = {"id": "upsert-test", "query": "Updated", "value": 2}
        await storage.save("annotations", data2)

        result = await storage.get("annotations", "upsert-test")
        assert result["query"] == "Updated"
        assert result["value"] == 2

    @pytest.mark.asyncio
    async def test_json_field_handling(self, clean_mysql_storage):
        """Test handling of complex JSON fields."""
        storage = clean_mysql_storage

        data = {
            "id": "json-test",
            "query": "Test query",
            "gt_documents": ["Doc 1", "Doc 2", "Doc 3"],
            "custom_fields": {
                "nested": {
                    "deep": {
                        "value": 42,
                    },
                },
                "list": [1, 2, 3],
            },
        }

        await storage.save("annotations", data)
        result = await storage.get("annotations", "json-test")

        assert result["gt_documents"] == ["Doc 1", "Doc 2", "Doc 3"]
        assert result["custom_fields"]["nested"]["deep"]["value"] == 42
        assert result["custom_fields"]["list"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_connection_pool(self, mysql_storage):
        """Test that connection pool works correctly."""
        # Run multiple concurrent operations
        async def save_item(i):
            await mysql_storage.save("annotations", {
                "id": f"pool-test-{i}",
                "query": f"Query {i}",
            })

        # Run 10 concurrent saves
        import asyncio
        tasks = [save_item(i) for i in range(10)]
        await asyncio.gather(*tasks)

        # Verify all saved
        count = await mysql_storage.count("annotations")
        assert count >= 10