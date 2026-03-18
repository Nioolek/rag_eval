"""Tests for utility functions."""

import pytest
from pathlib import Path

from src.utils.validators import (
    validate_query,
    validate_annotation_data,
    sanitize_input,
    validate_file_path,
)
from src.utils.file_handlers import format_file_size


class TestValidators:
    """Tests for validation utilities."""

    def test_validate_query(self):
        """Test query validation."""
        # Valid query
        result = validate_query("这是一个有效的查询")
        assert result == "这是一个有效的查询"

        # Query with extra whitespace
        result = validate_query("  带空格的查询  ")
        assert result == "带空格的查询"

    def test_validate_query_empty(self):
        """Test empty query validation."""
        from src.core.exceptions import ValidationError

        with pytest.raises(ValidationError):
            validate_query("")

    def test_validate_query_too_long(self):
        """Test query max length."""
        from src.core.exceptions import ValidationError

        long_query = "a" * 20000
        with pytest.raises(ValidationError):
            validate_query(long_query, max_length=10000)

    def test_sanitize_input(self):
        """Test input sanitization."""
        # Normal text
        assert sanitize_input("正常文本") == "正常文本"

        # Text with null bytes
        assert sanitize_input("文本\x00内容") == "文本内容"

        # Text with control characters
        assert sanitize_input("文本\x1b内容") == "文本内容"

    def test_validate_annotation_data(self):
        """Test annotation data validation."""
        data = {
            "query": "测试查询",
            "conversation_history": ["消息1", "消息2"],
        }

        result = validate_annotation_data(data)

        assert result["query"] == "测试查询"
        assert len(result["conversation_history"]) == 2

    def test_validate_annotation_data_missing_query(self):
        """Test annotation validation without query."""
        from src.core.exceptions import ValidationError

        with pytest.raises(ValidationError):
            validate_annotation_data({"agent_id": "test"})

    def test_validate_file_path(self):
        """Test file path validation."""
        base_dir = Path("/tmp/test")

        # Valid path
        result = validate_file_path("subdir/file.txt", base_dir)
        assert result.is_absolute()

    def test_validate_file_path_traversal(self):
        """Test path traversal detection."""
        from src.core.exceptions import PathTraversalError
        from tempfile import tempdir

        base_dir = Path(tempdir)

        with pytest.raises(PathTraversalError):
            validate_file_path("../../../etc/passwd", base_dir)

    def test_validate_file_path_extension(self):
        """Test file extension validation."""
        from src.core.exceptions import ValidationError
        from tempfile import tempdir

        base_dir = Path(tempdir)

        with pytest.raises(ValidationError):
            validate_file_path(
                "test.exe",
                base_dir,
                allowed_extensions={".txt", ".json"}
            )


class TestFileHandlers:
    """Tests for file handling utilities."""

    def test_format_file_size(self):
        """Test file size formatting."""
        assert format_file_size(500) == "500.0 B"
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"


class TestAsyncHelpers:
    """Tests for async helper utilities."""

    def test_run_async(self):
        """Test run_async helper."""
        import asyncio
        from src.utils.async_helpers import run_async

        async def async_function():
            await asyncio.sleep(0.01)
            return "result"

        result = run_async(async_function())
        assert result == "result"

    @pytest.mark.asyncio
    async def test_gather_with_concurrency(self):
        """Test concurrent task gathering."""
        import asyncio
        from src.utils.async_helpers import gather_with_concurrency

        async def task(n):
            await asyncio.sleep(0.01)
            return n

        tasks = [task(i) for i in range(5)]
        results = await gather_with_concurrency(tasks, concurrency=2)

        assert results == [0, 1, 2, 3, 4]