"""
File handling utilities with async support.
"""

import asyncio
import csv
import json
from pathlib import Path
from typing import Any, Optional

import aiofiles

from ..core.exceptions import StorageError
from ..core.logging import logger


async def read_json_file(
    file_path: Path,
    chunk_size: int = 8192,
) -> Any:
    """
    Read a JSON file asynchronously.

    Args:
        file_path: Path to JSON file
        chunk_size: Chunk size for reading

    Returns:
        Parsed JSON data
    """
    if not file_path.exists():
        raise StorageError(f"File not found: {file_path}")

    try:
        async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
            content = await f.read()
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise StorageError(f"Invalid JSON in {file_path}: {e}")
    except Exception as e:
        raise StorageError(f"Failed to read {file_path}: {e}")


async def write_json_file(
    file_path: Path,
    data: Any,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """
    Write data to a JSON file asynchronously.

    Args:
        file_path: Path to output file
        data: Data to write
        indent: JSON indentation
        ensure_ascii: Whether to escape non-ASCII characters
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        content = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)

        async with aiofiles.open(file_path, mode='w', encoding='utf-8') as f:
            await f.write(content)

        logger.debug(f"Wrote JSON to {file_path}")
    except Exception as e:
        raise StorageError(f"Failed to write {file_path}: {e}")


async def read_csv_file(
    file_path: Path,
    encoding: str = 'utf-8',
) -> list[dict[str, Any]]:
    """
    Read a CSV file asynchronously.

    Args:
        file_path: Path to CSV file
        encoding: File encoding

    Returns:
        List of row dictionaries
    """
    if not file_path.exists():
        raise StorageError(f"File not found: {file_path}")

    try:
        # CSV reading is sync, run in executor
        loop = asyncio.get_event_loop()

        def _read_csv():
            rows = []
            with open(file_path, mode='r', encoding=encoding, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(dict(row))
            return rows

        return await loop.run_in_executor(None, _read_csv)
    except Exception as e:
        raise StorageError(f"Failed to read CSV {file_path}: {e}")


async def write_csv_file(
    file_path: Path,
    data: list[dict[str, Any]],
    fieldnames: Optional[list[str]] = None,
    encoding: str = 'utf-8',
) -> None:
    """
    Write data to a CSV file asynchronously.

    Args:
        file_path: Path to output file
        data: List of row dictionaries
        fieldnames: Column names (auto-detected if not provided)
        encoding: File encoding
    """
    if not data:
        raise StorageError("No data to write")

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if fieldnames is None:
            fieldnames = list(data[0].keys())

        # CSV writing is sync, run in executor
        loop = asyncio.get_event_loop()

        def _write_csv():
            with open(file_path, mode='w', encoding=encoding, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)

        await loop.run_in_executor(None, _write_csv)
        logger.debug(f"Wrote CSV to {file_path}")
    except Exception as e:
        raise StorageError(f"Failed to write CSV {file_path}: {e}")


async def read_file_lines(
    file_path: Path,
    skip_empty: bool = True,
) -> list[str]:
    """
    Read file lines asynchronously.

    Args:
        file_path: Path to file
        skip_empty: Whether to skip empty lines

    Returns:
        List of lines
    """
    if not file_path.exists():
        raise StorageError(f"File not found: {file_path}")

    try:
        async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
            content = await f.read()

        lines = content.split('\n')
        if skip_empty:
            lines = [line for line in lines if line.strip()]

        return lines
    except Exception as e:
        raise StorageError(f"Failed to read {file_path}: {e}")


async def copy_file(
    source: Path,
    destination: Path,
    chunk_size: int = 65536,
) -> None:
    """
    Copy a file asynchronously with chunked reading.

    Args:
        source: Source file path
        destination: Destination file path
        chunk_size: Chunk size for copying
    """
    if not source.exists():
        raise StorageError(f"Source file not found: {source}")

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(source, mode='rb') as src:
            async with aiofiles.open(destination, mode='wb') as dst:
                while True:
                    chunk = await src.read(chunk_size)
                    if not chunk:
                        break
                    await dst.write(chunk)

        logger.debug(f"Copied {source} to {destination}")
    except Exception as e:
        raise StorageError(f"Failed to copy {source} to {destination}: {e}")


def get_file_size(file_path: Path) -> int:
    """Get file size in bytes."""
    if not file_path.exists():
        return 0
    return file_path.stat().st_size


def format_file_size(size_bytes: int) -> str:
    """Format file size for display."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"