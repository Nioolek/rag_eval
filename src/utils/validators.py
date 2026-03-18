"""
Input validation and sanitization utilities.
"""

import re
from pathlib import Path
from typing import Any, Optional

from ..core.exceptions import ValidationError, PathTraversalError


def validate_query(query: str, max_length: int = 10000) -> str:
    """
    Validate and sanitize a query string.

    Args:
        query: Query string to validate
        max_length: Maximum allowed length

    Returns:
        Sanitized query string

    Raises:
        ValidationError: If validation fails
    """
    if not query:
        raise ValidationError("Query cannot be empty")

    query = query.strip()

    if len(query) > max_length:
        raise ValidationError(f"Query exceeds maximum length of {max_length}")

    # Remove potentially dangerous characters
    query = sanitize_input(query)

    return query


def validate_annotation_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate annotation data.

    Args:
        data: Annotation data dictionary

    Returns:
        Validated data

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(data, dict):
        raise ValidationError("Annotation data must be a dictionary")

    # Required field
    if "query" not in data:
        raise ValidationError("Annotation must have a 'query' field")

    # Validate query
    data["query"] = validate_query(data["query"])

    # Validate optional fields
    if "conversation_history" in data:
        if not isinstance(data["conversation_history"], list):
            raise ValidationError("conversation_history must be a list")
        data["conversation_history"] = [
            sanitize_input(str(h)) for h in data["conversation_history"]
        ]

    if "agent_id" in data:
        data["agent_id"] = sanitize_input(str(data["agent_id"]))[:100]

    if "gt_documents" in data:
        if not isinstance(data["gt_documents"], list):
            raise ValidationError("gt_documents must be a list")
        data["gt_documents"] = [
            sanitize_input(str(d)) for d in data["gt_documents"]
        ]

    if "standard_answers" in data:
        if not isinstance(data["standard_answers"], list):
            raise ValidationError("standard_answers must be a list")
        data["standard_answers"] = [
            sanitize_input(str(a)) for a in data["standard_answers"]
        ]

    return data


def sanitize_input(text: str) -> str:
    """
    Sanitize input text to prevent injection attacks.

    Args:
        text: Input text

    Returns:
        Sanitized text
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace('\x00', '')

    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # Normalize unicode
    text = text.encode('utf-8', errors='ignore').decode('utf-8')

    return text


def validate_file_path(
    path: str,
    base_dir: Path,
    allowed_extensions: Optional[set[str]] = None,
) -> Path:
    """
    Validate a file path to prevent path traversal attacks.

    Args:
        path: File path to validate
        base_dir: Base directory that the path must be within
        allowed_extensions: Set of allowed file extensions

    Returns:
        Validated Path object

    Raises:
        PathTraversalError: If path traversal detected
        ValidationError: If validation fails
    """
    if not path:
        raise ValidationError("File path cannot be empty")

    # Convert to Path object
    file_path = Path(path)

    # Check for path traversal patterns
    dangerous_patterns = ['../', '..\\', '/../', '\\..\\']
    for pattern in dangerous_patterns:
        if pattern in path:
            raise PathTraversalError(f"Path traversal detected: {path}")

    # Resolve to absolute path
    try:
        resolved_path = (base_dir / file_path).resolve()
    except Exception as e:
        raise ValidationError(f"Invalid file path: {e}")

    # Ensure path is within base directory
    try:
        resolved_path.relative_to(base_dir.resolve())
    except ValueError:
        raise PathTraversalError(f"Path outside base directory: {path}")

    # Check extension if specified
    if allowed_extensions:
        ext = resolved_path.suffix.lower()
        if ext not in allowed_extensions:
            raise ValidationError(
                f"File extension '{ext}' not allowed. "
                f"Allowed: {allowed_extensions}"
            )

    return resolved_path


def validate_json_size(data: Any, max_size_mb: float = 10.0) -> None:
    """
    Validate JSON data size.

    Args:
        data: JSON data to check
        max_size_mb: Maximum size in megabytes

    Raises:
        ValidationError: If data exceeds size limit
    """
    import json
    import sys

    try:
        size = sys.getsizeof(json.dumps(data))
        max_size_bytes = max_size_mb * 1024 * 1024

        if size > max_size_bytes:
            raise ValidationError(
                f"Data size ({size / 1024 / 1024:.2f} MB) exceeds "
                f"maximum allowed ({max_size_mb} MB)"
            )
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Cannot determine data size: {e}")