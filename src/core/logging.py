"""
Logging configuration with security controls.
Ensures sensitive information is never logged.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Sensitive keys that should never be logged
SENSITIVE_KEYS = frozenset([
    "api_key", "apikey", "password", "secret", "token", "credential",
    "openai_api_key", "auth", "authorization"
])


class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive information from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._redact_sensitive(record.msg)
        if hasattr(record, 'args') and record.args:
            record.args = tuple(
                self._redact_sensitive(arg) if isinstance(arg, str) else arg
                for arg in record.args
            )
        return True

    def _redact_sensitive(self, text: str) -> str:
        """Redact sensitive values from text."""
        import re
        for key in SENSITIVE_KEYS:
            # Match patterns like key=value or key: value
            pattern = rf'({key}[=:]\s*)[^\s,;]+'
            text = re.sub(pattern, r'\1[REDACTED]', text, flags=re.IGNORECASE)
        return text


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging with security filters.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format_string: Custom format string

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create logger
    logger = logging.getLogger("rag_eval")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(SensitiveDataFilter())
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.addFilter(SensitiveDataFilter())
        logger.addHandler(file_handler)

    return logger


# Default logger instance
logger = setup_logging()