"""
Logging Utilities for RAG Pipeline.

Provides structured logging with:
- JSON format for production
- Colored output for development
- Async-safe handlers
- Context-aware logging
"""

import logging
import sys
from datetime import datetime
from typing import Any, Optional

# Try to import colorama for Windows color support
try:
    from colorama import Fore, Style, init as colorama_init

    colorama_init()
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False
    Fore = None
    Style = None


# ANSI color codes for fallback
class Colors:
    """ANSI color codes."""

    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


class LogColors:
    """Log level colors."""

    DEBUG = Colors.CYAN
    INFO = Colors.GREEN
    WARNING = Colors.YELLOW
    ERROR = Colors.RED
    CRITICAL = Colors.MAGENTA


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for development."""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: bool = True,
    ):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and (HAS_COLORAMA or sys.stdout.isatty())

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Save original levelname
        orig_levelname = record.levelname

        if self.use_colors:
            # Add colors to level name
            color = getattr(LogColors, record.levelname, Colors.WHITE)
            record.levelname = f"{color}{record.levelname}{Colors.RESET}"

        result = super().format(record)

        # Restore original levelname
        record.levelname = orig_levelname

        return result


class JSONFormatter(logging.Formatter):
    """JSON log formatter for production."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        import json

        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


class ContextFilter(logging.Filter):
    """Add context information to log records."""

    def __init__(self, context: Optional[dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to record."""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


class RAGLogger(logging.Logger):
    """Custom logger with extra methods for RAG pipeline."""

    def __init__(self, name: str):
        super().__init__(name)

    def with_context(self, **kwargs: Any) -> "RAGLogger":
        """Create logger with additional context."""
        logger = RAGLogger(self.name)
        logger.handlers = self.handlers
        logger.level = self.level
        logger.addFilter(ContextFilter(kwargs))
        return logger

    def stage(self, stage_name: str, message: str, **kwargs: Any) -> None:
        """Log pipeline stage information."""
        self.info(
            f"[{stage_name}] {message}",
            extra={"extra_data": {"stage": stage_name, **kwargs}},
        )

    def timing(self, stage_name: str, duration_ms: float) -> None:
        """Log timing information for a stage."""
        self.debug(
            f"[{stage_name}] completed in {duration_ms:.2f}ms",
            extra={"extra_data": {"stage": stage_name, "duration_ms": duration_ms}},
        )


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """
    Set up logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format for production
        log_file: Optional log file path
    """
    # Set custom logger class
    logging.setLoggerClass(RAGLogger)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = ColoredFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())  # Always JSON for files
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> RAGLogger:
    """Get a logger instance."""
    return logging.getLogger(name)  # type: ignore


# Module-level logger
logger = get_logger("rag_rag")


# Convenience functions
def log_stage(stage_name: str, message: str, **kwargs: Any) -> None:
    """Log pipeline stage information."""
    logger.stage(stage_name, message, **kwargs)


def log_timing(stage_name: str, duration_ms: float) -> None:
    """Log timing information for a stage."""
    logger.debug(
        f"[{stage_name}] completed in {duration_ms:.2f}ms",
        extra={"extra_data": {"stage": stage_name, "duration_ms": duration_ms}},
    )


def log_error(
    stage: str,
    error: Exception,
    query: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Log error with context."""
    logger.error(
        f"[{stage}] Error: {error}",
        extra={
            "extra_data": {
                "stage": stage,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "query": query,
                **kwargs,
            }
        },
        exc_info=True,
    )