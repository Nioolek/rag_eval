"""
Custom exception hierarchy for RAG Evaluation System.
"""


class RAGEvalError(Exception):
    """Base exception for all RAG evaluation errors."""

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ConfigurationError(RAGEvalError):
    """Raised when configuration is invalid or missing."""
    pass


class StorageError(RAGEvalError):
    """Raised when storage operations fail."""
    pass


class AnnotationError(RAGEvalError):
    """Raised when annotation operations fail."""
    pass


class EvaluationError(RAGEvalError):
    """Raised when evaluation operations fail."""
    pass


class RAGConnectionError(RAGEvalError):
    """Raised when RAG service connection fails."""
    pass


class MetricCalculationError(RAGEvalError):
    """Raised when metric calculation fails."""
    pass


class ValidationError(RAGEvalError):
    """Raised when input validation fails."""
    pass


class PathTraversalError(StorageError):
    """Raised when path traversal attack is detected."""
    pass