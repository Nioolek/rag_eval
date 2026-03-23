"""
Exception Hierarchy for RAG Pipeline.

All exceptions inherit from RAGError for unified error handling.
"""


class RAGError(Exception):
    """Base exception for all RAG pipeline errors."""

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(RAGError):
    """Configuration related errors."""

    pass


class StorageError(RAGError):
    """Storage layer errors."""

    pass


class VectorStoreError(StorageError):
    """Vector store specific errors."""

    pass


class FulltextStoreError(StorageError):
    """Fulltext store specific errors."""

    pass


class GraphStoreError(StorageError):
    """Graph store specific errors."""

    pass


class FAQStoreError(StorageError):
    """FAQ store specific errors."""

    pass


class SessionStoreError(StorageError):
    """Session store specific errors."""

    pass


class ServiceError(RAGError):
    """External service errors."""

    pass


class LLMServiceError(ServiceError):
    """LLM service specific errors."""

    pass


class EmbeddingServiceError(ServiceError):
    """Embedding service specific errors."""

    pass


class RerankServiceError(ServiceError):
    """Rerank service specific errors."""

    pass


class IngestionError(RAGError):
    """Data ingestion errors."""

    pass


class ValidationError(RAGError):
    """Validation errors for input data."""

    pass


class RefusalError(RAGError):
    """Refusal related errors."""

    pass


class TimeoutError(RAGError):
    """Timeout errors for operations."""

    pass


class RateLimitError(ServiceError):
    """Rate limit exceeded errors."""

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        details: dict | None = None,
    ):
        super().__init__(message, details)
        self.retry_after = retry_after


class CircuitBreakerOpenError(ServiceError):
    """Circuit breaker is open, service unavailable."""

    def __init__(
        self,
        service_name: str,
        recovery_timeout: float,
        details: dict | None = None,
    ):
        super().__init__(
            f"Circuit breaker open for service: {service_name}",
            details,
        )
        self.service_name = service_name
        self.recovery_timeout = recovery_timeout