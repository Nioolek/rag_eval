"""
Degradation Manager Implementation.

Coordinates circuit breakers and fallback strategies.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Optional

from rag_rag.degradation.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from rag_rag.degradation.fallback_handlers import (
    FallbackHandler,
    initialize_default_fallbacks,
    get_fallback,
)
from rag_rag.core.logging import get_logger

logger = get_logger("rag_rag.degradation.manager")


@dataclass
class ServiceStatus:
    """Status of a service."""

    name: str
    available: bool
    circuit_state: str
    failure_rate: float
    last_error: Optional[str] = None


class DegradationManager:
    """
    Manages service degradation across the RAG pipeline.

    Features:
    - Circuit breaker management
    - Fallback strategy coordination
    - Health monitoring
    - Dynamic weight adjustment for retrieval
    """

    _instance: Optional["DegradationManager"] = None

    def __new__(cls) -> "DegradationManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._circuits: dict[str, CircuitBreaker] = {}
        self._fallbacks: dict[str, FallbackHandler] = {}
        self._service_status: dict[str, ServiceStatus] = {}
        self._retrieval_weights: dict[str, float] = {
            "vector": 0.5,
            "fulltext": 0.3,
            "graph": 0.2,
        }
        self._lock = asyncio.Lock()
        self._initialized = True

    def register_service(
        self,
        name: str,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[FallbackHandler] = None,
    ) -> CircuitBreaker:
        """
        Register a service with circuit breaker and fallback.

        Args:
            name: Service name
            circuit_config: Circuit breaker configuration
            fallback: Fallback handler

        Returns:
            Circuit breaker instance
        """
        circuit = CircuitBreaker(name, circuit_config)
        self._circuits[name] = circuit

        if fallback:
            self._fallbacks[name] = fallback

        self._service_status[name] = ServiceStatus(
            name=name,
            available=True,
            circuit_state="closed",
            failure_rate=0.0,
        )

        logger.info(f"Registered service: {name}")
        return circuit

    def get_circuit(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker for a service."""
        return self._circuits.get(name)

    def get_fallback(self, name: str) -> Optional[FallbackHandler]:
        """Get fallback handler for a service."""
        return self._fallbacks.get(name)

    async def call_with_fallback(
        self,
        service_name: str,
        func: Callable[..., Any],
        *args: Any,
        fallback_func: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Call a service with fallback handling.

        Args:
            service_name: Service name
            func: Main function to call
            *args: Function arguments
            fallback_func: Optional fallback function
            **kwargs: Function keyword arguments

        Returns:
            Result from main or fallback function
        """
        circuit = self._circuits.get(service_name)

        if circuit is None:
            # No circuit breaker - call directly
            return await func(*args, **kwargs)

        try:
            result = await circuit.call(func, *args, **kwargs)
            self._update_status(service_name, success=True)
            return result

        except Exception as e:
            self._update_status(service_name, success=False, error=str(e))

            # Try fallback
            fallback = self._fallbacks.get(service_name)
            if fallback:
                logger.warning(
                    f"Service [{service_name}] failed, using fallback: {e}"
                )
                return await fallback.handle(*args, **kwargs)

            if fallback_func:
                logger.warning(
                    f"Service [{service_name}] failed, using fallback function: {e}"
                )
                return await fallback_func(*args, **kwargs)

            raise

    def _update_status(
        self,
        service_name: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Update service status."""
        if service_name not in self._service_status:
            return

        status = self._service_status[service_name]
        circuit = self._circuits.get(service_name)

        if circuit:
            stats = circuit.get_stats()
            status.available = not circuit.is_open
            status.circuit_state = stats["state"]
            status.failure_rate = (
                stats["total_failures"] / max(stats["total_calls"], 1)
            )

        if not success and error:
            status.last_error = error

    def get_service_status(self, name: str) -> Optional[ServiceStatus]:
        """Get status of a service."""
        return self._service_status.get(name)

    def get_all_status(self) -> dict[str, ServiceStatus]:
        """Get status of all services."""
        return self._service_status.copy()

    def adjust_retrieval_weights(self) -> dict[str, float]:
        """
        Adjust retrieval weights based on service availability.

        When a retrieval source is unavailable, redistribute its weight.
        """
        available_sources = []

        for source, weight in self._retrieval_weights.items():
            status = self._service_status.get(f"{source}_store")
            if status and status.available:
                available_sources.append(source)

        if len(available_sources) == len(self._retrieval_weights):
            # All sources available - use original weights
            return self._retrieval_weights.copy()

        if not available_sources:
            # No sources available
            logger.error("All retrieval sources unavailable!")
            return {}

        # Redistribute weights
        total_weight = sum(self._retrieval_weights.values())
        available_weight = sum(
            self._retrieval_weights[s] for s in available_sources
        )

        adjusted = {}
        for source in available_sources:
            original = self._retrieval_weights[source]
            # Scale up proportionally
            adjusted[source] = (original / available_weight) * total_weight

        logger.info(f"Adjusted retrieval weights: {adjusted}")
        return adjusted

    def reset_circuit(self, name: str) -> None:
        """Reset a circuit breaker."""
        circuit = self._circuits.get(name)
        if circuit:
            circuit.reset()
            self._update_status(name, success=True)

    def reset_all_circuits(self) -> None:
        """Reset all circuit breakers."""
        for name, circuit in self._circuits.items():
            circuit.reset()
            self._update_status(name, success=True)

        logger.info("All circuit breakers reset")

    def get_stats(self) -> dict[str, Any]:
        """Get degradation statistics."""
        return {
            "services": {
                name: {
                    "available": status.available,
                    "circuit_state": status.circuit_state,
                    "failure_rate": status.failure_rate,
                    "last_error": status.last_error,
                }
                for name, status in self._service_status.items()
            },
            "retrieval_weights": self._retrieval_weights,
            "adjusted_weights": self.adjust_retrieval_weights(),
        }


def get_degradation_manager() -> DegradationManager:
    """Get singleton DegradationManager instance."""
    return DegradationManager()


def initialize_degradation(
    services: Optional[list[str]] = None,
    embedding_dimension: int = 1024,
) -> DegradationManager:
    """
    Initialize degradation manager with default services.

    Args:
        services: List of services to register
        embedding_dimension: Embedding dimension for fallback

    Returns:
        Configured DegradationManager
    """
    manager = get_degradation_manager()

    # Initialize default fallbacks
    initialize_default_fallbacks(embedding_dimension=embedding_dimension)

    # Register default services
    default_services = services or [
        "llm",
        "embedding",
        "rerank",
        "vector_store",
        "fulltext_store",
        "graph_store",
    ]

    for service in default_services:
        fallback = get_fallback(service)
        manager.register_service(service, fallback=fallback)

    return manager