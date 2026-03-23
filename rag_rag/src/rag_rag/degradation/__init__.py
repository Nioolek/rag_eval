"""Degradation module: circuit breaker, fallback handlers, degradation manager."""

from rag_rag.degradation.circuit_breaker import CircuitBreaker
from rag_rag.degradation.degradation_manager import DegradationManager

__all__ = [
    "CircuitBreaker",
    "DegradationManager",
]