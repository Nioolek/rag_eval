"""
Circuit Breaker Implementation.

Protects services from cascading failures by temporarily blocking calls.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from rag_rag.core.logging import get_logger

logger = get_logger("rag_rag.degradation.circuit_breaker")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, blocking calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 3
    success_threshold: int = 2  # For half-open to closed
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    timeout: float = 30.0  # Operation timeout


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_state_change: float = field(default_factory=time.time)
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0
    total_timeouts: int = 0
    total_circuit_opens: int = 0


class CircuitBreaker:
    """
    Circuit Breaker pattern implementation.

    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Failing, calls are blocked immediately
    - HALF_OPEN: Testing recovery, limited calls allowed
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current state."""
        return self._stats.state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self._stats.state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed."""
        return self._stats.state == CircuitState.CLOSED

    def _should_allow_call(self) -> bool:
        """Check if call should be allowed."""
        if self._stats.state == CircuitState.CLOSED:
            return True

        if self._stats.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            elapsed = time.time() - self._stats.last_failure_time
            if elapsed >= self.config.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
                return True
            return False

        # HALF_OPEN: Allow limited calls
        return True

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._stats.state
        self._stats.state = new_state
        self._stats.last_state_change = time.time()

        if new_state == CircuitState.OPEN:
            self._stats.total_circuit_opens += 1

        logger.info(
            f"Circuit breaker [{self.name}] transitioned: "
            f"{old_state.value} -> {new_state.value}"
        )

    def _record_success(self) -> None:
        """Record a successful call."""
        self._stats.success_count += 1
        self._stats.total_successes += 1
        self._stats.total_calls += 1

        if self._stats.state == CircuitState.HALF_OPEN:
            if self._stats.success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
                self._stats.failure_count = 0
                self._stats.success_count = 0

        elif self._stats.state == CircuitState.CLOSED:
            # Reset failure count on success
            self._stats.failure_count = 0

    def _record_failure(self) -> None:
        """Record a failed call."""
        self._stats.failure_count += 1
        self._stats.total_failures += 1
        self._stats.total_calls += 1
        self._stats.last_failure_time = time.time()

        if self._stats.state == CircuitState.HALF_OPEN:
            # Failure in half-open -> back to open
            self._transition_to(CircuitState.OPEN)

        elif self._stats.state == CircuitState.CLOSED:
            if self._stats.failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def _record_timeout(self) -> None:
        """Record a timeout."""
        self._stats.total_timeouts += 1
        self._record_failure()

    async def call(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Async function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: If function fails
        """
        from rag_rag.core.exceptions import CircuitBreakerOpenError

        async with self._lock:
            if not self._should_allow_call():
                raise CircuitBreakerOpenError(
                    self.name,
                    self.config.recovery_timeout,
                )

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout,
            )
            self._record_success()
            return result

        except asyncio.TimeoutError:
            self._record_timeout()
            raise

        except Exception as e:
            self._record_failure()
            raise

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._stats = CircuitBreakerStats()
        logger.info(f"Circuit breaker [{self.name}] reset to CLOSED")

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._stats.state.value,
            "failure_count": self._stats.failure_count,
            "success_count": self._stats.success_count,
            "total_calls": self._stats.total_calls,
            "total_failures": self._stats.total_failures,
            "total_successes": self._stats.total_successes,
            "total_timeouts": self._stats.total_timeouts,
            "total_circuit_opens": self._stats.total_circuit_opens,
        }


def with_circuit_breaker(
    circuit: CircuitBreaker,
    fallback: Optional[Callable[..., Any]] = None,
):
    """
    Decorator to wrap a function with circuit breaker.

    Args:
        circuit: Circuit breaker instance
        fallback: Optional fallback function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await circuit.call(func, *args, **kwargs)
            except Exception as e:
                if fallback:
                    logger.warning(
                        f"Circuit breaker [{circuit.name}] caught error, "
                        f"using fallback: {e}"
                    )
                    return await fallback(*args, **kwargs)
                raise

        return wrapper

    return decorator