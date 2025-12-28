"""
Retry Module: Exponential backoff and retry logic for network operations.

Provides:
- Exponential backoff strategy for retries
- Jitter to prevent thundering herd
- Configurable retry policies for different error types
- Circuit breaker pattern for failing operations
"""
from __future__ import annotations

import logging
import time
import random
from typing import Callable, TypeVar, Optional, Any
from functools import wraps
from datetime import datetime, timedelta, timezone

from .errors import AgentSentinelError, NetworkError, SyncError, TimeoutError

logger = logging.getLogger("agent_sentinel")

T = TypeVar("T")


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[list[type[Exception]]] = None,
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of attempts (including initial)
            initial_delay: Initial delay in seconds before first retry
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff calculation
            jitter: Whether to add random jitter to delays
            retryable_exceptions: List of exception types to retry on
        """
        self.max_attempts = max(1, max_attempts)
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [
            NetworkError,
            TimeoutError,
            Exception,  # Catch-all for unknown errors
        ]
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay before retry attempt.
        
        Args:
            attempt: Zero-based attempt number
            
        Returns:
            Delay in seconds
        """
        if attempt < 0:
            return 0
        
        # Exponential backoff: initial_delay * (base ^ attempt)
        delay = self.initial_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        # Add jitter (±10% of delay)
        if self.jitter:
            jitter_amount = delay * 0.1
            delay = delay + random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    States: CLOSED (working) -> OPEN (failing) -> HALF_OPEN (testing)
    """
    
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Failing, reject new requests
    HALF_OPEN = "HALF_OPEN"  # Testing if recovered
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] = Exception,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type that triggers the breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result of function call
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == self.OPEN:
            if self._should_attempt_recovery():
                self.state = self.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state, attempting recovery")
            else:
                raise AgentSentinelError(
                    "Circuit breaker is OPEN. Service unavailable.",
                    error_code="CIRCUIT_OPEN",
                    details={
                        "state": self.state,
                        "failure_count": self.failure_count,
                    },
                    recoverable=False,
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful call."""
        if self.state == self.HALF_OPEN:
            logger.info("Circuit breaker recovered, returning to CLOSED state")
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.failure_count >= self.failure_threshold:
            self.state = self.OPEN
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )
        else:
            logger.debug(
                f"Circuit breaker failure count: {self.failure_count}/{self.failure_threshold}"
            )


def with_retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry a function with exponential backoff.
    
    Usage:
        @with_retry(config=RetryConfig(max_attempts=3))
        def flaky_network_call():
            return api_call()
    
    Args:
        config: RetryConfig instance (uses defaults if None)
        on_retry: Optional callback called on each retry with (attempt, exception)
        
    Returns:
        Decorated function that retries on failure
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if this exception is retryable
                    is_retryable = any(
                        isinstance(e, exc_type)
                        for exc_type in config.retryable_exceptions
                    )
                    
                    if not is_retryable or attempt == config.max_attempts - 1:
                        # Not retryable or last attempt, raise immediately
                        raise
                    
                    # Calculate delay and notify
                    delay = config.get_delay(attempt)
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{config.max_attempts} failed: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    
                    if on_retry:
                        on_retry(attempt + 1, e)
                    
                    time.sleep(delay)
            
            # Should not reach here, but raise last exception if we do
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Unexpected error in retry logic for {func.__name__}")
        
        return wrapper
    
    return decorator


def with_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to add circuit breaker pattern to a function.
    
    Usage:
        @with_circuit_breaker(failure_threshold=5)
        def external_api_call():
            return call_external_api()
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds before attempting recovery
        
    Returns:
        Decorated function with circuit breaker
    """
    breaker = CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
    )
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return breaker.call(func, *args, **kwargs)
        
        # Expose breaker for testing/monitoring
        wrapper.breaker = breaker  # type: ignore
        
        return wrapper
    
    return decorator


