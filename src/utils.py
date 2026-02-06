"""
Shared utilities for the PM Agent.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, Optional

import structlog

# ── Structured Logging ───────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("pm_agent")


# ── Retry decorator with exponential backoff ─────────────────────

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator: retry a function with exponential backoff.
    Satisfies AC-10.1 retry logic.
    """

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            "retry_attempt",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            delay=delay,
                            error=str(e),
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "retry_exhausted",
                            function=func.__name__,
                            max_retries=max_retries,
                            error=str(e),
                        )
            raise last_exception  # type: ignore

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            "retry_attempt",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            delay=delay,
                            error=str(e),
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "retry_exhausted",
                            function=func.__name__,
                            max_retries=max_retries,
                            error=str(e),
                        )
            raise last_exception  # type: ignore

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ── Circuit Breaker ──────────────────────────────────────────────

class CircuitBreaker:
    """
    Simple circuit breaker (AC-10.2).
    Opens after `threshold` consecutive failures. Resets after `reset_timeout` seconds.
    """

    def __init__(self, threshold: int = 5, reset_timeout: float = 60.0):
        self.threshold = threshold
        self.reset_timeout = reset_timeout
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = "closed"  # closed, open, half-open

    @property
    def is_open(self) -> bool:
        if self._state == "open":
            if self._last_failure_time and (time.time() - self._last_failure_time) > self.reset_timeout:
                self._state = "half-open"
                return False
            return True
        return False

    def record_success(self):
        self._failure_count = 0
        self._state = "closed"

    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.threshold:
            self._state = "open"
            logger.error("circuit_breaker_opened", failure_count=self._failure_count)


# ── Helpers ──────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json_dumps(obj: Any) -> str:
    """JSON serialize with fallback for non-serializable objects."""
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        return str(obj)


def truncate(text: str, max_length: int = 500) -> str:
    """Truncate text for display / logging."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
