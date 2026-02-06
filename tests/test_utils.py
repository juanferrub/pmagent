"""
Tests for utility functions.

Validates:
- AC-10.1: Retry logic
- AC-10.2: Circuit breaker
- Helpers
"""

from __future__ import annotations

import time

import pytest

from src.utils import (
    CircuitBreaker,
    retry_with_backoff,
    now_iso,
    safe_json_dumps,
    truncate,
)


class TestCircuitBreaker:
    """Test circuit breaker implementation (AC-10.2)."""

    def test_initial_state_closed(self):
        cb = CircuitBreaker(threshold=3)
        assert cb.is_open is False
        assert cb._state == "closed"

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(threshold=3, reset_timeout=60)
        cb.record_failure()
        assert cb.is_open is False
        cb.record_failure()
        assert cb.is_open is False
        cb.record_failure()  # 3rd failure = opens
        assert cb.is_open is True

    def test_success_resets_counter(self):
        cb = CircuitBreaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb._failure_count == 0
        assert cb.is_open is False

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(threshold=1, reset_timeout=0.1)
        cb.record_failure()
        assert cb.is_open is True
        time.sleep(0.15)
        assert cb.is_open is False  # transitions to half-open
        assert cb._state == "half-open"

    def test_record_success_after_half_open(self):
        cb = CircuitBreaker(threshold=1, reset_timeout=0.1)
        cb.record_failure()
        time.sleep(0.15)
        _ = cb.is_open  # trigger half-open
        cb.record_success()
        assert cb._state == "closed"


class TestRetryWithBackoff:
    """Test retry decorator (AC-10.1)."""

    def test_sync_success_no_retry(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = succeed()
        assert result == "ok"
        assert call_count == 1

    def test_sync_retries_then_succeeds(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary error")
            return "ok"

        result = fail_twice()
        assert result == "ok"
        assert call_count == 3

    def test_sync_exhausts_retries(self):
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fail():
            raise ValueError("permanent error")

        with pytest.raises(ValueError, match="permanent error"):
            always_fail()

    @pytest.mark.asyncio
    async def test_async_success(self):
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        async def async_succeed():
            return "async_ok"

        result = await async_succeed()
        assert result == "async_ok"

    @pytest.mark.asyncio
    async def test_async_retries(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        async def async_fail_once():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("network error")
            return "recovered"

        result = await async_fail_once()
        assert result == "recovered"
        assert call_count == 2


class TestHelpers:
    """Test helper functions."""

    def test_now_iso_format(self):
        ts = now_iso()
        assert "T" in ts
        assert "+" in ts or "Z" in ts or ts.endswith("00")

    def test_safe_json_dumps(self):
        assert safe_json_dumps({"key": "value"}) == '{"key": "value"}'
        assert "2026" in safe_json_dumps({"date": "2026-02-06"})

    def test_safe_json_dumps_non_serializable(self):
        result = safe_json_dumps({"obj": object()})
        assert isinstance(result, str)

    def test_truncate_short(self):
        assert truncate("hello", 10) == "hello"

    def test_truncate_long(self):
        result = truncate("a" * 100, 20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_truncate_exact(self):
        assert truncate("hello", 5) == "hello"
