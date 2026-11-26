"""Model HTTP client that retries a failed request instead of the whole run."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx2 as httpx  # pydantic-ai deprecates plain httpx clients for providers

# 429 rate limit, 502/503/504 gateway trouble, 529 provider overloaded
RETRY_STATUSES = frozenset({429, 502, 503, 504, 529})
RETRY_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError)

# Set by the CLI to report retries; (reason, attempt about to run, attempts, wait seconds)
on_retry: Callable[[str, int, int, float], None] | None = None


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after-ms")
    if value:
        try:
            return float(value) / 1000
        except ValueError:
            pass
    value = resp.headers.get("retry-after")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
    except (TypeError, ValueError):
        return None


class RetryTransport(httpx.AsyncBaseTransport):
    """Retry one request on transient upstream failures, honouring Retry-After.

    Only the response status is inspected, before any body is read, so a
    streamed reply that started successfully is never replayed.
    """

    def __init__(
        self,
        wrapped: httpx.AsyncBaseTransport | None = None,
        attempts: int = 4,
        max_wait: float = 30.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.wrapped = wrapped or httpx.AsyncHTTPTransport()
        self.attempts = attempts
        self.max_wait = max_wait
        self.sleep = sleep

    def _backoff(self, attempt: int) -> float:
        return min(self.max_wait, 2.0 ** (attempt - 1))

    async def _pause(self, reason: str, attempt: int, wait: float) -> None:
        wait = max(0.0, min(self.max_wait, wait))
        if on_retry is not None:
            on_retry(reason, attempt + 1, self.attempts, wait)
        await self.sleep(wait)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        for attempt in range(1, self.attempts + 1):
            last = attempt == self.attempts
            try:
                resp = await self.wrapped.handle_async_request(request)
            except RETRY_ERRORS as exc:
                if last:
                    raise
                await self._pause(type(exc).__name__, attempt, self._backoff(attempt))
                continue
            if resp.status_code not in RETRY_STATUSES or last:
                return resp
            wait = _retry_after(resp)
            await resp.aclose()
            await self._pause(
                f"HTTP {resp.status_code}",
                attempt,
                self._backoff(attempt) if wait is None else wait,
            )
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        await self.wrapped.aclose()


def model_client() -> httpx.AsyncClient:
    """Client for model calls (same timeouts as pydantic-ai's default)."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=600, connect=5), transport=RetryTransport()
    )
