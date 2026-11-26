import asyncio

import httpx2 as httpx
import pytest
from mrai import retry


def _run(transport: retry.RetryTransport) -> httpx.Response:
    async def go():
        async with httpx.AsyncClient(transport=transport) as client:
            return await client.post("https://api.example/v1/chat", json={"q": 1})

    return asyncio.run(go())


def _transport(responses, waits, **kw):
    seen = []

    def handler(request):
        seen.append(request)
        nxt = responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    async def sleep(s):
        waits.append(s)

    t = retry.RetryTransport(httpx.MockTransport(handler), sleep=sleep, **kw)
    return t, seen


def test_retries_a_transient_status_then_succeeds():
    waits = []
    t, seen = _transport([httpx.Response(529), httpx.Response(200, json={"ok": 1})], waits)
    resp = _run(t)
    assert resp.status_code == 200 and len(seen) == 2
    assert waits == [1.0]
    assert seen[1].content == seen[0].content  # body replayed intact


def test_honours_retry_after_capped_at_max_wait():
    waits = []
    t, _ = _transport(
        [httpx.Response(429, headers={"retry-after": "120"}), httpx.Response(200)],
        waits,
        max_wait=30,
    )
    assert _run(t).status_code == 200
    assert waits == [30]


def test_gives_up_after_attempts_and_returns_the_last_response():
    waits = []
    t, seen = _transport([httpx.Response(504)] * 3, waits, attempts=3)
    assert _run(t).status_code == 504
    assert len(seen) == 3 and waits == [1.0, 2.0]


def test_non_transient_status_is_not_retried():
    waits = []
    t, seen = _transport([httpx.Response(400)], waits)
    assert _run(t).status_code == 400 and len(seen) == 1 and waits == []


def test_connect_errors_are_retried_then_raised():
    waits = []
    err = httpx.ConnectError("refused")
    t, seen = _transport([err, err], waits, attempts=2)
    with pytest.raises(httpx.ConnectError):
        _run(t)
    assert len(seen) == 2


def test_reports_each_retry(monkeypatch):
    calls = []
    monkeypatch.setattr(retry, "on_retry", lambda *a: calls.append(a))
    t, _ = _transport([httpx.Response(503), httpx.Response(200)], [])
    _run(t)
    assert calls == [("HTTP 503", 2, 4, 1.0)]
