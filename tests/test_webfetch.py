import httpx
import pytest
from mrai.settings import Settings
from mrai.webfetch import ScoredUrl, WebFetchTool


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=1.0)


def _raise(*args, **kwargs):
    raise AssertionError("should not be reached")


def _html(request=None, text="<html><body><h1>Title</h1><p>Hello <b>world</b>.</p></body></html>"):
    return httpx.Response(200, text=text, headers={"content-type": "text/html"})


def _scored(*pairs) -> list[ScoredUrl]:
    return [ScoredUrl(url=u, score=s) for u, s in pairs]


def test_jev_ranks_and_filters_then_human_approves():
    seen = []

    def screen(urls, reason):
        return _scored(("https://x.example/no", 0.1), ("https://y.example/keep", 0.95))

    def approve(keepers, meta):
        seen.append(([k.url for k in keepers], meta["method"]))
        return [keepers[0].url]

    tool = WebFetchTool(screen=screen, approve=approve, client=_client(_html))
    out = tool.web_fetch(["https://x.example/no", "https://y.example/keep"], "why")
    assert seen == [(["https://y.example/keep"], "GET")]  # below-threshold candidate dropped
    assert "Title" in out and "Hello world" in out
    assert (tool.fetched, tool.denials) == (1, 0)
    assert tool.fetched_urls == ["https://y.example/keep"]  # recorded for the footer


def test_screen_order_is_best_first():
    seen = []

    def screen(urls, reason):
        return _scored(
            ("https://mid.example", 0.7), ("https://best.example", 0.98)
        )  # worst-first in

    tool = WebFetchTool(
        screen=screen,
        approve=lambda u, m: seen.append([k.url for k in u]) or [],
        client=_client(_html),
    )
    tool.web_fetch(["https://mid.example", "https://best.example"], "r")
    assert seen == [["https://best.example", "https://mid.example"]]


def test_all_below_threshold_is_noise_without_human_prompt():
    tool = WebFetchTool(
        screen=lambda u, r: _scored(("https://a.example", 0.1), ("https://b.example", 0.3)),
        approve=_raise,
        client=_client(_html),
    )
    out = tool.web_fetch(["https://a.example", "https://b.example"], "r")
    assert "NOISE" in out and tool.fetched == 0


def test_jev_unavailable_fetches_nothing():
    tool = WebFetchTool(screen=lambda u, r: None, approve=_raise, client=_client(_html))
    out = tool.web_fetch(["https://a.example", "https://b.example"], "r")
    assert "UNAVAILABLE" in out and tool.fetched == 0


def test_approved_none_counts_denial_then_extinguishes():
    approvals = [0, 0, 0]
    tool = WebFetchTool(
        screen=lambda u, r: _scored((u[0], 0.9)),
        approve=lambda u, m: [k.url for k in u[: approvals.pop(0)]],
        client=_client(_html),
        max_attempts=3,
    )
    assert "APPROVED NONE" in tool.web_fetch(["https://a.example/x"], "r")
    assert tool.denials == 1
    tool.web_fetch(["https://b.example/x"], "r")
    assert "EXTINGUISHED" in tool.web_fetch(["https://c.example/x"], "r")
    assert "DISABLED" in tool.web_fetch(["https://d.example/x"], "r")  # skips screen+prompt


def test_budget_enforced_before_screen_and_after_fetches():
    # budget=0: refused without consulting Jev or the human
    calls = {"screen": 0, "approve": 0}
    tool = WebFetchTool(
        screen=lambda u, r: calls.__setitem__("screen", calls["screen"] + 1) or [],
        approve=lambda u, m: calls.__setitem__("approve", calls["approve"] + 1) or [u[0].url],
        client=_client(_html),
        budget=0,
    )
    assert "BUDGET" in tool.web_fetch(["https://a.example/x"], "r")
    assert calls == {"screen": 0, "approve": 0}

    # budget=1: one fetch allowed, then budget message
    tool = WebFetchTool(
        screen=lambda u, r: _scored((u[0], 0.9)),
        approve=lambda u, m: [u[0].url],
        client=_client(_html),
        budget=1,
    )
    assert "WEB_FETCHED" in tool.web_fetch(["https://a.example/x"], "r")
    assert tool.fetched == 1
    assert "BUDGET" in tool.web_fetch(["https://b.example/x"], "r")
    assert tool.fetched == 1


def test_validation_refuses_without_jev_or_prompt():
    approval_calls = []
    tool = WebFetchTool(
        screen=lambda u, r: (_ for _ in ()).throw(AssertionError("screened")),
        approve=lambda u, m: approval_calls.append(u) or [u[0].url],
        client=_client(_html),
    )
    out = tool.web_fetch(["file:///etc/passwd", "https://localhost:8000/x", "ftp://h/x"], "r")
    assert "REFUSED" in out and approval_calls == []


def test_fetch_failure_reported_and_counts():
    tool = WebFetchTool(
        screen=lambda u, r: _scored((u[0], 0.9)),
        approve=lambda u, m: [u[0].url],
        client=_client(lambda req: httpx.Response(404, text="nope")),
    )
    out = tool.web_fetch(["https://example.com/missing"], "r")
    assert "404" in out and tool.fetched == 1


def test_http_error_reported_with_reassess():
    tool = WebFetchTool(
        screen=lambda u, r: _scored((u[0], 0.9)),
        approve=lambda u, m: [u[0].url],
        client=_client(lambda req: httpx.Response(503, text="down")),
    )
    out = tool.web_fetch(["https://example.com/down"], "r")
    assert "WEB_FETCH ERROR [503]" in out
    assert "REASSESS" in out and "do NOT repeat" in out
    assert tool.fetched == 1


def test_timeout_reported_with_reassess():
    def handler(req):
        raise httpx.ReadTimeout("server stopped sending", request=req)

    tool = WebFetchTool(
        screen=lambda u, r: _scored((u[0], 0.9)),
        approve=lambda u, m: [u[0].url],
        client=_client(handler),
    )
    out = tool.web_fetch(["https://slow.example/x"], "r")
    assert "WEB_FETCH TIMEOUT" in out
    assert "REASSESS" in out and "do NOT retry" in out
    assert tool.fetched == 1


def test_network_error_reported_with_reassess():
    def handler(req):
        raise httpx.ConnectError("no route to host", request=req)

    tool = WebFetchTool(
        screen=lambda u, r: _scored((u[0], 0.9)),
        approve=lambda u, m: [u[0].url],
        client=_client(handler),
    )
    out = tool.web_fetch(["https://down.example/x"], "r")
    assert "WEB_FETCH NETWORK ERROR" in out
    assert "REASSESS" in out and "do NOT retry" in out
    assert tool.fetched == 1


def test_success_resets_network_failure_count():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom", request=req)
        return _html()

    tool = WebFetchTool(
        screen=lambda u, r: _scored((u[0], 0.9)),
        approve=lambda u, m: [u[0].url],
        client=_client(handler),
        budget=10,
        max_attempts=3,
    )
    assert "NETWORK ERROR" in tool.web_fetch(["https://a.example/x"], "r")
    assert tool.network_failures == 1
    assert "WEB_FETCHED" in tool.web_fetch(["https://a.example/x"], "r")
    assert tool.network_failures == 0


def test_consecutive_network_failures_extinguish():
    def handler(req):
        raise httpx.ConnectError("no route to host", request=req)

    tool = WebFetchTool(
        screen=lambda u, r: _scored((u[0], 0.9)),
        approve=lambda u, m: [u[0].url],
        client=_client(handler),
        budget=10,
        max_attempts=3,
    )
    assert "NETWORK ERROR" in tool.web_fetch(["https://a.example/x"], "r")
    assert tool.network_failures == 1
    assert "NETWORK ERROR" in tool.web_fetch(["https://b.example/x"], "r")
    assert "EXTINGUISHED" in tool.web_fetch(["https://c.example/x"], "r")
    assert "DISABLED" in tool.web_fetch(["https://d.example/x"], "r")  # no screen/approve/fetch


def test_fetch_returns_all_approved():
    def handler(req):
        return _html(text=f"<html><body>{req.url.path}</body></html>")

    tool = WebFetchTool(
        screen=lambda u, r: _scored(*[(url, 0.9) for url in u]),
        approve=lambda u, m: [k.url for k in u],
        client=_client(handler),
        budget=5,
    )
    out = tool.web_fetch(["https://a.example/one", "https://b.example/two"], "r")
    assert "WEB_FETCHED" in out
    assert "/one" in out and "/two" in out
    assert tool.fetched == 2
    assert tool.network_failures == 0


def _guarded_client(redirect_to: str) -> httpx.Client:
    tool = WebFetchTool(
        screen=lambda u, r: _scored((u[0], 0.9)),
        approve=lambda u, m: [u[0].url],
        client=_client(_html),
    )

    def handler(req):
        if req.url.path == "/page":
            return httpx.Response(302, headers={"location": redirect_to})
        return _html()

    return httpx.Client(
        transport=httpx.MockTransport(handler),
        timeout=1.0,
        follow_redirects=True,
        event_hooks={"request": [tool._guard_redirect]},
    )


@pytest.mark.parametrize(
    "target",
    [
        "http://127.0.0.1/steal",
        "http://localhost/steal",
        "http://192.168.1.1/steal",
        "file:///etc/passwd",
    ],
)
def test_redirect_to_private_target_is_blocked(target):
    tool = WebFetchTool(
        screen=lambda u, r: _scored((u[0], 0.9)),
        approve=lambda u, m: [u[0].url],
        client=_guarded_client(target),
    )
    out = tool.web_fetch(["https://public.example/page"], "r")
    assert "NETWORK ERROR" in out
    assert tool.fetched == 1


def test_redirect_to_public_target_is_allowed():
    tool = WebFetchTool(
        screen=lambda u, r: _scored((u[0], 0.9)),
        approve=lambda u, m: [u[0].url],
        client=_guarded_client("https://public.example/ok"),
    )
    out = tool.web_fetch(["https://a.example/page"], "r")
    assert "WEB_FETCHED" in out


def test_download_is_capped():
    big = "x" * 50_000
    tool = WebFetchTool(
        screen=lambda u, r: _scored((u[0], 0.9)),
        approve=lambda u, m: [u[0].url],
        client=_client(
            lambda req: httpx.Response(200, text=big, headers={"content-type": "text/plain"})
        ),
        settings=Settings(max_chars=100),
    )
    out = tool.web_fetch(["https://big.example/x"], "r")
    assert len(out) < 2_000
    assert "x" * 5_000 not in out
    assert tool.fetched == 1


def test_post_payload_exactly_as_submitted():
    captured = {}

    def handler(req):
        captured["body"] = req.content
        captured["method"] = req.method
        return httpx.Response(
            200, text='{"ok": true}', headers={"content-type": "application/json"}
        )

    tool = WebFetchTool(
        screen=lambda u, r: _scored((u[0], 0.9)),
        approve=lambda u, m: [u[0].url],
        client=_client(handler),
    )
    out = tool.web_fetch(
        ["https://api.example.com/search"], "search", method="POST", payload='{"q":"search"}'
    )
    assert captured["method"] == "POST"
    assert captured["body"] == b'{"q":"search"}'
    assert '"ok": true' in out
