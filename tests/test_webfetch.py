import socket
from unittest.mock import patch

import httpx
import pytest
from mrai import webfetch
from mrai.settings import Settings
from mrai.webfetch import (
    ScoredUrl,
    WebFetchTool,
    _resolve_and_check,
    _validate_url,
    jev_screen_urls,
)


def _dns(*addrs: str):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, 0)) for a in addrs]


@pytest.fixture(autouse=True)
def public_dns():
    """Every hostname resolves to a public IP unless a test patches DNS itself (no real lookups)."""
    with patch("mrai.webfetch.socket.getaddrinfo", return_value=_dns("93.184.216.34")):
        yield


def _html(request=None, text="<html><body><h1>Title</h1><p>Hello <b>world</b>.</p></body></html>"):
    return httpx.Response(200, text=text, headers={"content-type": "text/html"})


def _client(handler=_html, **kw) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=1.0, **kw)


def _scored(*pairs) -> list[ScoredUrl]:
    return [ScoredUrl(url=u, score=s) for u, s in pairs]


def _tool(handler=_html, *, client=None, **kw) -> WebFetchTool:
    """A tool where Jev rates every URL 0.9 and the human approves the first one."""
    kw.setdefault("screen", lambda urls, reason: _scored(*[(u, 0.9) for u in urls]))
    kw.setdefault("approve", lambda keepers, meta: [keepers[0].url])
    return WebFetchTool(client=client or _client(handler), **kw)


def _never(*args):
    pytest.fail("should not be called")


def _raises(exc):
    def handler(req):
        raise exc(request=req, message="boom")

    return handler


# --- Jev screening + human approval ------------------------------------------


def test_jev_ranks_and_filters_then_human_approves():
    seen = []

    def approve(keepers, meta):
        seen.append(([k.url for k in keepers], meta["method"]))
        return [keepers[0].url]

    tool = _tool(
        screen=lambda u, r: _scored(
            ("https://x.example/no", 0.1), ("https://y.example/keep", 0.95)
        ),
        approve=approve,
    )
    out = tool.web_fetch(["https://x.example/no", "https://y.example/keep"], "why")
    assert seen == [(["https://y.example/keep"], "GET")]  # below-threshold candidate dropped
    assert "Title" in out and "Hello world" in out
    assert (tool.fetched, tool.denials) == (1, 0)
    assert tool.fetched_urls == ["https://y.example/keep"]  # recorded for the footer


def test_human_sees_candidates_best_first():
    seen = []
    tool = _tool(
        screen=lambda u, r: _scored(("https://mid.example", 0.7), ("https://best.example", 0.98)),
        approve=lambda keepers, m: seen.append([k.url for k in keepers]) or [],
    )
    tool.web_fetch(["https://mid.example", "https://best.example"], "r")
    assert seen == [["https://best.example", "https://mid.example"]]


def test_all_below_threshold_is_noise_without_human_prompt():
    tool = _tool(screen=lambda u, r: _scored((u[0], 0.1), (u[1], 0.3)), approve=_never)
    out = tool.web_fetch(["https://a.example", "https://b.example"], "r")
    assert "NOISE" in out and tool.fetched == 0


def test_jev_unavailable_fetches_nothing():
    tool = _tool(screen=lambda u, r: None, approve=_never)
    assert "UNAVAILABLE" in tool.web_fetch(["https://a.example"], "r")
    assert tool.fetched == 0


def test_invalid_urls_refused_without_jev_or_human():
    tool = _tool(screen=_never, approve=_never)
    out = tool.web_fetch(["file:///etc/passwd", "https://localhost:8000/x", "ftp://h/x"], "r")
    assert "REFUSED" in out


def test_repeated_denials_extinguish_the_tool():
    tool = _tool(approve=lambda keepers, m: [], max_attempts=3)
    assert "APPROVED NONE" in tool.web_fetch(["https://a.example/x"], "r")
    assert tool.denials == 1
    tool.web_fetch(["https://b.example/x"], "r")
    assert "EXTINGUISHED" in tool.web_fetch(["https://c.example/x"], "r")
    assert "DISABLED" in tool.web_fetch(["https://d.example/x"], "r")  # skips screen + prompt


def test_zero_budget_refuses_before_asking_jev():
    tool = _tool(screen=_never, approve=_never, budget=0)
    assert "BUDGET" in tool.web_fetch(["https://a.example/x"], "r")


def test_budget_stops_further_fetches():
    tool = _tool(budget=1)
    assert "WEB_FETCHED" in tool.web_fetch(["https://a.example/x"], "r")
    assert "BUDGET" in tool.web_fetch(["https://b.example/x"], "r")
    assert tool.fetched == 1


def test_fetch_returns_all_approved():
    tool = _tool(
        lambda req: _html(text=f"<html><body>{req.url.path}</body></html>"),
        approve=lambda keepers, m: [k.url for k in keepers],
        budget=5,
    )
    out = tool.web_fetch(["https://a.example/one", "https://b.example/two"], "r")
    assert "/one" in out and "/two" in out
    assert tool.fetched == 2


# --- the terminal approval prompt (no `approve` callback) --------------------


@pytest.mark.parametrize(
    "typed,expected",
    [
        ("2", ["https://b.example"]),
        ("1, 2", ["https://a.example", "https://b.example"]),
        ("9 x", []),  # out-of-range and junk are ignored
        ("", []),  # Enter = approve none
    ],
)
def test_terminal_prompt_approves_typed_choices(monkeypatch, typed, expected):
    monkeypatch.setattr("builtins.input", lambda prompt: typed)
    tool = WebFetchTool(client=_client())
    keepers = _scored(("https://a.example", 0.9), ("https://b.example", 0.8))
    assert tool._terminal_prompt(keepers, None, "r") == expected


def test_terminal_prompt_d_denies_all_and_extinguishes(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "d")
    tool = WebFetchTool(screen=lambda u, r: _scored((u[0], 0.9)), client=_client())
    assert "EXTINGUISHED" in tool.web_fetch(["https://a.example"], "r")
    assert tool.fetched == 0


def test_terminal_prompt_without_stdin_denies_all(monkeypatch):
    def no_stdin(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", no_stdin)
    tool = WebFetchTool(client=_client())
    assert isinstance(
        tool._terminal_prompt(_scored(("https://a.example", 0.9)), None, "r"),
        type(webfetch._DENY_ALL),
    )


# --- jev_screen_urls (the decisions API call) --------------------------------


def _decisions(monkeypatch, response):
    """Make the decisions API return `response` (or raise it, if an exception)."""

    def post(url, **kwargs):
        if isinstance(response, Exception):
            raise response
        return httpx.Response(200, json=response, request=httpx.Request("POST", url))

    monkeypatch.setattr(webfetch.httpx, "post", post)


def test_jev_screen_ranks_best_first_and_reports_cost(monkeypatch):
    _decisions(
        monkeypatch,
        {"answers": {"u0": {"noul": 0.2}, "u1": {"noul": 0.9}}, "usage": {"cost": 0.003}},
    )
    scored, err, cost = jev_screen_urls(["https://a.example", "https://b.example"], "r", key="k")
    assert scored == _scored(("https://b.example", 0.9), ("https://a.example", 0.2))
    assert (err, cost) == (None, 0.003)


def test_jev_screen_missing_answer_scores_zero(monkeypatch):
    _decisions(monkeypatch, {"answers": {}})
    scored, err, cost = jev_screen_urls(["https://a.example"], "r", key="k")
    assert scored == _scored(("https://a.example", 0.0))
    assert cost is None


def test_jev_screen_unreachable_returns_error(monkeypatch):
    _decisions(monkeypatch, httpx.ConnectError("down"))
    assert jev_screen_urls(["https://a.example"], "r", key="k") == (
        None,
        "ConnectError: down",
        None,
    )


def test_default_screen_accumulates_cost(monkeypatch):
    _decisions(monkeypatch, {"answers": {"u0": {"noul": 0.1}}, "usage": {"cost": 0.002}})
    tool = WebFetchTool(approve=_never, client=_client(), settings=Settings(openrouter_api_key="k"))
    tool.web_fetch(["https://a.example"], "r")
    tool.web_fetch(["https://b.example"], "r")
    assert tool.screen_cost == pytest.approx(0.004)


# --- fetch failures ----------------------------------------------------------


@pytest.mark.parametrize(
    "handler,expected",
    [
        (lambda req: httpx.Response(404, text="nope"), "WEB_FETCH ERROR [404]"),
        (lambda req: httpx.Response(503, text="down"), "WEB_FETCH ERROR [503]"),
        (_raises(httpx.ReadTimeout), "WEB_FETCH TIMEOUT"),
        (_raises(httpx.ConnectError), "WEB_FETCH NETWORK ERROR"),
    ],
)
def test_fetch_failures_tell_the_model_to_reassess(handler, expected):
    tool = _tool(handler)
    out = tool.web_fetch(["https://a.example/x"], "r")
    assert expected in out and "REASSESS" in out
    assert tool.fetched == 1  # a failed fetch still spends budget


def test_unexpected_client_error_fails_the_fetch_not_the_tool():
    def handler(req):
        raise RuntimeError("bug in transport")

    assert "NETWORK ERROR" in _tool(handler).web_fetch(["https://a.example/x"], "r")


def test_success_resets_network_failure_count():
    responses = [_raises(httpx.ConnectError), _html]
    tool = _tool(lambda req: responses.pop(0)(req), budget=10)
    tool.web_fetch(["https://a.example/x"], "r")
    assert tool.network_failures == 1
    assert "WEB_FETCHED" in tool.web_fetch(["https://a.example/x"], "r")
    assert tool.network_failures == 0


def test_consecutive_network_failures_extinguish():
    tool = _tool(_raises(httpx.ConnectError), budget=10, max_attempts=3)
    tool.web_fetch(["https://a.example/x"], "r")
    tool.web_fetch(["https://b.example/x"], "r")
    assert "EXTINGUISHED" in tool.web_fetch(["https://c.example/x"], "r")
    assert "DISABLED" in tool.web_fetch(["https://d.example/x"], "r")


def test_download_is_capped():
    big = httpx.Response(200, text="x" * 50_000, headers={"content-type": "text/plain"})
    out = _tool(lambda req: big, settings=Settings(max_chars=100)).web_fetch(
        ["https://a.example"], "r"
    )
    assert len(out) < 2_000


def test_payload_sent_exactly_as_submitted_over_get():
    seen = {}

    def handler(req):
        seen.update(method=req.method, body=req.content)
        return httpx.Response(
            200, text='{"ok": true}', headers={"content-type": "application/json"}
        )

    out = _tool(handler).web_fetch(["https://api.example/search"], "r", payload='{"q":"search"}')
    assert seen == {"method": "GET", "body": b'{"q":"search"}'}
    assert '"ok": true' in out


# --- SSRF guards: redirects and DNS ------------------------------------------


def _redirecting_tool(to: str) -> WebFetchTool:
    """/page 302s to `to`; every hop goes through the tool's redirect guard."""

    def handler(req):
        if req.url.path == "/page":
            return httpx.Response(302, headers={"location": to})
        return _html()

    tool = _tool()
    tool._client = _client(
        handler, follow_redirects=True, event_hooks={"request": [tool._guard_redirect]}
    )
    return tool


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
    assert "NETWORK ERROR" in _redirecting_tool(target).web_fetch(["https://a.example/page"], "r")


def test_redirect_to_public_target_is_allowed():
    assert "WEB_FETCHED" in _redirecting_tool("https://b.example/ok").web_fetch(
        ["https://a.example/page"], "r"
    )


def test_redirect_to_dns_rebind_target_is_blocked():
    """Public URL 302 -> hostname that resolves to the cloud metadata IP."""

    def dns(host, *_):
        return _dns("169.254.169.254" if host == "rebind.example" else "93.184.216.34")

    tool = _redirecting_tool("http://rebind.example/steal")
    with patch("mrai.webfetch.socket.getaddrinfo", side_effect=dns):
        assert "NETWORK ERROR" in tool.web_fetch(["https://safe.example/page"], "r")


@pytest.mark.parametrize(
    "addrs,bad",
    [
        (["10.0.0.1"], "10.0.0.1"),  # private
        (["169.254.169.254"], "169.254.169.254"),  # cloud metadata (IMDS)
        (["169.254.1.1"], "169.254.1.1"),  # link-local
        (["127.0.0.1"], "127.0.0.1"),  # loopback
        (["8.8.8.8", "10.0.0.1", "1.1.1.1"], "10.0.0.1"),  # one bad record is enough
    ],
)
def test_hostname_resolving_to_non_public_ip_is_refused(addrs, bad):
    with patch("mrai.webfetch.socket.getaddrinfo", return_value=_dns(*addrs)):
        assert bad in _resolve_and_check("host.example")
        assert bad in _validate_url("https://host.example/admin")


def test_ipv6_loopback_is_refused():
    v6 = [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 0, 0, 0))]
    with patch("mrai.webfetch.socket.getaddrinfo", return_value=v6):
        assert "::1" in (_resolve_and_check("host.example") or "")


def test_public_hostname_is_allowed():
    assert _resolve_and_check("public.example") is None
    assert _validate_url("https://public.example") is None


def test_dns_failure_fails_open():
    with patch("mrai.webfetch.socket.getaddrinfo", side_effect=socket.gaierror("no such host")):
        assert _resolve_and_check("nx.example") is None


@pytest.mark.parametrize(
    "url", ["https:///nohost", "http://10.0.0.1/x", "http://[::1]/x", "http://printer.local/x"]
)
def test_literal_non_public_urls_are_refused(url):
    assert _validate_url(url) is not None
