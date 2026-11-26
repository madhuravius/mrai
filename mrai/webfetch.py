"""Jev-screened, human-approved web fetch; runs inline in the agent's tool pool."""

from __future__ import annotations

import html
import ipaddress
import re
import socket
import threading
import urllib.parse
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import httpx
from rich.console import Console
from rich.markup import escape

from .leaks import find_secret
from .settings import Settings

_TAG_RE = re.compile(r"<[^>]+>")

_REFUSED = "WEB_FETCH REFUSED: no valid public http(s) URL supplied."
_DISABLED = "WEB_FETCH DISABLED: denied 3 times this run. Stop calling web_fetch; use the local codebase only."
_UNAVAILABLE = "WEB_FETCH UNAVAILABLE: Jev could not be reached; nothing was fetched. Retry with different URLs or proceed locally."
_SECRET = (
    "WEB_FETCH REFUSED: the request carries what looks like a credential ({what}). "
    "Never put repo contents, keys or tokens in a URL or payload. Do not retry this request."
)
_LONG_QUERY = 100
_NOISE = "WEB_FETCH NOISE: Jev judged none of the candidates worth fetching. Propose different URLs or another angle."

# Failure replies are prefixed with one of these so the tool can tell an
# all-failed batch apart from a mixed one (used to extinguish after repeated
# network trouble) and so the model gets an explicit REASSESS directive.
_FAILURE_PREFIXES = ("WEB_FETCH TIMEOUT:", "WEB_FETCH ERROR", "WEB_FETCH NETWORK ERROR:")

_EXTINGUISHED = (
    "\nWEB_FETCH EXTINGUISHED: the network failed several batches in a row. "
    "Stop calling web_fetch; answer strictly from the local codebase."
)


# Sentinel returned by the terminal prompt when the developer denies everything
# ("d" / EOF): the extinction bookkeeping lives in `_deny()` only.
class _DenyAll:
    pass


_DENY_ALL = _DenyAll()


def _timeout_msg(url: str, budget: float) -> str:
    return (
        f"WEB_FETCH TIMEOUT: fetching {url} exceeded the {budget:.0f}s budget "
        "(server slow or hanging). REASSESS: do NOT retry this URL or domain; "
        "pick a lighter candidate URL or answer strictly from local repo context."
    )


def _http_error_msg(url: str, status: int) -> str:
    return (
        f"WEB_FETCH ERROR [{status}]: {url} is unusable. "
        "REASSESS: do NOT repeat this request; answer from remaining codebase "
        "evidence or choose another source."
    )


def _net_error_msg(url: str, exc: BaseException) -> str:
    return (
        f"WEB_FETCH NETWORK ERROR: {url} unreachable ({exc}). "
        "REASSESS: do NOT retry; pick a different candidate or answer from "
        "local repo context."
    )


def _resolve_and_check(host: str) -> str | None:
    """Fails open on DNS errors: `_guard_redirect` re-checks before connect."""
    try:
        infos = socket.getaddrinfo(host, None, 0, socket.SOCK_STREAM)
    except socket.gaierror:
        return None
    for _, _, _, _, sockaddr in infos:
        addr = sockaddr[0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return f"refusing non-public address {host!r} (resolves to {addr})"
    return None


def _validate_url(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"only http/https URLs are allowed, got {parsed.scheme!r}"
    host = parsed.hostname
    if not host:
        return "URL must include a host"
    if host == "localhost" or host.endswith((".local", ".internal")):
        return f"refusing non-public host {host!r}"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return _resolve_and_check(host)
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        return f"refusing non-public address {host!r}"
    return None


def _query_warning(url: str) -> str | None:
    query = urllib.parse.urlparse(url).query
    if len(query) > _LONG_QUERY:
        return f"query string is {len(query)} chars: check it carries nothing from your repo"
    return None


def _to_text(raw: str) -> str:
    text = re.sub(r"(?m)^[ \t]+", "", re.sub(_TAG_RE, " ", html.unescape(raw)))
    return re.sub(r"[ \t]{2,}", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


def _read_capped(resp, max_chars: int) -> str:
    """Drain a streaming response, stopping once `max_chars` bytes have arrived."""
    chunks = bytearray()
    for part in resp.iter_bytes():
        chunks += part
        if len(chunks) >= max_chars:
            break
    return bytes(chunks).decode(errors="replace")


@dataclass(frozen=True)
class ScoredUrl:
    url: str
    score: float


def jev_screen_urls(
    urls: Sequence[str],
    reason: str,
    key: str | None = None,
    settings: Settings | None = None,
) -> tuple[list[ScoredUrl] | None, str | None, float | None]:
    """Ask Jev (one noul per URL) how worth-fetching each candidate is.

    Returns (rankings, None, cost) on success, (None, error, None) when Jev is
    unreachable. `cost` is the screening call's spend from the decisions API.
    """
    s = settings or Settings()
    questions = {
        f"u{i}": {
            "type": "noul",
            "instructions": f"Is this URL worth fetching? URL: {url}",
            "criteria": {
                "true": "Directly relevant; likely has the needed info.",
                "false": "Irrelevant, spammy, redundant, or a search-results page.",
            },
        }
        for i, url in enumerate(urls)
    }
    try:
        resp = httpx.post(
            s.decisions_url,
            headers={"Authorization": f"Bearer {key or s.openrouter_api_key}"},
            json={"model": s.jev_model, "state": {"reason": reason}, "questions": questions},
            timeout=s.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        answers = body["answers"]
        usage = body.get("usage") or {}
        cost = float(usage["cost"]) if usage.get("cost") is not None else None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", None
    scored = [
        ScoredUrl(u, float(answers.get(f"u{i}", {}).get("noul", 0.0))) for i, u in enumerate(urls)
    ]
    return sorted(scored, key=lambda x: x.score, reverse=True), None, cost


class WebFetchTool:
    def __init__(
        self,
        *,
        screen: Callable[[list[str], str], list[ScoredUrl] | None] | None = None,
        approve: Callable[[list[ScoredUrl], dict], list[str]] | None = None,
        client: httpx.Client | None = None,
        budget: int | None = None,
        max_attempts: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        s = settings or Settings()
        self.last_error: str | None = None
        self.screen_cost: float = 0.0

        def _default_screen(urls, reason):
            scored, err, cost = jev_screen_urls(urls, reason, settings=s)
            self.last_error = err
            if cost:
                self.screen_cost += cost
            return scored

        self._screen = screen or _default_screen
        self._approve = approve
        self._timeout = httpx.Timeout(
            s.fetch_timeout,
            connect=s.connect_timeout,
            read=s.read_timeout,
            write=s.fetch_timeout,
            pool=s.connect_timeout,
        )
        self._fetch_budget = s.fetch_timeout
        self._client = client or httpx.Client(
            timeout=self._timeout,
            follow_redirects=True,
            event_hooks={"request": [self._guard_redirect]},
            transport=httpx.HTTPTransport(retries=2),
        )
        self._budget = s.web_budget if budget is None else budget
        self._max = s.max_attempts if max_attempts is None else max_attempts
        self._max_urls = s.max_urls
        self._max_recommended = s.max_recommended
        self._max_chars = s.max_chars
        self._threshold = s.web_screen_threshold
        self._user_agent = s.user_agent
        self._allowed = tuple(d.lower().strip(".") for d in s.web_allowed_domains)
        self._known_secrets = (s.openrouter_api_key,)
        self._lock = threading.Lock()  # serialize terminal prompts when invoked in parallel
        self.denials = 0
        self.fetched = 0
        self.network_failures = 0
        self.extinguished = False
        self.fetched_urls: list[str] = []

    def web_fetch(
        self,
        urls: list[str],
        reason: str,
        payload: str | None = None,
    ) -> str:
        """Fetch after Jev ranks the candidates and the developer approves the exact request."""
        if self.extinguished:
            return _DISABLED
        if leak := find_secret(payload, self._known_secrets) or next(
            (w for u in urls if (w := find_secret(u, self._known_secrets))), None
        ):
            return _SECRET.format(what=leak)
        ok = [u for u in urls if self._refusal(u) is None]
        if not ok:
            reasons = "; ".join(sorted({r for u in urls if (r := self._refusal(u))}))
            return f"{_REFUSED} ({reasons})" if reasons else _REFUSED
        with self._lock:
            if self.fetched >= self._budget:
                return f"WEB_FETCH BUDGET: {self._budget} approved fetch(es) used this run. Do not ask again."
            scored = self._screen(list(ok[: self._max_urls]), reason)
            if scored is None:
                if self.last_error:
                    Console().print(f"[dim]warning: Jev unreachable ({self.last_error})[/dim]")
                return _UNAVAILABLE
            scored = sorted(scored, key=lambda x: x.score, reverse=True)
            self._print_scores(scored)
            keepers = [x for x in scored if x.score >= self._threshold][: self._max_recommended]
            if not keepers:
                return _NOISE
            chosen = self._pick(keepers, payload, reason)
            if isinstance(chosen, _DenyAll):
                return self._deny(force=True)
            if not chosen:
                return self._deny()
            self.denials = 0
            remaining = self._budget - self.fetched
            chosen = [x.url for x in keepers if x.url in chosen][:remaining]
            results = [self._fetch(url, payload) for url in chosen]
            if results and all(r.startswith(_FAILURE_PREFIXES) for r in results):
                self.network_failures += 1
                if self.network_failures >= self._max:
                    self.extinguished = True
                    results.append(_EXTINGUISHED)
            else:
                self.network_failures = 0
            return "\n\n".join(results) or "WEB_FETCH: nothing fetched."

    def _pick(self, keepers, payload, reason) -> list[str] | _DenyAll:
        if self._approve:
            meta = {"method": "GET", "payload": payload, "reason": reason}
            return [u for u in self._approve(list(keepers), meta) if u in {x.url for x in keepers}]
        return self._terminal_prompt(keepers, payload, reason)

    def _deny(self, force: bool = False) -> str:
        self.denials += 1
        if force or self.denials >= self._max:
            self.extinguished = True
            return (
                f"WEB_FETCH EXTINGUISHED: the developer approved nothing {self._max} times "
                "in sequence. Do not call web_fetch again this run; use the local codebase only."
            )
        return (
            f"DEVELOPER APPROVED NONE (attempt {self.denials}/{self._max}). "
            "Propose alternative URLs or another angle; do not repeat rejected URLs."
        )

    def _print_scores(self, scored: list[ScoredUrl]) -> None:
        c = Console()
        c.print(f"[web_fetch] [yellow]Jev ranked[/yellow] {len(scored)} candidate(s):")
        for i, x in enumerate(scored, 1):
            mark = "[green]✓ keep[/green]" if x.score >= self._threshold else "[red]✗ drop[/red]"
            top = " [bold][top pick][/bold]" if i == 1 and x.score >= self._threshold else ""
            c.print(f"  {mark} {escape(x.url)} [dim](Jev {x.score:.2f})[/dim]{top}")

    def _terminal_prompt(self, keepers, payload, reason) -> list[str] | _DenyAll:
        c = Console()
        c.print(
            f"[yellow]web_fetch review[/yellow] (denials {self.denials}/{self._max} · fetched {self.fetched}/{self._budget})"
        )
        for i, x in enumerate(keepers, 1):
            top = " · top pick" if i == 1 else ""
            c.print(f"  [bold]{i}.[/bold] {escape(x.url)} [dim](Jev {x.score:.2f}{top})[/dim]")
            if warning := _query_warning(x.url):
                c.print(f"     [red]⚠ {warning}[/red]")
        c.print(f"  method=GET payload={escape(payload or '(none)')}\n  reason: {escape(reason)}")
        try:
            a = (
                input("Approve which URL(s)? (e.g. 1,2 — Enter=none, 'd'=deny all) ")
                .strip()
                .lower()
            )
        except EOFError:
            return _DENY_ALL
        if a == "d":
            return _DENY_ALL
        return [
            keepers[int(p) - 1].url
            for p in re.split(r"[,\s]+", a)
            if p.isdigit() and 1 <= int(p) <= len(keepers)
        ]

    def _guard_redirect(self, req: httpx.Request) -> None:
        """Reject any hop (incl. redirect targets) to a non-public URL."""
        err = self._refusal(str(req.url))
        if err:
            raise httpx.HTTPError(err)

    def _refusal(self, url: str) -> str | None:
        if err := _validate_url(url):
            return err
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        if self._allowed and not any(host == d or host.endswith("." + d) for d in self._allowed):
            return f"{host!r} is not in web_allowed_domains"
        return None

    def _fetch(self, url: str, payload: str | None) -> str:
        self.fetched_urls.append(url)
        self.fetched += 1
        try:
            with self._client.stream(
                "GET",
                url,
                headers={"User-Agent": self._user_agent},
                content=payload or None,
                timeout=self._timeout,
            ) as resp:
                if resp.status_code >= 400:
                    return _http_error_msg(url, resp.status_code)
                ctype = resp.headers.get("content-type", "unknown")
                text = _read_capped(resp, self._max_chars)
        except httpx.TimeoutException:
            return _timeout_msg(url, self._fetch_budget)
        except httpx.HTTPError as exc:
            return _net_error_msg(url, exc)
        except Exception as exc:  # unexpected: fail the fetch, not the whole tool
            return _net_error_msg(url, exc)
        body = _to_text(text) if "html" in ctype.lower() else text
        return (
            f"WEB_FETCHED {url} [HTTP {resp.status_code}]"
            f" ctype={ctype}\n\n{body[: self._max_chars]}"
        )
