from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import httpx

from .settings import Settings


@dataclass(frozen=True)
class Route:
    model: str
    confidence: float
    source: str  # "jev" | "flag" | "env" | "fallback"
    subagents: bool = False
    multi_loop: float | None = None
    jev_cost: float | None = None
    jev_error: str | None = None
    choice: str | None = None  # "simple" | "complex" when source == "jev"
    tier: str = ""  # "flash" | "pro" | "" (custom/unknown)

    def describe(self) -> str:
        label = {"flash": "flash", "pro": "pro"}.get(self.tier, self.model)
        bits = [self.source, label]
        if self.source == "jev":
            bits.append(f"conf {self.confidence:.2f}")
            if self.multi_loop is not None:
                bits.append(f"multi-loop {self.multi_loop:.2f}")
        if self.subagents:
            bits.append("subagents on")
        return f"[dim]{' · '.join(bits)}[/dim]"

    def jev_summary(self) -> list[tuple[str, float, str, bool]] | None:
        """(label, score, one-line blurb, chosen) per candidate; None if not Jev-routed."""
        if self.source != "jev" or self.choice is None:
            return None
        simple = self.confidence if self.choice == "simple" else 1 - self.confidence
        return [
            (
                "flash",
                simple,
                "One narrow read-only step: locate a file, read a snippet, one grep.",
                self.tier == "flash",
            ),
            (
                "pro",
                1 - simple,
                "Several steps or judgment: debugging, architecture, design, tradeoffs.",
                self.tier == "pro",
            ),
            (
                "explorer subagent",
                float(self.multi_loop or 0.0),
                "Wide exploration across files/subsystems is likely.",
                self.subagents,
            ),
        ]


@dataclass(frozen=True)
class JevVerdict:
    choice: str  # "simple" | "complex"
    confidence: float
    multi_loop: float
    cost: float | None


def _payload(s: Settings, query: str) -> dict:
    return {
        "model": s.jev_model,
        "state": {"query": query},
        "questions": {
            "difficulty": {
                "type": "choice",
                "instructions": "Classify the developer's task by how much reasoning it needs.",
                "criteria": {
                    "simple": "One narrow read-only step: locate one file, read a short snippet, one grep, or a definition.",
                    "complex": "Needs several steps or judgment: architecture, debugging, multi-file analysis, tradeoffs, design.",
                },
            },
            "multi_loop": {
                "type": "noul",
                "instructions": "Will answering this take more than a few search/read loops?",
                "criteria": {
                    "true": "Wide exploration: many files or subsystems, cross-referencing, multi-step investigation.",
                    "false": "One or two direct lookups; a single file or grep.",
                },
            },
        },
    }


def classify_task(
    query: str, key: str | None, settings: Settings | None = None
) -> tuple[JevVerdict | None, str | None]:
    """Jev verdict on difficulty + multi-loop; (None, error) when the call fails."""
    s = settings or Settings()
    try:
        resp = httpx.post(
            s.decisions_url,
            headers={"Authorization": f"Bearer {key}"},
            json=_payload(s, query),
            timeout=s.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        answers, usage = body["answers"], body.get("usage") or {}
        diff, ml = answers["difficulty"], answers["multi_loop"]
        cost = float(usage["cost"]) if usage.get("cost") is not None else None
        return (
            JevVerdict(
                choice=diff["choice"],
                confidence=float(diff.get("confidence", 0.0)),
                multi_loop=float(ml.get("noul", 0.0)),
                cost=cost,
            ),
            None,
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def select_final_answer(
    query: str,
    candidates: Sequence[tuple[str, str, str]],  # (topic, title, markdown)
    key: str | None = None,
    settings: Settings | None = None,
) -> tuple[int | None, list[float], float | None, str | None]:
    """Jev picks the best candidate for the original prompt.

    Returns (idx, scores, cost, error): `idx` is Jev's authoritative `best`
    choice; `scores` are per-candidate nouls (for display, an independent
    judgment from the choice).
    """
    s = settings or Settings()

    def blurb(c: tuple[str, str, str]) -> str:
        _, title, md = c
        return f"{title}: {' '.join(md.split())[:400]}"

    questions = {
        "best": {
            "type": "choice",
            "instructions": "Choose the single best final answer for the original prompt.",
            "criteria": {str(i): blurb(c) for i, c in enumerate(candidates)},
        },
        **{
            f"a{i}": {
                "type": "noul",
                "instructions": f"How good is this answer? Answer: {blurb(c)}",
                "criteria": {"true": "Good, complete answer.", "false": "Weak or wrong."},
            }
            for i, c in enumerate(candidates)
        },
    }
    try:
        resp = httpx.post(
            s.decisions_url,
            headers={"Authorization": f"Bearer {key or s.openrouter_api_key}"},
            json={"model": s.jev_model, "state": {"query": query}, "questions": questions},
            timeout=s.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        answers = body["answers"]
        usage = body.get("usage") or {}
        cost = float(usage["cost"]) if usage.get("cost") is not None else None
        idx = int(answers["best"]["choice"])
        scores = [float(answers.get(f"a{i}", {}).get("noul", 0.0)) for i in range(len(candidates))]
    except Exception as exc:
        return None, [], None, f"{type(exc).__name__}: {exc}"
    if not 0 <= idx < len(candidates):
        return None, scores, cost, f"Jev returned out-of-range choice {idx}"
    return idx, scores, cost, None


def _forced(
    model: str, source: str, subagents: bool | None, jev_error: str | None = None, tier: str = ""
) -> Route:
    return Route(model, 1.0, source, bool(subagents), jev_error=jev_error, tier=tier)


def resolve_route(
    query: str,
    flash: bool = False,
    pro: bool = False,
    model: str | None = None,
    subagents: bool | None = None,
    settings: Settings | None = None,
) -> Route:
    """Pick the model + subagent setting: explicit flags/env win, else Jev decides."""
    s = settings or Settings()
    if model:
        return _forced(model, "flag", subagents)  # custom model, no tier
    if flash:
        return _forced(s.flash_model, "flag", subagents, tier="flash")
    if pro:
        return _forced(s.pro_model, "flag", subagents, tier="pro")
    if s.model:
        return _forced(s.model, "env", subagents)

    verdict, jev_error = classify_task(query, s.openrouter_api_key, s)
    if verdict is None:
        return _forced(
            s.pro_model, "fallback", None, jev_error=jev_error, tier="pro"
        )  # fail-safe: most capable
    use_subagents = (
        subagents if subagents is not None else verdict.multi_loop >= s.multi_loop_threshold
    )
    flash_ok = verdict.choice == "simple" and verdict.confidence >= s.flash_threshold
    model = s.flash_model if flash_ok else s.pro_model
    return Route(
        model,
        verdict.confidence,
        "jev",
        use_subagents,
        verdict.multi_loop,
        verdict.cost,
        choice=verdict.choice,
        tier="flash" if flash_ok else "pro",
    )
