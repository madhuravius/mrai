from unittest.mock import Mock

import pytest
from mrai import router
from mrai.settings import DEFAULT_FLASH_MODEL as FLASH_MODEL
from mrai.settings import DEFAULT_PRO_MODEL as PRO_MODEL


def _fake_post(choice, confidence, multi_loop=0.5, cost=0.00002):
    def post(*args, **kwargs):
        resp = Mock()
        resp.json.return_value = {
            "answers": {
                "difficulty": {"choice": choice, "confidence": confidence},
                "multi_loop": {"noul": multi_loop},
            },
            "usage": {"cost": cost},
        }
        return resp

    return post


def _fail_post(*args, **kwargs):
    raise AssertionError("network call should not happen")


@pytest.mark.parametrize(
    "kwargs,model,source",
    [
        ({"model": "some/model"}, "some/model", "flag"),
        ({"flash": True}, FLASH_MODEL, "flag"),
        ({"pro": True}, PRO_MODEL, "flag"),
    ],
)
def test_flags_skip_jev(monkeypatch, kwargs, model, source):
    monkeypatch.setattr(router.httpx, "post", _fail_post)
    r = router.resolve_route("q", **kwargs)
    assert (r.model, r.source) == (model, source)


def test_env_model_skips_jev(monkeypatch):
    monkeypatch.setenv("MRAI_MODEL", "env/model")
    monkeypatch.setattr(router.httpx, "post", _fail_post)
    r = router.resolve_route("q")
    assert (r.model, r.source) == ("env/model", "env")


def test_forced_subagents_override(monkeypatch):
    monkeypatch.setattr(router.httpx, "post", _fail_post)
    assert router.resolve_route("q", flash=True, subagents=True).subagents is True
    monkeypatch.setattr(router.httpx, "post", _fake_post("simple", 0.95, multi_loop=0.9))
    assert router.resolve_route("q", subagents=False).subagents is False


@pytest.mark.parametrize(
    "choice,confidence,multi_loop,expected,subagents",
    [
        ("simple", 0.95, 0.2, FLASH_MODEL, False),  # simple + confident → flash
        ("simple", 0.5, 0.2, FLASH_MODEL, False),
        ("complex", 0.6, 0.2, FLASH_MODEL, False),
        ("complex", 0.7, 0.3, PRO_MODEL, False),
        ("complex", 0.9, 0.9, PRO_MODEL, True),  # multi-loop → subagents on
    ],
)
def test_jev_verdict_routes(monkeypatch, choice, confidence, multi_loop, expected, subagents):
    monkeypatch.setattr(router.httpx, "post", _fake_post(choice, confidence, multi_loop))
    r = router.resolve_route("q")
    assert (r.model, r.subagents, r.source, r.jev_cost) == (expected, subagents, "jev", 0.00002)


def test_multi_loop_boundary(monkeypatch):
    monkeypatch.setattr(router.httpx, "post", _fake_post("complex", 0.9, 0.64))
    assert router.resolve_route("q").subagents is False
    monkeypatch.setattr(router.httpx, "post", _fake_post("complex", 0.9, 0.65))
    assert router.resolve_route("q").subagents is True


def test_jev_failure_falls_back_to_pro(monkeypatch):
    monkeypatch.setattr(
        router.httpx, "post", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("jev down"))
    )
    r = router.resolve_route("anything")
    assert (r.model, r.source, r.subagents) == (PRO_MODEL, "fallback", False)


def test_jev_summary_marks_chosen_rows(monkeypatch):
    monkeypatch.setattr(router.httpx, "post", _fake_post("simple", 0.9, multi_loop=0.85))
    r = router.resolve_route("q")
    rows = r.jev_summary()
    assert rows is not None and len(rows) == 3
    by = {label: (score, chosen) for label, score, _blurb, chosen in rows}
    assert set(by) == {"flash", "pro", "explorer subagent"}
    assert r.choice == "simple"
    # simple -> flash, multi-loop -> subagents: both chosen
    assert [label for label, score, _blurb, chosen in rows if chosen] == [
        "flash",
        "explorer subagent",
    ]
    assert by["flash"][0] == pytest.approx(0.9)
    assert by["pro"][0] == pytest.approx(0.1)


def test_jev_summary_complex_favors_pro(monkeypatch):
    monkeypatch.setattr(router.httpx, "post", _fake_post("complex", 0.8, multi_loop=0.2))
    r = router.resolve_route("q")
    by = {label: (score, chosen) for label, score, _blurb, chosen in (r.jev_summary() or [])}
    assert by["pro"] == (pytest.approx(0.8), True)
    assert by["flash"] == (pytest.approx(0.2), False)
    assert by["explorer subagent"][1] is False  # low multi-loop


def test_jev_summary_none_without_jev_route():
    assert router.resolve_route("q", flash=True).jev_summary() is None
    assert router.resolve_route("q", model="x/y").jev_summary() is None


def _pick_post(choice, cost=0.00002, scores=(0.8, 0.9, 0.7)):
    def post(*args, **kwargs):
        resp = Mock()
        resp.json.return_value = {
            "answers": {
                "best": {"choice": str(choice)},
                **{f"a{i}": {"noul": s} for i, s in enumerate(scores)},
            },
            "usage": {"cost": cost},
        }
        return resp

    return post


CANDIDATES = [("a", "A", "one"), ("b", "B", "two"), ("c", "C", "three")]


def test_select_final_answer_picks_index(monkeypatch):
    monkeypatch.setattr(router.httpx, "post", _pick_post(1))
    idx, scores, cost, err = router.select_final_answer("q", CANDIDATES)
    assert (idx, cost, err) == (1, 0.00002, None)
    assert scores == [0.8, 0.9, 0.7]


def test_select_final_answer_failure(monkeypatch):
    monkeypatch.setattr(
        router.httpx, "post", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("jev down"))
    )
    idx, scores, cost, err = router.select_final_answer("q", CANDIDATES)
    assert idx is None and scores == [] and cost is None and err is not None


def test_select_final_answer_out_of_range(monkeypatch):
    monkeypatch.setattr(router.httpx, "post", _pick_post(5))
    idx, scores, cost, err = router.select_final_answer("q", CANDIDATES)
    assert idx is None and scores == [0.8, 0.9, 0.7] and cost == 0.00002
    assert err is not None and "out-of-range" in err


def test_payload_classifies_by_scope_not_difficulty():
    from mrai.settings import Settings

    crit = router._payload(Settings(), "q")["questions"]["difficulty"]["criteria"]
    assert "explaining what code does" in crit["simple"]
    assert "debugging" in crit["complex"] and "debugging" not in crit["simple"]
