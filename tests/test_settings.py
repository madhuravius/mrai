import hashlib
from pathlib import Path

from mrai.settings import Settings, default_mrai_dir, mrai_home, workspace_key


def test_workspace_key_is_slug_plus_path_hash():
    path = "/home/u/work/My Repo"
    expected_hash = hashlib.sha256(path.encode()).hexdigest()[:8]
    assert workspace_key(Path(path)) == f"my-repo-{expected_hash}"


def test_workspace_key_edge_names():
    assert workspace_key(Path("/")).startswith("root-")
    long = workspace_key(Path("/x/" + "a" * 60))
    assert long.split("-")[0] == "a" * 40


def test_default_mrai_dir_lives_under_mrai_home(tmp_path):
    assert mrai_home() == tmp_path / "mrai-home"  # conftest isolation
    d = default_mrai_dir(Path("/w/api"))
    assert d.parent == tmp_path / "mrai-home" / "workspaces"


def test_yaml_layering_repo_overrides_shared(tmp_path, monkeypatch):
    home = tmp_path / "mrai-home"
    home.mkdir()
    (home / "mrai.yml").write_text("max_history: 7\ntool_timeout: 11\n")
    repo = tmp_path / "repo"
    (repo / ".mrai").mkdir(parents=True)
    (repo / ".mrai" / "mrai.yml").write_text("max_history: 3\n")
    monkeypatch.chdir(repo)
    s = Settings()
    assert s.max_history == 3  # repo wins
    assert s.tool_timeout == 11  # shared still applies


def test_yaml_path_that_is_a_directory_is_ignored(tmp_path, monkeypatch):
    # docker turns a missing bind source into a directory; that must not break config
    (tmp_path / "mrai-home" / "mrai.yml").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert Settings().max_history == 20
