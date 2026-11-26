import hashlib
from pathlib import Path

import pytest
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
    assert mrai_home() == tmp_path / "mrai-home"
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
    assert s.max_history == 3
    assert s.tool_timeout == 11


def test_yaml_path_that_is_a_directory_is_ignored(tmp_path, monkeypatch):
    # docker turns a missing bind source into a directory; that must not break config
    (tmp_path / "mrai-home" / "mrai.yml").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert Settings().max_history == 20


PROFILES = """\
profile: openrouter
max_history: 9
profiles:
  openrouter:
    provider: openrouter
    pro_model: deepseek/x
  anthropic:
    provider: anthropic
    api_key_env: TEST_ANTHROPIC_KEY
    jev_api_key_env: TEST_OPENROUTER_KEY
    pro_model: claude-opus-5-5
"""


def _with_profiles(tmp_path, monkeypatch, text=PROFILES):
    home = tmp_path / "mrai-home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "mrai.yml").write_text(text)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MRAI_PROFILE", raising=False)


def test_profile_from_yaml_overlays_top_level(tmp_path, monkeypatch):
    _with_profiles(tmp_path, monkeypatch)
    s = Settings()
    assert (s.profile, s.provider, s.pro_model, s.max_history) == (
        "openrouter",
        "openrouter",
        "deepseek/x",
        9,
    )


def test_env_switches_profile_and_keys_come_from_named_env(tmp_path, monkeypatch):
    _with_profiles(tmp_path, monkeypatch)
    monkeypatch.setenv("MRAI_PROFILE", "anthropic")
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "sk-ant")
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "sk-or")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    s = Settings()
    assert (s.provider, s.pro_model, s.max_history) == ("anthropic", "claude-opus-5-5", 9)
    assert s.openrouter_api_key == "sk-ant" and s.jev_api_key == "sk-or"


def test_named_key_env_is_read_from_dotenv(tmp_path, monkeypatch):
    _with_profiles(tmp_path, monkeypatch)
    monkeypatch.setenv("MRAI_PROFILE", "anthropic")
    monkeypatch.delenv("TEST_ANTHROPIC_KEY", raising=False)
    (tmp_path / ".env").write_text("TEST_ANTHROPIC_KEY=from-dotenv\n")
    assert Settings().openrouter_api_key == "from-dotenv"


def test_jev_key_defaults_to_model_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    assert Settings().jev_api_key == "sk-or"


def test_unknown_profile_is_a_clear_error(tmp_path, monkeypatch):
    _with_profiles(tmp_path, monkeypatch)
    monkeypatch.setenv("MRAI_PROFILE", "nope")
    with pytest.raises(ValueError, match="unknown mrai profile 'nope'.*openrouter, anthropic"):
        Settings()


def test_example_config_profiles_all_load(tmp_path, monkeypatch):
    example = (Path(__file__).parent.parent / "mrai.example.yml").read_text()
    _with_profiles(tmp_path, monkeypatch, example)
    for name in ("openrouter", "anthropic", "gateway"):
        monkeypatch.setenv("MRAI_PROFILE", name)
        assert Settings().profile == name
