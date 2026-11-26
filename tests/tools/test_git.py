import pytest
from mrai.tools.git import GitTools
from tests.conftest import _git


def test_status_reports_modified_and_untracked(git_repo):
    (git_repo / "a.txt").write_text("two\n")
    (git_repo / "new.txt").write_text("x\n")
    out = GitTools(git_repo).git_status()
    assert "M a.txt" in out  # porcelain: " M a.txt"
    assert "?? new.txt" in out


def test_diff_unstaged_and_path_filter(git_repo):
    (git_repo / "a.txt").write_text("two\n")
    tools = GitTools(git_repo)
    assert "+two" in tools.git_diff()
    assert "+two" in tools.git_diff("a.txt")  # `-- a.txt` honored


def test_diff_staged_uses_cached(git_repo):
    (git_repo / "a.txt").write_text("two\n")
    _git(git_repo, "add", "a.txt")
    assert "+two" in GitTools(git_repo).git_diff(staged=True)


def test_compare_against_branch_and_path_filter(git_repo):
    (git_repo / "a.txt").write_text("two\n")  # unstaged vs HEAD == main
    (git_repo / "b.txt").write_text("x\n")  # untracked
    tools = GitTools(git_repo)
    assert "+two" in tools.git_compare("main")
    assert "b.txt" not in tools.git_compare("main")  # untracked isn't diffed
    limited = tools.git_compare("main", "a.txt")
    assert "+two" in limited and "b.txt" not in limited
    assert tools.git_compare("main", "b.txt") == "(no output)"


def test_compare_rejects_unknown_ref_and_injection(git_repo):
    tools = GitTools(git_repo)
    assert tools.git_compare("nope") == "git tool error: unknown ref 'nope'"
    assert tools.git_compare("main; rm -rf /") == "git tool error: unknown ref 'main; rm -rf /'"
    assert tools.git_compare("") == "(no output)"  # "" -> HEAD, clean tree


def test_known_refs_short_full_and_remote_forms(git_repo):
    _git(git_repo, "branch", "feature/x")
    _git(git_repo, "tag", "v1")
    _git(git_repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    refs = GitTools(git_repo)._known_refs()
    assert {
        "HEAD",
        "main",
        "refs/heads/main",
        "feature/x",
        "refs/heads/feature/x",
        "v1",
        "refs/tags/v1",
        "origin/main",
        "refs/remotes/origin/main",
    } <= refs


def test_run_captures_failure(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("no git")

    monkeypatch.setattr("mrai.tools.git.safe_run", boom)
    assert GitTools(tmp_path).git_status() == "git tool error: OSError: no git"


@pytest.fixture
def hostile_repo(git_repo):
    (git_repo / ".gitattributes").write_text("*.txt filter=evil diff=evil\n")
    _git(git_repo, "add", ".gitattributes")
    for key, cmd in {
        "core.fsmonitor": "touch PWNED_fsmonitor; false",
        "diff.external": "touch PWNED_external",
        "diff.evil.textconv": "touch PWNED_textconv; cat",
        "filter.evil.clean": "touch PWNED_clean; cat",
        "filter.evil.process": "touch PWNED_process",
    }.items():
        _git(git_repo, "config", key, cmd)
    (git_repo / "a.txt").write_text("two\n")
    return git_repo


def test_hostile_repo_config_runs_nothing(hostile_repo):
    tools = GitTools(hostile_repo)
    assert "a.txt" in tools.git_status()
    assert "+two" in tools.git_diff()
    assert "+two" in tools.git_compare("main")
    assert list(hostile_repo.glob("PWNED*")) == []


@pytest.fixture
def repo_with_secrets(git_repo):
    (git_repo / ".env").write_text("TOKEN=old\n")
    (git_repo / "config").mkdir()
    (git_repo / "config" / "secrets.yml").write_text("k: old\n")
    _git(git_repo, "add", "-A")
    _git(git_repo, "-c", "user.email=a@b.c", "-c", "user.name=t", "commit", "-qm", "secrets")
    (git_repo / ".env").write_text("TOKEN=hunter2\n")
    (git_repo / "config" / "secrets.yml").write_text("k: hunter2\n")
    (git_repo / "a.txt").write_text("two\n")
    return git_repo


def test_diffs_leave_out_denied_files(repo_with_secrets):
    tools = GitTools(repo_with_secrets)
    for out in (tools.git_diff(), tools.git_compare("main")):
        assert "+two" in out
        assert "hunter2" not in out


def test_staged_diff_leaves_out_denied_files(repo_with_secrets):
    _git(repo_with_secrets, "add", "-A")
    out = GitTools(repo_with_secrets).git_diff(staged=True)
    assert "+two" in out and "hunter2" not in out


@pytest.mark.parametrize("path", [".env", "config/secrets.yml"])
def test_asking_for_a_denied_path_is_refused(repo_with_secrets, path):
    assert "denied pattern" in GitTools(repo_with_secrets).git_diff(path)


def test_asking_for_a_folder_still_leaves_out_its_denied_files(repo_with_secrets):
    assert "hunter2" not in GitTools(repo_with_secrets).git_diff("config")


def test_extra_denied_patterns_apply(repo_with_secrets):
    out = GitTools(repo_with_secrets, denied_patterns=["**/a.txt"]).git_diff()
    assert "+two" not in out
    assert "hunter2" in out


@pytest.mark.parametrize("path", [":(top)", ":!a.txt", "../x", "/etc/passwd"])
def test_unsafe_paths_are_refused(git_repo, path):
    assert GitTools(git_repo).git_diff(path).startswith("git tool error")
