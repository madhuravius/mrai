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
