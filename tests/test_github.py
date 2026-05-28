"""Unit tests for github_client.py with PyGithub mocked."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import github_client


def _make_fake_file(filename="foo.py", status="modified", patch="+x = 1", additions=1, deletions=0):
    return SimpleNamespace(
        filename=filename, status=status, patch=patch,
        additions=additions, deletions=deletions,
    )


# ── get_pr_diff ───────────────────────────────────────────────────────


def test_get_pr_diff_returns_dicts_with_expected_keys():
    files = [_make_fake_file("a.py"), _make_fake_file("b.py", additions=3)]
    pr = MagicMock()
    pr.get_files.return_value = files

    out = github_client.get_pr_diff(pr)
    assert len(out) == 2
    for entry in out:
        for k in ("filename", "status", "patch", "additions", "deletions"):
            assert k in entry


def test_get_pr_diff_empty_list():
    pr = MagicMock()
    pr.get_files.return_value = []
    assert github_client.get_pr_diff(pr) == []


# ── get_file_content ──────────────────────────────────────────────────


def test_get_file_content_decodes_utf8():
    repo = MagicMock()
    content = MagicMock()
    content.decoded_content = "hello world".encode("utf-8")
    repo.get_contents.return_value = content
    pr = MagicMock()
    pr.head.sha = "abc123"

    out = github_client.get_file_content(repo, "x.py", pr)
    assert out == "hello world"


def test_get_file_content_returns_none_on_error():
    repo = MagicMock()
    repo.get_contents.side_effect = RuntimeError("404 not found")
    pr = MagicMock()
    pr.head.sha = "abc"
    assert github_client.get_file_content(repo, "missing.py", pr) is None


# ── post_inline_comments ──────────────────────────────────────────────


def test_post_inline_comments_counts_successes():
    pr = MagicMock()
    pr.get_commits.return_value = [MagicMock(name="c1"), MagicMock(name="c2")]
    pr.create_review_comment.return_value = None
    repo = MagicMock()
    comments = [
        {"line": 3, "severity": "warning", "comment": "x"},
        {"line": 7, "severity": "critical", "comment": "y"},
    ]
    posted = github_client.post_inline_comments(pr, repo, comments, "file.py")
    assert posted == 2
    assert pr.create_review_comment.call_count == 2


def test_post_inline_comments_swallows_per_comment_errors():
    pr = MagicMock()
    pr.get_commits.return_value = [MagicMock()]
    # First call succeeds, second raises
    pr.create_review_comment.side_effect = [None, RuntimeError("422 line not in diff")]
    repo = MagicMock()
    comments = [
        {"line": 1, "severity": "warning", "comment": "a"},
        {"line": 99, "severity": "warning", "comment": "b"},
    ]
    posted = github_client.post_inline_comments(pr, repo, comments, "file.py")
    assert posted == 1


# ── post_summary_and_verdict ──────────────────────────────────────────


def test_post_summary_returns_approve_when_no_critical():
    pr = MagicMock()
    pr.create_issue_comment.return_value = None
    verdict = github_client.post_summary_and_verdict(pr, [
        {"line": 1, "severity": "warning", "comment": "x"},
        {"line": 2, "severity": "suggestion", "comment": "y"},
    ])
    assert verdict == "APPROVE"
    pr.create_issue_comment.assert_called_once()


def test_post_summary_returns_request_changes_on_critical():
    pr = MagicMock()
    pr.create_issue_comment.return_value = None
    verdict = github_client.post_summary_and_verdict(pr, [
        {"line": 1, "severity": "critical", "comment": "boom"},
    ])
    assert verdict == "REQUEST_CHANGES"


def test_post_summary_handles_empty_comments():
    pr = MagicMock()
    pr.create_issue_comment.return_value = None
    verdict = github_client.post_summary_and_verdict(pr, [])
    assert verdict == "APPROVE"


# ── get_github_client dual auth ───────────────────────────────────────


def test_get_github_client_pat_default(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_pat")
    monkeypatch.delenv("GITHUB_APP_ID", raising=False)
    monkeypatch.delenv("GITHUB_APP_PRIVATE_KEY", raising=False)
    # No installation_id → must use PAT path, no exception
    client = github_client.get_github_client()
    assert client is not None


def test_get_github_client_app_when_id_provided(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_pat")
    monkeypatch.setenv("GITHUB_APP_ID", "12345")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", "fake_key")

    captured = {"installation_id": None}

    def fake_app_client(installation_id):
        captured["installation_id"] = installation_id
        return MagicMock(name="app_client")

    import github_app
    monkeypatch.setattr(github_app, "get_github_client_for_installation", fake_app_client)

    client = github_client.get_github_client(installation_id=99)
    assert captured["installation_id"] == 99
    assert client is not None


def test_get_github_client_falls_back_to_pat_when_app_fails(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_pat")
    monkeypatch.setenv("GITHUB_APP_ID", "12345")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", "fake_key")

    import github_app
    def boom(_id):
        raise RuntimeError("App auth broke")
    monkeypatch.setattr(github_app, "get_github_client_for_installation", boom)

    # App auth fails → fall back to PAT instead of raising
    client = github_client.get_github_client(installation_id=99)
    assert client is not None


def test_get_github_client_raises_when_no_auth_configured(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_APP_ID", raising=False)
    monkeypatch.delenv("GITHUB_APP_PRIVATE_KEY", raising=False)
    with pytest.raises(ValueError, match="authentication"):
        github_client.get_github_client()


def test_get_pr_details_passes_installation_id(monkeypatch):
    """get_pr_details should forward installation_id to get_github_client."""
    captured = {"installation_id": "unset"}

    def fake_client(installation_id=None):
        captured["installation_id"] = installation_id
        g = MagicMock()
        g.get_repo.return_value.get_pull.return_value = MagicMock(
            title="T", user=MagicMock(login="u"), changed_files=1,
        )
        return g

    monkeypatch.setattr(github_client, "get_github_client", fake_client)
    github_client.get_pr_details("o/r", 1, installation_id=42)
    assert captured["installation_id"] == 42

    # Default path: installation_id stays None
    github_client.get_pr_details("o/r", 1)
    assert captured["installation_id"] is None
