"""FastAPI webhook tests for server.py."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

import server


# ── Fixtures ──────────────────────────────────────────────────────────


class _StubQM:
    def __init__(self):
        self.enqueued = []
        self.accept_next = True
        self._shutting_down = False

    async def enqueue_pr(self, repo, pr):
        self.enqueued.append((repo, pr))
        return self.accept_next

    async def get_status(self):
        return {"queues": {}, "stats": {"total_queued": len(self.enqueued)}, "total_active_repos": 0}

    async def shutdown(self):
        self._shutting_down = True


@pytest.fixture
def stub_queue(monkeypatch):
    """Replace the queue singleton with an in-memory stub."""
    s = _StubQM()
    monkeypatch.setattr(server, "queue_manager", s)
    return s


@pytest.fixture
def no_secret(monkeypatch):
    """Bypass signature checks."""
    monkeypatch.setattr(server, "WEBHOOK_SECRET", None)


@pytest.fixture
def stub_refresh(monkeypatch):
    """Replace refresh_repo_index so BackgroundTasks doesn't try to clone real repos."""
    called: list[str] = []
    monkeypatch.setattr(server, "refresh_repo_index", lambda repo: called.append(repo))
    return called


@pytest.fixture
def client(stub_queue, no_secret, stub_refresh):
    return TestClient(server.app)


# ── Health + status ───────────────────────────────────────────────────


def test_health_check(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "running"


def test_status_endpoint(client):
    r = client.get("/status")
    assert r.status_code == 200
    data = r.json()
    assert "queues" in data
    assert "stats" in data


# ── Signature ─────────────────────────────────────────────────────────


def test_webhook_invalid_signature(monkeypatch, stub_queue):
    monkeypatch.setattr(server, "WEBHOOK_SECRET", "secret_xyz")
    c = TestClient(server.app)
    r = c.post(
        "/webhook",
        json={},
        headers={"X-GitHub-Event": "ping", "X-Hub-Signature-256": "sha256=nope"},
    )
    assert r.status_code == 401


def test_webhook_valid_signature_accepted(monkeypatch, stub_queue):
    monkeypatch.setattr(server, "WEBHOOK_SECRET", "secret_xyz")
    payload = json.dumps({}).encode()
    sig = "sha256=" + hmac.new(b"secret_xyz", payload, hashlib.sha256).hexdigest()
    c = TestClient(server.app)
    r = c.post(
        "/webhook",
        data=payload,
        headers={
            "X-GitHub-Event": "ping",
            "X-Hub-Signature-256": sig,
            "Content-Type": "application/json",
        },
    )
    assert r.status_code == 200
    assert r.json()["status"] == "pong"


# ── ping ──────────────────────────────────────────────────────────────


def test_webhook_ping_event(client):
    r = client.post("/webhook", json={}, headers={"X-GitHub-Event": "ping"})
    assert r.status_code == 200
    assert r.json()["status"] == "pong"


# ── Pull-request events ───────────────────────────────────────────────


def _pr(action, *, draft=False, state="open", num=42, repo="o/r"):
    return {
        "action": action,
        "repository": {"full_name": repo},
        "pull_request": {"number": num, "state": state, "draft": draft},
    }


def test_webhook_pr_opened(client, stub_queue):
    r = client.post("/webhook", json=_pr("opened"), headers={"X-GitHub-Event": "pull_request"})
    assert r.status_code == 200
    assert r.json()["status"] == "accepted"
    assert stub_queue.enqueued == [("o/r", 42)]


def test_webhook_pr_synchronize(client, stub_queue):
    r = client.post("/webhook", json=_pr("synchronize"), headers={"X-GitHub-Event": "pull_request"})
    assert r.json()["status"] == "accepted"


def test_webhook_pr_reopened(client, stub_queue):
    r = client.post("/webhook", json=_pr("reopened"), headers={"X-GitHub-Event": "pull_request"})
    assert r.json()["status"] == "accepted"


def test_webhook_pr_closed_ignored(client, stub_queue):
    r = client.post("/webhook", json=_pr("closed"), headers={"X-GitHub-Event": "pull_request"})
    assert r.json()["status"] == "ignored"
    assert stub_queue.enqueued == []


def test_webhook_pr_draft_ignored(client, stub_queue):
    r = client.post("/webhook", json=_pr("opened", draft=True), headers={"X-GitHub-Event": "pull_request"})
    body = r.json()
    assert body["status"] == "ignored"
    assert "draft" in body["reason"].lower()


def test_webhook_pr_queue_full(client, stub_queue):
    stub_queue.accept_next = False
    r = client.post("/webhook", json=_pr("opened"), headers={"X-GitHub-Event": "pull_request"})
    assert r.json()["status"] == "rejected"


def test_webhook_missing_repository_field(client):
    r = client.post("/webhook", json={"action": "opened"}, headers={"X-GitHub-Event": "pull_request"})
    assert r.status_code == 400


def test_webhook_non_int_pr_number(client):
    r = client.post(
        "/webhook",
        json={"action": "opened", "repository": {"full_name": "o/r"},
              "pull_request": {"number": "x"}},
        headers={"X-GitHub-Event": "pull_request"},
    )
    assert r.status_code == 400


# ── Push events ───────────────────────────────────────────────────────


def _push(ref="refs/heads/main", *, default_branch="main", files=("src/foo.py",), commits_count=1):
    return {
        "ref": ref,
        "repository": {"full_name": "o/r", "default_branch": default_branch},
        "commits": [{"added": [], "modified": list(files), "removed": []}] * commits_count,
    }


def test_webhook_push_to_main_schedules_refresh(client, stub_refresh):
    r = client.post("/webhook", json=_push(), headers={"X-GitHub-Event": "push"})
    assert r.json()["action"] == "index_refresh_scheduled"
    assert "o/r" in stub_refresh


def test_webhook_push_to_branch_ignored(client):
    r = client.post(
        "/webhook",
        json=_push(ref="refs/heads/feature-x"),
        headers={"X-GitHub-Event": "push"},
    )
    assert r.json()["status"] == "ignored"


def test_webhook_push_no_code_files(client):
    r = client.post(
        "/webhook",
        json=_push(files=("README.md", "docs/notes.txt")),
        headers={"X-GitHub-Event": "push"},
    )
    body = r.json()
    assert body["status"] == "ignored"
    assert "code" in body["reason"].lower()


def test_webhook_push_zero_commits(client):
    r = client.post(
        "/webhook",
        json={"ref": "refs/heads/main", "repository": {"full_name": "o/r", "default_branch": "main"}, "commits": []},
        headers={"X-GitHub-Event": "push"},
    )
    assert r.json()["status"] == "ignored"


def test_webhook_push_master_default_branch(client):
    r = client.post(
        "/webhook",
        json=_push(ref="refs/heads/master", default_branch="master"),
        headers={"X-GitHub-Event": "push"},
    )
    assert r.json()["action"] == "index_refresh_scheduled"


def test_webhook_unknown_event_ignored(client):
    r = client.post("/webhook", json={}, headers={"X-GitHub-Event": "release"})
    assert r.json()["status"] == "ignored"


# ── Installation events ───────────────────────────────────────────────


def _installation(action, *, install_id=42, repos=None):
    """Build an installation webhook payload."""
    body = {
        "action": action,
        "installation": {"id": install_id},
    }
    if repos is not None:
        body["repositories"] = repos
    return body


def test_webhook_installation_created_schedules_indexing(client, stub_refresh):
    r = client.post(
        "/webhook",
        json=_installation("created", install_id=42, repos=[
            {"full_name": "owner/repoA"},
            {"full_name": "owner/repoB"},
        ]),
        headers={"X-GitHub-Event": "installation"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "accepted"
    assert body["action"] == "installation_recorded"
    assert body["count"] == 2
    assert "owner/repoA" in stub_refresh and "owner/repoB" in stub_refresh


def test_webhook_installation_created_empty_repos(client, stub_refresh):
    r = client.post(
        "/webhook",
        json=_installation("created", repos=[]),
        headers={"X-GitHub-Event": "installation"},
    )
    body = r.json()
    assert body["status"] == "accepted"
    assert body["action"] == "no_repos_to_index"
    assert stub_refresh == []


def test_webhook_installation_created_missing_repos_field(client, stub_refresh):
    r = client.post(
        "/webhook",
        json=_installation("created"),
        headers={"X-GitHub-Event": "installation"},
    )
    body = r.json()
    assert body["action"] == "no_repos_to_index"


def test_webhook_installation_deleted_invalidates(client, monkeypatch):
    invalidations = []
    monkeypatch.setattr(server, "invalidate_cache", lambda r: invalidations.append(r))
    r = client.post(
        "/webhook",
        json=_installation("deleted", install_id=42, repos=[
            {"full_name": "owner/repoA"},
            {"full_name": "owner/repoB"},
        ]),
        headers={"X-GitHub-Event": "installation"},
    )
    body = r.json()
    assert body["status"] == "accepted"
    assert body["action"] == "installation_removed"
    assert body["count"] == 2
    assert sorted(invalidations) == ["owner/repoA", "owner/repoB"]


def test_webhook_installation_missing_installation_field(client):
    r = client.post(
        "/webhook",
        json={"action": "created"},
        headers={"X-GitHub-Event": "installation"},
    )
    assert r.status_code == 400


def test_webhook_installation_invalid_id(client):
    r = client.post(
        "/webhook",
        json={"action": "created", "installation": {"id": "not_an_int"}},
        headers={"X-GitHub-Event": "installation"},
    )
    assert r.status_code == 400


def test_webhook_installation_unknown_action_ignored(client):
    r = client.post(
        "/webhook",
        json=_installation("suspend"),
        headers={"X-GitHub-Event": "installation"},
    )
    assert r.json()["status"] == "ignored"


def test_webhook_pr_event_registers_installation(client, stub_queue):
    """When a PR webhook carries an ``installation`` field, cache repo→install."""
    import github_app
    github_app.forget_repo_installation("o/r")  # ensure clean slate

    payload = _pr("opened")
    payload["installation"] = {"id": 7777}
    r = client.post(
        "/webhook",
        json=payload,
        headers={"X-GitHub-Event": "pull_request"},
    )
    assert r.json()["status"] == "accepted"
    assert github_app.get_installation_for_repo("o/r") == 7777
