"""Shared pytest fixtures.

Module-level side effects (env-var defaults, CHROMA_PATH isolation) run
*before* pytest imports any test files, so test modules that transitively
import ``llm_reviewer`` (and through it ``rag``) don't blow up on missing
real credentials.
"""

from __future__ import annotations

import os
import sys
import tempfile

# Ensure dummy creds are present BEFORE any test module imports the agent code.
# Groq SDK raises at construction if GROQ_API_KEY is unset; we never make a
# real network call in unit tests because we mock the client.
os.environ.setdefault("GROQ_API_KEY", "test_dummy_groq_key")
os.environ.setdefault("GITHUB_TOKEN", "test_dummy_github_token")
os.environ.setdefault("WEBHOOK_SECRET", "")

# Isolate ChromaDB storage to a per-process temp dir so tests never collide
# with the developer's real ``./chroma_db`` or with each other across runs.
_TEST_CHROMA_DIR = tempfile.mkdtemp(prefix="cra_test_chroma_")
os.environ["CHROMA_PATH"] = _TEST_CHROMA_DIR

# Allow tests to import top-level modules (rag, agent, etc.) directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure ``flake8``/``bandit`` subprocess invocations from linter.py can find
# the binaries. When pytest is run via ``venv/bin/python -m pytest`` the venv
# bin dir isn't always on PATH; prepending ``dirname(sys.executable)`` makes
# the linter tests portable across shells.
_PY_BIN_DIR = os.path.dirname(os.path.abspath(sys.executable))
if _PY_BIN_DIR not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _PY_BIN_DIR + os.pathsep + os.environ.get("PATH", "")

import pytest


@pytest.fixture
def temp_python_file():
    """Factory fixture: writes content to a temp file and returns its path."""

    def _make_file(content: str, filename: str = "test.py") -> str:
        temp_dir = tempfile.mkdtemp()
        filepath = os.path.join(temp_dir, filename)
        with open(filepath, "w") as f:
            f.write(content)
        return filepath

    return _make_file


@pytest.fixture
def sql_injection_code() -> str:
    """Python code with an obvious string-concatenation SQL injection."""
    return '''import sqlite3

def get_user(username, password):
    conn = sqlite3.connect("users.db")
    query = "SELECT * FROM users WHERE username='" + username + "'"
    user = conn.execute(query).fetchone()
    if user and user[2] == password:
        return user
    return None
'''


@pytest.fixture
def clean_code() -> str:
    """Same intent as ``sql_injection_code`` but written safely."""
    return '''import sqlite3

def get_user(username, password):
    conn = sqlite3.connect("users.db")
    user = conn.execute(
        "SELECT * FROM users WHERE username=?",
        (username,)
    ).fetchone()
    if user and verify_password(password, user[2]):
        return user
    return None
'''


@pytest.fixture
def mock_pr_diff() -> str:
    """A short unified-diff-style PR patch with an injection bug."""
    return """+def cancel_ticket(ticket_id, user_id):
+    query = "SELECT * FROM tickets WHERE id=" + ticket_id
+    ticket = db.execute(query).fetchone()
+    if ticket == None:
+        return False
+    return True"""


@pytest.fixture
def sample_repo(tmp_path):
    """A minimal three-file fake repo on disk; returns its string path."""
    (tmp_path / "auth.py").write_text('''import sqlite3

def get_user(username, password):
    conn = sqlite3.connect("users.db")
    user = conn.execute(
        "SELECT * FROM users WHERE username=?",
        (username,)
    ).fetchone()
    return user
''')
    (tmp_path / "payment.py").write_text('''from datetime import datetime

def process_refund(ticket_id, amount):
    refund_pct = 0.9
    return {"refund": amount * refund_pct}
''')
    (tmp_path / "models.py").write_text('''TICKETS_TABLE = "tickets"
REFUNDS_TABLE = "refunds"
''')
    return str(tmp_path)


# ─── Mocking helpers ──────────────────────────────────────────────────


class _FakeChoice:
    def __init__(self, content: str):
        self.message = type("M", (), {"content": content})()


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class FakeGroqClient:
    """Drop-in Groq replacement.

    Either ``responses`` (a list of strings consumed in order) or ``response``
    (a single string returned for every call) can be supplied. Each
    ``chat.completions.create(...)`` returns one canned response. When the
    queue is exhausted, the last value repeats.
    """

    def __init__(self, response=None, responses=None):
        if responses is None:
            responses = [response if response is not None else "[]"]
        self._responses = list(responses)
        self.calls = []
        self.chat = self._Chat(self)

    class _Chat:
        def __init__(self, outer):
            self.completions = outer._Completions(outer)

    class _Completions:
        def __init__(self, outer):
            self._outer = outer

        def create(self, model, messages, **kwargs):
            self._outer.calls.append({"model": model, "messages": messages, **kwargs})
            if len(self._outer.calls) <= len(self._outer._responses):
                content = self._outer._responses[len(self._outer.calls) - 1]
            else:
                content = self._outer._responses[-1]
            return _FakeResponse(content)


@pytest.fixture
def fake_groq_client():
    """Factory returning a configurable ``FakeGroqClient``."""
    return FakeGroqClient


@pytest.fixture
def patch_groq(monkeypatch, fake_groq_client):
    """Patch ``llm_reviewer.client`` with a FakeGroqClient and return it.

    Usage::

        def test_x(patch_groq):
            client = patch_groq(response='[{"line":1,"severity":"warning","comment":"x"}]')
    """

    def _patch(response=None, responses=None):
        import llm_reviewer
        fake = fake_groq_client(response=response, responses=responses)
        monkeypatch.setattr(llm_reviewer, "client", fake)
        return fake

    return _patch
