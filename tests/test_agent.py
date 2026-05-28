"""Integration tests for the full agent pipeline.

These tests run real network calls (GitHub + Groq) and are deselected by
default via the ``integration`` marker in ``pytest.ini``. To run them::

    pytest -m integration --override-ini="addopts="

The tests skip themselves if either credential is missing or appears to be a
test dummy.
"""

from __future__ import annotations

import os

import pytest


def _real_creds_present() -> bool:
    groq = os.getenv("GROQ_API_KEY", "")
    gh = os.getenv("GITHUB_TOKEN", "")
    return bool(groq) and bool(gh) and not groq.startswith("test_") and not gh.startswith("test_")


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _real_creds_present(), reason="Real GROQ_API_KEY + GITHUB_TOKEN required"),
]


def test_agent_pipeline_smoke(tmp_path):
    """Build the LangGraph and verify the wiring runs without raising.

    Indexes a tiny on-disk repo and exercises ``run_agent`` against a stub
    PR. We don't assert on LLM output content — only that the pipeline
    completes when given valid inputs.
    """
    # Set up a minimal repo
    (tmp_path / "auth.py").write_text(
        "import sqlite3\n\ndef get_user(name):\n    return sqlite3.connect('x.db')\n"
    )

    # This will hit GitHub for PR details; we only run it if a real repo+PR
    # are configured via env vars to keep the test self-contained.
    repo_env = os.getenv("INTEGRATION_TEST_REPO")
    pr_env = os.getenv("INTEGRATION_TEST_PR")
    if not repo_env or not pr_env:
        pytest.skip("Set INTEGRATION_TEST_REPO and INTEGRATION_TEST_PR to run")

    from agent import run_agent
    run_agent(repo_env, int(pr_env), str(tmp_path))
