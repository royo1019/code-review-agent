"""Unit tests for llm_reviewer.py.

The Groq client is replaced with ``FakeGroqClient`` so no real LLM calls run.
There's an additional ``@pytest.mark.integration`` test that exercises the
real LLM — deselected by default and gated on ``GROQ_API_KEY``.
"""

from __future__ import annotations

import json
import os

import pytest

import llm_reviewer
from llm_reviewer import build_prompt, call_llm, review_pr


SAMPLE_GOOD_JSON = json.dumps([
    {"line": 3, "severity": "critical", "comment": "SQL injection — use parameterized queries"},
    {"line": 5, "severity": "warning", "comment": "Compare to None with `is None`"},
])


# ── review_pr structure ───────────────────────────────────────────────


def test_review_pr_returns_list(patch_groq):
    patch_groq(response=SAMPLE_GOOD_JSON)
    out = review_pr("+x = 1", [], [])
    assert isinstance(out, list)


def test_review_pr_returns_correct_structure(patch_groq):
    patch_groq(response=SAMPLE_GOOD_JSON)
    out = review_pr("+x = 1", [], [])
    assert out
    for c in out:
        assert set(c.keys()) >= {"line", "severity", "comment"}


def test_review_pr_severity_values_valid(patch_groq):
    patch_groq(response=SAMPLE_GOOD_JSON)
    out = review_pr("+x = 1", [], [])
    valid = {"critical", "warning", "suggestion"}
    assert all(c["severity"] in valid for c in out)


def test_review_pr_line_numbers_are_integers(patch_groq):
    patch_groq(response=SAMPLE_GOOD_JSON)
    out = review_pr("+x = 1", [], [])
    assert all(isinstance(c["line"], int) and c["line"] >= 1 for c in out)


# ── review_pr null/empty inputs ───────────────────────────────────────


def test_review_pr_handles_empty_diff(patch_groq):
    patch_groq(response="[]")
    out = review_pr("", [], [])
    assert isinstance(out, list)


def test_review_pr_handles_none_lint_findings(patch_groq):
    patch_groq(response="[]")
    out = review_pr("+x = 1", None, [])
    assert isinstance(out, list)


def test_review_pr_handles_none_rag_chunks(patch_groq):
    patch_groq(response="[]")
    out = review_pr("+x = 1", [], None)
    assert isinstance(out, list)


# ── call_llm retry behavior ───────────────────────────────────────────


def test_call_llm_retries_on_json_error(patch_groq):
    fake = patch_groq(responses=["not json", "still not json", SAMPLE_GOOD_JSON])
    out = call_llm("ignored prompt", retries=3)
    assert len(fake.calls) == 3
    assert isinstance(out, list) and len(out) == 2


def test_call_llm_returns_empty_after_max_retries(patch_groq):
    fake = patch_groq(responses=["bad", "still bad", "yet again"])
    out = call_llm("ignored prompt", retries=3)
    assert out == []
    assert len(fake.calls) == 3


def test_call_llm_strips_markdown_fences(patch_groq):
    wrapped = "```json\n" + SAMPLE_GOOD_JSON + "\n```"
    patch_groq(response=wrapped)
    out = call_llm("ignored prompt", retries=1)
    assert len(out) == 2


# ── build_prompt content ──────────────────────────────────────────────


def test_build_prompt_includes_diff():
    p = build_prompt("+def foo():\n+    return 1\n", [], [])
    assert "def foo()" in p


def test_build_prompt_includes_lint_findings():
    findings = [{"line": 4, "severity": "warning", "tool": "flake8", "message": "imported but unused"}]
    p = build_prompt("+x", findings, [])
    assert "imported but unused" in p


def test_build_prompt_includes_rag_context():
    chunks = [{"text": "def helper(): pass", "filename": "auth.py", "start_line": 10}]
    p = build_prompt("+x", [], chunks)
    assert "auth.py" in p


def test_build_prompt_includes_ast_context_when_supplied():
    p = build_prompt("+x", [], [], ast_context="=== AST ANALYSIS ===\nmy_func found at line 1")
    assert "CODE STRUCTURE ANALYSIS" in p
    assert "my_func" in p


def test_build_prompt_omits_ast_section_when_absent():
    p = build_prompt("+x", [], [])
    assert "AST analysis not available" in p


# ── Integration (real LLM) ─────────────────────────────────────────────


def _real_groq_key_present() -> bool:
    k = os.getenv("GROQ_API_KEY", "")
    return bool(k) and not k.startswith("test_")


@pytest.mark.integration
@pytest.mark.skipif(not _real_groq_key_present(), reason="No real GROQ_API_KEY")
def test_review_pr_catches_sql_injection_real_llm(mock_pr_diff):
    """Exercises the real Groq endpoint — costs tokens. Deselected by default."""
    out = review_pr(mock_pr_diff, [], [])
    assert isinstance(out, list)
    blob = " ".join((c.get("comment", "") or "").lower() for c in out)
    assert "sql" in blob or "injection" in blob or "parameteri" in blob
