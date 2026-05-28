"""Unit tests for linter.py — every spec case."""

from __future__ import annotations

import pytest

from linter import run_linters


# ── Bandit (security) ────────────────────────────────────────────────


def test_bandit_catches_sql_injection_string_concat():
    code = '''import sqlite3

def get_user(user_id):
    conn = sqlite3.connect("x.db")
    query = "SELECT * FROM users WHERE id='" + user_id + "'"
    return conn.execute(query).fetchone()
'''
    findings = run_linters("x.py", code)
    bandit = [f for f in findings if f["tool"] == "bandit"]
    assert len(bandit) >= 1
    assert any("SQL" in f["message"].lower() or "sql" in f["message"].lower() for f in bandit)


def test_bandit_catches_sql_injection_fstring():
    code = '''import sqlite3

def get_user(user_id):
    conn = sqlite3.connect("x.db")
    return conn.execute(f"SELECT * FROM users WHERE id={user_id}").fetchone()
'''
    findings = run_linters("x.py", code)
    bandit = [f for f in findings if f["tool"] == "bandit"]
    assert len(bandit) >= 1


def test_bandit_catches_hardcoded_password():
    code = 'PASSWORD = "admin123"\n'
    findings = run_linters("x.py", code)
    bandit = [f for f in findings if f["tool"] == "bandit"]
    assert any(
        "password" in f["message"].lower() or "hardcoded" in f["message"].lower() or "secret" in f["message"].lower()
        for f in bandit
    )


def test_bandit_catches_eval():
    code = "x = eval(input())\n"
    findings = run_linters("x.py", code)
    bandit = [f for f in findings if f["tool"] == "bandit"]
    assert len(bandit) >= 1


# ── Flake8 (style) ───────────────────────────────────────────────────


def test_flake8_catches_unused_import():
    code = "import os\nimport sys\n\ndef foo():\n    pass\n"
    findings = run_linters("x.py", code)
    flake = [f for f in findings if f["tool"] == "flake8"]
    assert any("imported but unused" in f["message"].lower() for f in flake)


def test_flake8_catches_none_comparison():
    code = "def foo(x):\n    if x == None:\n        return 1\n"
    findings = run_linters("x.py", code)
    flake = [f for f in findings if f["tool"] == "flake8"]
    assert any("E711" in f["message"] for f in flake)


def test_flake8_catches_bare_except():
    code = "def foo():\n    try:\n        x = 1\n    except:\n        pass\n"
    findings = run_linters("x.py", code)
    flake = [f for f in findings if f["tool"] == "flake8"]
    assert any("bare" in f["message"].lower() or "E722" in f["message"] for f in flake)


def test_flake8_line_numbers_are_correct():
    code = "x = 1\nx = 2\nimport os\n"  # unused 'os' on line 3
    findings = run_linters("x.py", code)
    flake = [f for f in findings if f["tool"] == "flake8" and "unused" in f["message"].lower()]
    assert flake, "expected an unused-import finding"
    assert flake[0]["line"] == 3


# ── Negative + edge cases ────────────────────────────────────────────


def test_clean_code_has_no_bandit_findings(clean_code):
    findings = run_linters("x.py", clean_code)
    bandit = [f for f in findings if f["tool"] == "bandit"]
    # The clean reference uses parameterized queries; bandit should be quiet.
    assert bandit == []


def test_unsupported_extension_returns_empty():
    findings = run_linters("test.java", "public class X {}")
    assert findings == []


def test_empty_file_returns_empty():
    findings = run_linters("x.py", "")
    # Linters may have no findings or only trivial ones; key requirement is no crash.
    assert isinstance(findings, list)


def test_linter_handles_syntax_error_gracefully():
    # Invalid Python — must not raise
    findings = run_linters("x.py", "def foo(:\n    pass\n")
    assert isinstance(findings, list)


def test_linter_handles_very_large_file():
    big = "\n".join(f"x{i} = {i}" for i in range(5000))
    findings = run_linters("x.py", big)
    assert isinstance(findings, list)
