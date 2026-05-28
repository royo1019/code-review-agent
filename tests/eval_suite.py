"""LLM-quality evaluation suite for the Code Review Agent.

Runs each test case against the real Groq endpoint ``runs`` times (default 3)
to smooth over LLM non-determinism, then computes:

  - overall pass rate
  - critical-issue detection rate
  - false-positive rate (clean code wrongly flagged)
  - per-case results

Usage::

    python tests/eval_suite.py
    python tests/eval_suite.py --runs 5

Filename intentionally starts with ``eval_`` not ``test_`` so pytest won't
auto-collect it — it costs real Groq tokens.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

# Make the project root importable when run directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


TEST_CASES: List[Dict[str, Any]] = [
    {
        "name": "SQL injection via string concat",
        "code": (
            "import sqlite3\n\n"
            "def get_user(uid):\n"
            "    return sqlite3.connect('x.db').execute("
            "\"SELECT * FROM users WHERE id='\" + uid + \"'\").fetchone()\n"
        ),
        "must_catch": ["sql", "injection", "parameteriz"],
        "expected_severity": "critical",
        "should_not_flag": [],
    },
    {
        "name": "SQL injection via f-string",
        "code": (
            "import sqlite3\n\n"
            "def get_user(uid):\n"
            "    return sqlite3.connect('x.db').execute(f\"SELECT * FROM users WHERE id={uid}\").fetchone()\n"
        ),
        "must_catch": ["sql", "injection"],
        "expected_severity": "critical",
        "should_not_flag": [],
    },
    {
        "name": "Hardcoded password",
        "code": 'PASSWORD = "admin123"\n',
        "must_catch": ["password", "hardcoded", "secret"],
        "expected_severity": "critical",
        "should_not_flag": [],
    },
    {
        "name": "Eval with user input",
        "code": "def go(user_input):\n    return eval(user_input)\n",
        "must_catch": ["eval", "dangerous", "arbitrary"],
        "expected_severity": "critical",
        "should_not_flag": [],
    },
    {
        "name": "None comparison with ==",
        "code": "def foo(x):\n    if x == None:\n        return 1\n    return 0\n",
        "must_catch": ["none", "is none"],
        "expected_severity": "warning",
        "should_not_flag": [],
    },
    {
        "name": "Bare except clause",
        "code": "def foo():\n    try:\n        x = 1\n    except:\n        pass\n",
        "must_catch": ["except", "specific", "bare"],
        "expected_severity": "warning",
        "should_not_flag": [],
    },
    {
        "name": "Division by zero risk",
        "code": "def discount(price, pct):\n    return price * pct / pct\n",
        "must_catch": ["division", "zero", "zerodivision"],
        "expected_severity": "critical",
        "should_not_flag": [],
    },
    {
        "name": "Unused variable",
        "code": "def foo():\n    x = 5\n    return 10\n",
        "must_catch": ["unused", "x"],
        "expected_severity": "warning",
        "should_not_flag": [],
    },
    {
        "name": "Clean code should not be flagged critical",
        "code": (
            "import sqlite3\n\n"
            "def get_user(username):\n"
            "    return sqlite3.connect('x.db').execute(\n"
            "        \"SELECT * FROM users WHERE username=?\",\n"
            "        (username,)\n"
            "    ).fetchone()\n"
        ),
        "must_catch": [],
        "expected_severity": None,
        "should_not_flag": ["sql injection"],
    },
    {
        "name": "Missing input validation",
        "code": (
            "def transfer_money(amount, to_account):\n"
            "    db.execute(\"UPDATE accounts SET balance = balance - ? WHERE id=1\", (amount,))\n"
            "    db.execute(\"UPDATE accounts SET balance = balance + ? WHERE id=?\", (amount, to_account))\n"
        ),
        "must_catch": ["validation", "negative", "amount"],
        "expected_severity": "warning",
        "should_not_flag": [],
    },
]


def _evaluate_case(case: Dict[str, Any], comments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score one case against one run's worth of comments."""
    blob = " ".join((c.get("comment", "") or "").lower() for c in comments)
    must = [t.lower() for t in case["must_catch"]]
    should_not = [t.lower() for t in case["should_not_flag"]]

    caught = (not must) or any(t in blob for t in must)
    false_positive = any(t in blob for t in should_not)

    sev_hit = True
    if case["expected_severity"] is not None and comments:
        sev_hit = any(c.get("severity") == case["expected_severity"] for c in comments)

    passed = caught and not false_positive and sev_hit
    return {
        "caught": caught,
        "severity_match": sev_hit,
        "false_positive": false_positive,
        "passed": passed,
        "n_comments": len(comments),
    }


def run_eval(runs: int = 3) -> Dict[str, Any]:
    """Run the evaluation suite.

    Returns a summary dict with overall score, critical-detection rate,
    false-positive rate, and per-case details.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY must be set to run eval_suite")

    from llm_reviewer import review_pr

    per_case = []
    total = 0
    passes = 0
    critical_total = 0
    critical_passes = 0
    fp_total = 0
    fp_count = 0

    for case in TEST_CASES:
        print(f"\n── {case['name']} ──")
        run_results = []
        for i in range(runs):
            try:
                comments = review_pr(case["code"], [], [])
            except Exception as e:
                print(f"  run {i + 1}: EXCEPTION {e}")
                comments = []
            score = _evaluate_case(case, comments)
            run_results.append(score)
            print(
                f"  run {i + 1}: caught={score['caught']} sev_match={score['severity_match']} "
                f"fp={score['false_positive']} → {'PASS' if score['passed'] else 'FAIL'} "
                f"({score['n_comments']} comments)"
            )

            total += 1
            if score["passed"]:
                passes += 1
            if case["expected_severity"] == "critical":
                critical_total += 1
                if score["caught"] and score["severity_match"]:
                    critical_passes += 1
            if case["should_not_flag"]:
                fp_total += 1
                if score["false_positive"]:
                    fp_count += 1

        per_case.append({"name": case["name"], "runs": run_results})

    summary = {
        "overall_score": passes / total if total else 0.0,
        "critical_detection_rate": (critical_passes / critical_total) if critical_total else 0.0,
        "false_positive_rate": (fp_count / fp_total) if fp_total else 0.0,
        "per_case_results": per_case,
        "runs": runs,
        "total_runs": total,
        "passes": passes,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    summary = run_eval(runs=args.runs)

    print("\n" + "=" * 50)
    print(f"Overall score:           {summary['overall_score']:.1%}")
    print(f"Critical detection rate: {summary['critical_detection_rate']:.1%}  (target ≥ 90%)")
    print(f"False positive rate:     {summary['false_positive_rate']:.1%}  (target ≤ 10%)")
    print(f"Total: {summary['passes']}/{summary['total_runs']} passed across {summary['runs']} run(s)")
    print("=" * 50)


if __name__ == "__main__":
    main()
