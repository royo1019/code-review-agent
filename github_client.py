"""GitHub API helpers used by the agent.

Supports two authentication modes, selected at runtime:

- **Personal Access Token** — uses ``GITHUB_TOKEN``. Default; what the
  original implementation used.
- **GitHub App** — uses ``GITHUB_APP_ID`` + ``GITHUB_APP_PRIVATE_KEY`` to
  mint a per-installation token. Activated when an ``installation_id`` is
  passed (typically extracted from a webhook payload) and App credentials
  are present.

When App credentials are absent, an explicit ``installation_id`` is silently
ignored and PAT auth is used — this keeps the CLI entrypoint working without
needing to wire a GitHub App.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from github import Github
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_github_client(installation_id: Optional[int] = None) -> Github:
    """Return an authenticated ``Github`` client.

    Args:
        installation_id: When provided AND GitHub App env vars are set, use
            App-based auth via ``github_app.get_github_client_for_installation``.
            Otherwise fall back to PAT auth.

    Raises:
        ValueError: if neither auth mode is configured (no PAT, and either no
            App credentials or no ``installation_id``).
    """
    app_id = os.getenv("GITHUB_APP_ID")
    private_key = os.getenv("GITHUB_APP_PRIVATE_KEY")

    if app_id and private_key and installation_id:
        # Local import to avoid forcing PyJWT into the import graph when
        # callers only need PAT auth.
        from github_app import get_github_client_for_installation
        try:
            return get_github_client_for_installation(installation_id)
        except Exception as e:
            logger.error(
                f"GitHub App auth failed for installation {installation_id}: {e}. "
                "Falling back to PAT if available.",
                exc_info=True,
            )
            # fall through to PAT

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError(
            "No GitHub authentication configured. "
            "Set GITHUB_TOKEN, or set GITHUB_APP_ID + GITHUB_APP_PRIVATE_KEY "
            "and pass installation_id."
        )
    return Github(token)


def get_pr_details(repo_name, pr_number, installation_id: Optional[int] = None):
    """Fetch the GitHub repo and PR objects.

    Resolves the auth mode at call time so the same function works under both
    PAT and App auth.
    """
    g = get_github_client(installation_id=installation_id)
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)

    print(f"PR Title: {pr.title}")
    print(f"PR Author: {pr.user.login}")
    print(f"Files changed: {pr.changed_files}")

    return repo, pr


def get_pr_diff(pr):
    """Return a list of ``{filename, status, patch, additions, deletions}`` dicts."""
    files = pr.get_files()
    changed_files = []

    for f in files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "patch": f.patch,
            "additions": f.additions,
            "deletions": f.deletions
        })
        print(f"  - {f.filename} (+{f.additions} / -{f.deletions})")

    return changed_files


def get_file_content(repo, filename, pr):
    """Fetch raw UTF-8 file content at the PR head SHA, or ``None`` on error."""
    try:
        content = repo.get_contents(filename, ref=pr.head.sha)
        return content.decoded_content.decode("utf-8")
    except Exception as e:
        print(f"Could not fetch {filename}: {e}")
        return None


def post_review_comment(pr, body):
    """Post a free-form issue comment on the PR."""
    pr.create_issue_comment(body)
    print("Comment posted successfully")


def post_inline_comments(pr, repo, comments, filename):
    """Post per-line review comments. Returns the number successfully posted."""
    commit = list(pr.get_commits())[-1]
    posted = 0

    for c in comments:
        try:
            pr.create_review_comment(
                body=c['comment'],
                commit=commit,
                path=filename,
                line=c['line']
            )
            posted += 1
            print(f"  Posted comment on line {c['line']}")
        except Exception as e:
            print(f"  Could not post inline on line {c['line']}: {e}")

    return posted


def post_summary_and_verdict(pr, comments):
    """Post the aggregate summary comment and return APPROVE / REQUEST_CHANGES."""
    critical = [c for c in comments if c['severity'] == 'critical']
    warnings = [c for c in comments if c['severity'] == 'warning']
    suggestions = [c for c in comments if c['severity'] == 'suggestion']

    summary = f"""## 🤖 Automated Code Review

| Severity | Count |
|----------|-------|
| 🔴 Critical | {len(critical)} |
| 🟡 Warning | {len(warnings)} |
| 🔵 Suggestion | {len(suggestions)} |
| **Total** | **{len(comments)}** |

"""
    if critical:
        summary += "### 🔴 Critical Issues (must fix before merge)\n"
        for c in critical:
            summary += f"- Line {c['line']}: {c['comment']}\n"
        summary += "\n"

    if warnings:
        summary += "### 🟡 Warnings\n"
        for c in warnings:
            summary += f"- Line {c['line']}: {c['comment']}\n"
        summary += "\n"

    summary += "_Reviewed by CodeReviewBot — powered by Groq LLaMA 3.3 + LangGraph + RAG_"

    pr.create_issue_comment(summary)
    print("\nSummary comment posted.")

    if critical:
        print("verdict: REQUEST CHANGES (critical issues found)")
        verdict = "REQUEST_CHANGES"
    else:
        print("Verdict: APPROVE (no critical issues)")
        verdict = "APPROVE"

    return verdict
