import os
from github import Github
from dotenv import load_dotenv

load_dotenv()

def get_github_client():
    token = os.getenv("GITHUB_TOKEN")
    return Github(token)

def get_pr_details(repo_name, pr_number):
    g = get_github_client()
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)

    print(f"PR Title: {pr.title}")
    print(f"PR Author: {pr.user.login}")
    print(f"Files changed: {pr.changed_files}")

    return repo, pr

def get_pr_diff(pr):
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
    try:
        content = repo.get_contents(filename, ref=pr.head.sha)
        return content.decoded_content.decode("utf-8")
    except Exception as e:
        print(f"Could not fetch {filename}: {e}")
        return None

def post_review_comment(pr, body):
    pr.create_issue_comment(body)
    print("Comment posted successfully")
    

def post_inline_comments(pr, repo, comments, filename):
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