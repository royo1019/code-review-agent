# Autonomous Code Review Agent — Complete Context + Spec for Claude Code

---

## 1. Project Overview

This is a production-grade agentic AI system that autonomously reviews GitHub Pull Requests. It replicates core CodeRabbit functionality at zero cost using open-source tools and free API tiers.

When a PR is opened, the agent:
1. Fetches the PR diff from GitHub
2. Runs static analysis (Flake8 + Bandit for Python, ESLint for JS/TS)
3. Retrieves semantically similar code from the codebase using RAG
4. Calls Groq LLaMA 3.3 70B to generate inline review comments
5. Posts comments directly on the PR on GitHub
6. Posts a summary with severity counts and a verdict (APPROVE or REQUEST_CHANGES)

---

## 2. Complete File Structure

```
code-review-agent/
├── agent.py           ← LangGraph 6-node state machine (orchestrator)
├── github_client.py   ← GitHub API (fetch PR, post comments, post verdict)
├── linter.py          ← Flake8 + Bandit (Python), ESLint (JS/TS), routing by file extension
├── rag.py             ← CodeBERT embeddings + ChromaDB vector search
├── llm_reviewer.py    ← Groq LLaMA 3.3 70B, prompt engineering, retry logic
├── server.py          ← FastAPI webhook server (receives GitHub webhook pings)
├── repo_cache.py      ← Clones repo locally, indexes once, caches per repo
├── main.py            ← CLI entrypoint (python3 main.py --repo owner/repo --pr 1 --path /path)
├── Dockerfile         ← Docker container (python:3.11-slim + git + nodejs + eslint)
├── railway.toml       ← Railway deployment config
├── requirements.txt   ← All Python dependencies
├── .env               ← GROQ_API_KEY, GITHUB_TOKEN, WEBHOOK_SECRET
└── .gitignore
```

---

## 3. Complete File Contents

### agent.py
```python
import os
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from github_client import (
    get_pr_details,
    get_pr_diff,
    get_file_content,
    post_inline_comments,
    post_summary_and_verdict
)
from linter import run_linters
from rag import index_codebase, retrieve_context
from llm_reviewer import review_pr

load_dotenv()

class AgentState(TypedDict):
    repo_name: str
    pr_number: int
    repo: Any
    pr: Any
    files: List[Dict]
    current_file: Dict
    lint_findings: List[Dict]
    rag_chunks: List[Dict]
    comments: List[Dict]
    all_comments: List[Dict]
    retry_count: int
    verdict: str

def fetch_pr_node(state: AgentState) -> AgentState:
    print("\n[Node 1] Fetching PR...")
    repo, pr = get_pr_details(state["repo_name"], state["pr_number"])
    files = get_pr_diff(pr)
    return {**state, "repo": repo, "pr": pr, "files": files, "all_comments": []}

def run_linters_node(state: AgentState) -> AgentState:
    print("\n[Node 2] Running linters...")
    f = state["current_file"]
    content = get_file_content(state["repo"], f["filename"], state["pr"])
    if not content:
        return {**state, "lint_findings": []}
    findings = run_linters(f["filename"], content)
    return {**state, "lint_findings": findings}

def fetch_rag_node(state: AgentState) -> AgentState:
    print("\n[Node 3] Fetching RAG context...")
    f = state["current_file"]
    chunks = retrieve_context(f["patch"])
    return {**state, "rag_chunks": chunks}

def call_llm_node(state: AgentState) -> AgentState:
    print("\n[Node 4] Calling LLM...")
    f = state["current_file"]
    comments = review_pr(
        f["patch"],
        state["lint_findings"],
        state["rag_chunks"]
    )
    return {**state, "comments": comments}

def retry_node(state: AgentState) -> AgentState:
    count = state.get("retry_count", 0) + 1
    print(f"\n[Retry] Attempt {count}...")
    return {**state, "retry_count": count}

def post_comments_node(state: AgentState) -> AgentState:
    print("\n[Node 5] Posting comments to GitHub...")
    f = state["current_file"]
    post_inline_comments(
        state["pr"],
        state["repo"],
        state["comments"],
        f["filename"]
    )
    all_comments = state.get("all_comments", []) + state["comments"]
    return {**state, "all_comments": all_comments}

def verdict_node(state: AgentState) -> AgentState:
    print("\n[Node 6] Posting verdict...")
    verdict = post_summary_and_verdict(state["pr"], state["all_comments"])
    return {**state, "verdict": verdict}

def should_retry(state: AgentState) -> str:
    if not state["comments"] and state.get("retry_count", 0) < 3:
        return "retry"
    return "post"

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("fetch_pr", fetch_pr_node)
    graph.add_node("run_linters", run_linters_node)
    graph.add_node("fetch_rag", fetch_rag_node)
    graph.add_node("call_llm", call_llm_node)
    graph.add_node("retry", retry_node)
    graph.add_node("post_comments", post_comments_node)
    graph.add_node("verdict", verdict_node)
    graph.set_entry_point("fetch_pr")
    graph.add_edge("fetch_pr", "run_linters")
    graph.add_edge("run_linters", "fetch_rag")
    graph.add_edge("fetch_rag", "call_llm")
    graph.add_conditional_edges("call_llm", should_retry, {
        "retry": "retry",
        "post": "post_comments"
    })
    graph.add_edge("retry", "call_llm")
    graph.add_edge("post_comments", "verdict")
    graph.add_edge("verdict", END)
    return graph.compile()

def run_agent(repo_name, pr_number, repo_path):
    index_codebase(repo_path)
    app = build_graph()
    repo, pr = get_pr_details(repo_name, pr_number)
    files = get_pr_diff(pr)
    for f in files:
        print(f"\n{'='*50}")
        print(f"Reviewing: {f['filename']}")
        print(f"{'='*50}")
        initial_state: AgentState = {
            "repo_name": repo_name,
            "pr_number": pr_number,
            "repo": repo,
            "pr": pr,
            "files": files,
            "current_file": f,
            "lint_findings": [],
            "rag_chunks": [],
            "comments": [],
            "all_comments": [],
            "retry_count": 0,
            "verdict": ""
        }
        app.invoke(initial_state)
```

### github_client.py
```python
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
```

### linter.py
```python
import os
import json
import subprocess
import tempfile

def save_file_to_temp(filename, content):
    temp_dir = tempfile.mkdtemp()
    filepath = os.path.join(temp_dir, os.path.basename(filename))
    with open(filepath, "w") as f:
        f.write(content)
    return filepath

def run_flake8(filepath):
    result = subprocess.run(["flake8", filepath], capture_output=True, text=True)
    findings = []
    for line in result.stdout.strip().split("\n"):
        if line:
            parts = line.split(":")
            if len(parts) >= 4:
                findings.append({
                    "tool": "flake8",
                    "line": int(parts[1]),
                    "severity": "warning",
                    "message": parts[3].strip()
                })
    return findings

def run_bandit(filepath):
    result = subprocess.run(
        ["bandit", "-r", filepath, "-f", "json", "-q"],
        capture_output=True, text=True
    )
    findings = []
    try:
        data = json.loads(result.stdout)
        for issue in data.get("results", []):
            findings.append({
                "tool": "bandit",
                "line": issue["line_number"],
                "severity": issue["issue_severity"].lower(),
                "message": issue["issue_text"]
            })
    except json.JSONDecodeError:
        pass
    return findings

def run_eslint(filepath):
    result = subprocess.run(
        ["eslint", filepath, "--format", "json", "--no-eslintrc",
         "--rule", "no-unused-vars: warn",
         "--rule", "no-undef: warn",
         "--rule", "eqeqeq: warn",
         "--rule", "no-eval: error",
         "--env", "browser,node,es6",
         "--parser-options", "ecmaVersion:2020"],
        capture_output=True, text=True
    )
    findings = []
    try:
        data = json.loads(result.stdout)
        for file_result in data:
            for msg in file_result.get("messages", []):
                findings.append({
                    "tool": "eslint",
                    "line": msg.get("line", 1),
                    "severity": "warning" if msg["severity"] == 1 else "critical",
                    "message": msg["message"]
                })
    except (json.JSONDecodeError, KeyError):
        pass
    return findings

def is_tool_installed(tool):
    result = subprocess.run(["which", tool], capture_output=True, text=True)
    return result.returncode == 0

def run_linters(filename, content):
    filepath = save_file_to_temp(filename, content)
    findings = []
    print(f"\nRunning linters on {filename}...")
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".py":
        flake8 = run_flake8(filepath)
        bandit = run_bandit(filepath)
        findings = flake8 + bandit
        print(f"  Flake8: {len(flake8)} issues")
        print(f"  Bandit: {len(bandit)} issues")
    elif ext in [".js", ".ts", ".jsx", ".tsx"]:
        if is_tool_installed("eslint"):
            eslint = run_eslint(filepath)
            findings = eslint
            print(f"  ESLint: {len(eslint)} issues")
        else:
            print("  ESLint not installed — skipping JS/TS linting")
    elif ext in [".java"]:
        print("  Java detected — LLM-only review")
    elif ext in [".go"]:
        print("  Go detected — LLM-only review")
    else:
        print(f"  {ext} file — LLM-only review")
    print(f"  Total: {len(findings)} issues")
    return findings
```

### rag.py
```python
import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "microsoft/codebert-base")
MODEL = SentenceTransformer(MODEL_NAME)
CLIENT = chromadb.Client()
COLLECTION = CLIENT.get_or_create_collection("codebase")

def chunk_file(filepath, chunk_size=50, overlap=10):
    with open(filepath, "r", errors="ignore") as f:
        lines = f.readlines()
    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + chunk_size, len(lines))
        chunk_text = "".join(lines[start:end])
        chunks.append({
            "text": chunk_text,
            "filename": filepath,
            "start_line": start + 1,
            "end_line": end
        })
        start += chunk_size - overlap
    return chunks

def index_codebase(repo_path):
    if COLLECTION.count() > 0:
        print(f"\nCodebase already indexed ({COLLECTION.count()} chunks). Skipping.")
        return
    print(f"\nIndexing codebase at: {repo_path}")
    all_chunks = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in [".git", "venv", "__pycache__", "node_modules"]]
        for filename in files:
            if filename.endswith((".py", ".js", ".ts", ".java", ".md")):
                filepath = os.path.join(root, filename)
                chunks = chunk_file(filepath)
                all_chunks.extend(chunks)
                print(f"  Indexed: {filename} ({len(chunks)} chunks)")
    if not all_chunks:
        print("No files found to index.")
        return
    texts = [c["text"] for c in all_chunks]
    embeddings = MODEL.encode(texts).tolist()
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    metadatas = [{"filename": c["filename"], "start_line": c["start_line"], "end_line": c["end_line"]} for c in all_chunks]
    COLLECTION.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"\nTotal chunks indexed: {len(all_chunks)}")

def retrieve_context(diff_text, n_results=3):
    query_embedding = MODEL.encode([diff_text]).tolist()
    results = COLLECTION.query(query_embeddings=query_embedding, n_results=n_results)
    context_chunks = []
    for i in range(len(results["documents"][0])):
        context_chunks.append({
            "text": results["documents"][0][i],
            "filename": results["metadatas"][0][i]["filename"],
            "start_line": results["metadatas"][0][i]["start_line"]
        })
    return context_chunks
```

### llm_reviewer.py
```python
import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_prompt(diff, lint_findings, rag_chunks):
    lint_text = ""
    for f in lint_findings:
        lint_text += f"  Line {f['line']} [{f['severity']}] ({f['tool']}): {f['message']}\n"
    rag_text = ""
    for i, chunk in enumerate(rag_chunks):
        rag_text += f"--- Existing code from {chunk['filename']} (line {chunk['start_line']}) ---\n"
        rag_text += chunk['text'][:300] + "\n\n"
    prompt = f"""You are a senior software engineer doing a code review.

You are given:
1. A PR diff (the new code being added)
2. Static analysis findings from Flake8 and Bandit
3. Relevant existing code from the codebase for context

Your job is to generate clear, specific, actionable review comments.

PR DIFF:
{diff}

STATIC ANALYSIS FINDINGS:
{lint_text if lint_text else "No issues found."}

EXISTING CODEBASE CONTEXT (retrieved via RAG):
{rag_text if rag_text else "No context retrieved."}

Generate a JSON array of review comments. Each comment must have:
- "line": the line number in the diff (integer)
- "severity": one of "critical", "warning", "suggestion"
- "comment": clear explanation of the issue and how to fix it specifically for this codebase

Rules:
- critical: security vulnerabilities, bugs that will crash the code
- warning: bad practices, performance issues, code smells
- suggestion: style improvements, minor enhancements
- Be specific — reference the actual variable names and function names in the code
- If RAG context shows a better pattern already exists in the codebase, mention it explicitly
- Return ONLY the JSON array, no other text

Example format:
[
  {{
    "line": 3,
    "severity": "critical",
    "comment": "SQL injection vulnerability — use parameterized queries instead."
  }}
]"""
    return prompt

def call_llm(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            comments = json.loads(raw)
            return comments
        except json.JSONDecodeError:
            print(f"  Attempt {attempt+1}: LLM returned malformed JSON, retrying...")
        except Exception as e:
            print(f"  Attempt {attempt+1}: Error — {e}, retrying...")
    print("  All retries failed. Returning empty comments.")
    return []

def review_pr(diff, lint_findings, rag_chunks):
    print("\nCalling LLM for review...")
    prompt = build_prompt(diff, lint_findings, rag_chunks)
    comments = call_llm(prompt)
    print(f"  LLM generated {len(comments)} comments")
    return comments
```

### server.py
```python
import os
import hmac
import hashlib
import asyncio
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from dotenv import load_dotenv
from repo_cache import get_repo_path
from agent import run_agent

load_dotenv()

app = FastAPI(title="AI Code Review Agent")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

def validate_signature(payload: bytes, signature: str) -> bool:
    if not WEBHOOK_SECRET:
        return True
    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)

async def process_pr(repo_name: str, pr_number: int):
    try:
        print(f"\nProcessing PR #{pr_number} from {repo_name}")
        repo_path = get_repo_path(repo_name)
        run_agent(repo_name, pr_number, repo_path)
        print(f"Review complete for PR #{pr_number}")
    except Exception as e:
        print(f"Error processing PR #{pr_number}: {e}")

@app.get("/")
def health_check():
    return {"status": "running", "service": "AI Code Review Agent", "version": "1.0.0"}

@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not validate_signature(payload, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")
    event = request.headers.get("X-GitHub-Event", "")
    if event != "pull_request":
        return {"status": "ignored", "reason": f"event {event} not handled"}
    data = await request.json()
    action = data.get("action")
    if action not in ["opened", "synchronize"]:
        return {"status": "ignored", "reason": f"action {action} not handled"}
    repo_name = data["repository"]["full_name"]
    pr_number = data["pull_request"]["number"]
    print(f"Received PR event: {action} on {repo_name}#{pr_number}")
    background_tasks.add_task(process_pr, repo_name, pr_number)
    return {"status": "accepted", "repo": repo_name, "pr": pr_number}
```

### repo_cache.py
```python
import os
import shutil
import tempfile
from git import Repo
from rag import index_codebase, COLLECTION
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
_cache = {}

def clone_repo(repo_name: str) -> str:
    print(f"\nCloning {repo_name}...")
    temp_dir = tempfile.mkdtemp()
    clone_url = f"https://{GITHUB_TOKEN}@github.com/{repo_name}.git"
    Repo.clone_from(clone_url, temp_dir)
    print(f"Cloned to {temp_dir}")
    return temp_dir

def get_repo_path(repo_name: str) -> str:
    if repo_name in _cache:
        print(f"Using cached index for {repo_name}")
        return _cache[repo_name]
    repo_path = clone_repo(repo_name)
    index_codebase(repo_path)
    _cache[repo_name] = repo_path
    print(f"Cached index for {repo_name}")
    return repo_path

def invalidate_cache(repo_name: str):
    if repo_name in _cache:
        old_path = _cache.pop(repo_name)
        shutil.rmtree(old_path, ignore_errors=True)
        print(f"Cache invalidated for {repo_name}")
```

### main.py
```python
import argparse
from agent import run_agent

parser = argparse.ArgumentParser(description="AI Code Review Agent")
parser.add_argument("--repo", required=True, help="GitHub repo e.g. royo1019/test-review-agent")
parser.add_argument("--pr", required=True, type=int, help="PR number e.g. 1")
parser.add_argument("--path", required=True, help="Local codebase path e.g. /Users/royo/test-repo")

args = parser.parse_args()
run_agent(args.repo, args.pr, args.path)
```

---

## 4. Environment Variables

```
GROQ_API_KEY=gsk_...        # Groq API key (free tier)
GITHUB_TOKEN=ghp_...        # GitHub Personal Access Token (repo scope)
WEBHOOK_SECRET=...          # Any random string for webhook validation
EMBEDDING_MODEL=microsoft/codebert-base  # Optional, defaults to codebert
```

---

## 5. How to Run Locally

```bash
# CLI mode
python3 main.py --repo owner/repo --pr 1 --path /local/repo/path

# Server mode
uvicorn server:app --reload --port 8000
```

---

## 6. What Works Right Now

- ✅ Fetches PR diff from any GitHub repo
- ✅ Runs Flake8 + Bandit on Python files
- ✅ Runs ESLint on JS/TS files (if installed)
- ✅ Falls back to LLM-only review for unsupported languages
- ✅ Indexes codebase with CodeBERT embeddings into ChromaDB
- ✅ Retrieves top 3 semantically similar chunks for each PR
- ✅ Calls Groq LLaMA 3.3 70B with diff + lint + RAG context
- ✅ Posts inline comments on the PR diff
- ✅ Posts summary table with severity counts
- ✅ Submits APPROVE or REQUEST_CHANGES verdict
- ✅ LangGraph state machine with retry logic
- ✅ FastAPI webhook server ready (not deployed yet)
- ✅ Docker container built successfully
- ✅ Repo cloning + per-repo RAG cache

---

## 7. Known Limitations (to fix)

- ❌ RAG treats code as plain text — no AST parsing, no call graph, no import graph
- ❌ ChromaDB is in-memory — resets on every server restart
- ❌ No queue system — simultaneous PRs block each other
- ❌ Webhook server not deployed yet (Fly.io needs credit card)
- ❌ No GitHub App — users must manually add webhook URL
- ❌ RAG index not refreshed when new commits pushed to main
- ❌ No test suite
- ❌ Large PRs (50+ files) not handled — no chunking strategy
- ❌ Java, Go, Ruby linters not integrated

---

## 8. Improvement Spec for Claude Code

Implement the following improvements in priority order:

---

### Priority 1 — AST-based code understanding

**Problem:** RAG currently treats code as plain text. It finds semantic similarity but doesn't understand code structure — function definitions, call chains, import relationships.

**Solution:** Add AST parsing layer using Python's built-in `ast` module.

**File to create:** `ast_parser.py`

```
What it should do:
- Walk every .py file in the repo
- Extract: function names, arguments, what functions they call, what they import
- Build a symbol table: {function_name: {file, line, calls: [], imported_by: []}}
- Build a call graph: {function_name: [list of functions it calls]}
- Build an import graph: {file: [list of files it imports from]}
- Expose: get_function_context(function_name) → returns related functions + their code
- Expose: get_file_dependencies(filename) → returns all files that import this file
```

**Integration:** In `rag.py`, after chunking, also extract AST metadata. Pass both RAG chunks AND AST context to `llm_reviewer.py`. The LLM prompt should say "here are similar code chunks (RAG) AND here are the functions this code calls/is called by (AST)."

**Expected outcome:** Agent can say "your cancel_ticket() calls process_refund() from payment.py but skips the 24-hour full refund window check on line 8 of payment.py" — instead of just "your refund calculation looks hardcoded."

---

### Priority 2 — Persistent ChromaDB

**Problem:** ChromaDB uses in-memory storage. Every server restart wipes the index — forcing a re-clone and re-index of every repo.

**Solution:** Switch to `PersistentClient` with a configurable storage path.

**Changes needed in `rag.py`:**
```python
# Change this:
CLIENT = chromadb.Client()

# To this:
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
CLIENT = chromadb.PersistentClient(path=CHROMA_PATH)
```

Also add a per-repo collection instead of one global collection:
```python
# Instead of one "codebase" collection, use repo-specific collections
# collection name = repo_name with / replaced by _ 
# e.g. "royo1019/ticketing-app" → "royo1019_ticketing_app"
```

This way multiple repos can be indexed simultaneously without collision.

---

### Priority 3 — Queue system for concurrent PRs

**Problem:** If two PRs are opened simultaneously, the second one blocks until the first finishes (30-60 seconds). FastAPI's BackgroundTasks runs tasks sequentially.

**Solution:** Add a simple async queue using Python's `asyncio.Queue`.

**File to create:** `queue_manager.py`

```
What it should do:
- Maintain an asyncio queue of (repo_name, pr_number) tuples
- Worker coroutine processes one PR at a time per repo
- Multiple repos processed concurrently (repo A and repo B in parallel)
- Same repo processed sequentially (PR #1 then PR #2 for same repo)
- Expose: enqueue_pr(repo_name, pr_number)
- Expose: start_worker() — starts background processing loop
- Log queue depth so you can monitor backlog
```

**Integration in `server.py`:** Instead of `background_tasks.add_task(process_pr, ...)`, call `queue_manager.enqueue_pr(repo_name, pr_number)`.

---

### Priority 4 — RAG index refresh on push

**Problem:** When developers push new commits to main, the RAG index becomes stale. The agent keeps recommending patterns from old code.

**Solution:** Add a push event handler in `server.py`.

**Changes needed in `server.py`:**
```
- Handle "push" event type from GitHub webhook
- Check if push is to the default branch (main/master)
- Call repo_cache.invalidate_cache(repo_name)
- Re-clone and re-index in background
- Log "Index refreshed for {repo_name}"
```

---

### Priority 5 — Test suite

**File to create:** `tests/` directory with:

```
tests/
├── test_linter.py      ← unit tests for each linter
├── test_rag.py         ← unit tests for RAG retrieval quality  
├── test_llm.py         ← unit tests for LLM output structure
├── test_agent.py       ← integration test for full pipeline
└── eval_suite.py       ← quality evaluation on known buggy files
```

**eval_suite.py spec:**
```
- Define 10 test cases: each has buggy code + expected issue type
- Run the full pipeline on each
- Measure: what % of issues does the agent catch?
- Run 3 times, report average (accounts for LLM non-determinism)
- Target: 80%+ catch rate on critical issues
```

---

### Priority 6 — GitHub App (replaces manual webhook)

**Problem:** Users currently have to manually add a webhook URL to their repo settings. A GitHub App allows one-click installation.

**What to build:**
- Register a GitHub App at github.com/settings/apps
- Handle the `installation` webhook event — auto-index the repo when installed
- Use GitHub App JWT authentication instead of personal access token
- Users install via: `github.com/apps/your-app-name` → Install → Select repos → Done

**Files to create/modify:**
- `github_app.py` — JWT token generation, app authentication
- Update `server.py` — handle `installation` event
- Update `github_client.py` — support both PAT and App authentication

---

### Priority 7 — Deployment on Fly.io

**Problem:** Webhook server runs locally only. Needs a public URL for GitHub to send webhooks to.

**Steps:**
- Add credit card to Fly.io (no charges for free tier usage)
- Run `flyctl launch --region bom --name code-review-agent --no-deploy --yes`
- Set secrets: `flyctl secrets set GROQ_API_KEY=... GITHUB_TOKEN=... WEBHOOK_SECRET=...`
- Set `CHROMA_PATH=/data/chroma_db` for persistent storage on Fly volume
- Create Fly volume: `flyctl volumes create chroma_data --size 1`
- Run `flyctl deploy`
- Get public URL and add to GitHub webhook settings

---

### Priority 8 — README

**File to create:** `README.md`

```
Sections:
1. What it does (2-3 sentences + screenshot of a real review)
2. Architecture diagram (ASCII or mermaid)
3. Tech stack table
4. How it works (the 6-step pipeline)  
5. How to use it (webhook setup instructions)
6. How to run locally (CLI instructions)
7. How to deploy (Fly.io steps)
8. Limitations and roadmap
```

---

## 9. Tech Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.11 | Everything |
| Agent framework | LangGraph | 6-node state machine |
| LLM | Groq LLaMA 3.3 70B | Review generation |
| Embeddings | Microsoft CodeBERT | Code-specific vectors |
| Vector DB | ChromaDB | Semantic search |
| Static analysis (Python) | Flake8 + Bandit | Style + security |
| Static analysis (JS/TS) | ESLint | Style + errors |
| GitHub integration | PyGithub | API calls |
| Web server | FastAPI + uvicorn | Webhook receiver |
| Containerization | Docker | Deployment |
| Repo cloning | GitPython | Local clone for RAG |

---

## 10. Key Design Decisions (explain these in interviews)

1. **Why LangGraph over plain Python?** State machine gives retry logic, conditional branching, and shared state without messy nested if/else.

2. **Why CodeBERT over MiniLM?** CodeBERT was trained on GitHub code+NL pairs — understands code semantics better than general-purpose models.

3. **Why RAG?** LLM has no knowledge of the target codebase. RAG retrieves relevant existing code so comments reference actual patterns rather than generic advice.

4. **Why Groq?** Free tier, fastest inference available (~500 tokens/sec), LLaMA 3.3 70B quality is sufficient for code review.

5. **Why background tasks in FastAPI?** GitHub webhook times out after 10 seconds. Agent takes 30-60 seconds. Must return 200 immediately and process in background.

6. **Why per-repo cache?** Re-cloning and re-indexing on every PR would add 45+ seconds of overhead. Cache gives instant retrieval after first PR.
```
