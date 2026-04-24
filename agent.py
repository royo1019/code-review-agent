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

# ─── State ───────────────────────────────────────────────
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

# ─── Nodes ───────────────────────────────────────────────
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


# ─── Conditional edges ───────────────────────────────────
def should_retry(state: AgentState) -> str:
    if not state["comments"] and state.get("retry_count", 0) < 3:
        return "retry"
    return "post"


# ─── Build graph ─────────────────────────────────────────
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


# ─── Run ─────────────────────────────────────────────────
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