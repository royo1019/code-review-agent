"""FastAPI webhook receiver for the Code Review Agent.

GitHub posts to ``/webhook``. We validate the signature, route by event type:

- ``pull_request`` → ``handle_pull_request`` → enqueue in the per-repo queue.
- ``push`` (to default branch, touching code files) → schedule
  ``refresh_repo_index`` so the RAG/AST indexes stay in sync with ``main``.
- ``ping`` → pong response so GitHub's "redeliver" UI reports green.

The queue guarantees serial processing within a repo and parallelism across
repos. Push refreshes run via FastAPI ``BackgroundTasks`` with a per-repo lock
so concurrent pushes to the same repo serialize cleanly.
"""

import hashlib
import hmac
import logging
import os
import threading
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from agent import run_agent
from github_app import (
    forget_repo_installation,
    register_repo_installation,
)
from queue_manager import queue_manager
from repo_cache import get_repo_path, invalidate_cache

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Code Review Agent")

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# File extensions that trigger an index refresh on push. README/JSON/YAML
# changes don't need to invalidate the vector index.
_CODE_EXTENSIONS = (".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rb")

# Per-repo refresh lock — prevents two concurrent pushes for the same repo
# from racing each other in invalidate_cache + re-clone + re-index.
_refresh_locks: Dict[str, threading.Lock] = {}
_refresh_locks_guard = threading.Lock()


def _get_refresh_lock(repo_name: str) -> threading.Lock:
    """Return (creating if needed) the threading.Lock for one repo's refresh."""
    with _refresh_locks_guard:
        lock = _refresh_locks.get(repo_name)
        if lock is None:
            lock = threading.Lock()
            _refresh_locks[repo_name] = lock
        return lock


# ─── Signature validation ─────────────────────────────────
def validate_signature(payload: bytes, signature: str) -> bool:
    """Validate GitHub's HMAC-SHA256 webhook signature.

    Returns ``True`` if no ``WEBHOOK_SECRET`` is configured (dev mode).
    """
    if not WEBHOOK_SECRET:
        return True
    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


# ─── Job processor (called by queue workers) ──────────────
def process_pr(repo_name: str, pr_number: int) -> None:
    """Run one PR review end-to-end.

    Synchronous so it can be dispatched via ``asyncio.to_thread`` from the
    queue worker. Raises on failure so the queue manager can decide whether
    to retry or mark the job as failed.
    """
    try:
        logger.info(f"Starting review of PR #{pr_number} for {repo_name}")
        repo_path = get_repo_path(repo_name)
        run_agent(repo_name, pr_number, repo_path)
        logger.info(f"Completed review of PR #{pr_number} for {repo_name}")
    except Exception as e:
        logger.error(
            f"Failed to review PR #{pr_number} for {repo_name}: {e}",
            exc_info=True,
        )
        raise


def refresh_repo_index(repo_name: str) -> None:
    """Invalidate and rebuild the RAG/AST indexes for ``repo_name``.

    Called as a background task when a push lands on the default branch.
    Holds a per-repo lock so concurrent pushes for the same repo run one at
    a time. Never raises — all failures are logged so a broken refresh
    doesn't take down the webhook handler.
    """
    lock = _get_refresh_lock(repo_name)
    if not lock.acquire(blocking=True, timeout=600):
        logger.error(
            f"refresh_repo_index: could not acquire refresh lock for "
            f"{repo_name} within 10 min — skipping"
        )
        return
    try:
        logger.info(f"Refreshing index for {repo_name}")
        try:
            invalidate_cache(repo_name)
        except Exception as e:
            logger.error(
                f"invalidate_cache failed for {repo_name}: {e}", exc_info=True
            )
            # continue — get_repo_path will still rebuild from scratch
        try:
            get_repo_path(repo_name)
        except Exception as e:
            logger.error(
                f"Failed to re-clone/re-index {repo_name}: {e}", exc_info=True
            )
            return
        logger.info(f"Index refresh complete for {repo_name}")
    finally:
        lock.release()


# ─── Lifecycle ────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Server startup hook."""
    logger.info("Server starting up")


@app.on_event("shutdown")
async def shutdown_event():
    """Drain the queue gracefully on shutdown."""
    logger.info("Server shutting down")
    await queue_manager.shutdown()


# ─── Health check ─────────────────────────────────────────
@app.get("/")
def health_check():
    """Liveness probe."""
    return {
        "status": "running",
        "service": "AI Code Review Agent",
        "version": "1.0.0",
    }


@app.get("/status")
async def get_queue_status():
    """Return current queue depths, in-flight jobs, and processing stats."""
    return await queue_manager.get_status()


# ─── Event handlers ───────────────────────────────────────
async def handle_pull_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle ``pull_request`` event payloads.

    Accepts only ``opened``, ``synchronize``, and ``reopened`` actions. Skips
    closed PRs and drafts. Enqueues via the queue manager.
    """
    action = data.get("action")
    if action not in ("opened", "synchronize", "reopened"):
        return {"status": "ignored", "reason": f"action {action} not handled"}

    try:
        repo_name = data["repository"]["full_name"]
        pull_request = data["pull_request"]
        pr_number_raw = pull_request["number"]
    except (KeyError, TypeError) as e:
        logger.warning(f"PR webhook missing required field: {e}")
        raise HTTPException(status_code=400, detail="Malformed webhook payload")

    try:
        pr_number = int(pr_number_raw)
    except (TypeError, ValueError):
        logger.warning(f"PR webhook: non-integer pr_number {pr_number_raw!r}")
        raise HTTPException(status_code=400, detail="Invalid pr_number")

    pr_state = pull_request.get("state")
    if pr_state == "closed":
        return {"status": "ignored", "reason": "PR is closed"}

    if pull_request.get("draft"):
        return {"status": "ignored", "reason": "draft PRs not reviewed"}

    # If GitHub sent us an installation id (i.e. delivery is from a GitHub
    # App), remember the repo → installation mapping so PR review code can
    # authenticate per-installation if/when it wants to.
    installation = data.get("installation") or {}
    installation_id = installation.get("id")
    if isinstance(installation_id, int) and installation_id > 0:
        register_repo_installation(repo_name, installation_id)

    logger.info(f"Received PR event: {action} on {repo_name}#{pr_number}")

    success = await queue_manager.enqueue_pr(repo_name, pr_number)
    if not success:
        return {"status": "rejected", "reason": "queue full for this repo"}

    return {
        "status": "accepted",
        "repo": repo_name,
        "pr": pr_number,
    }


async def handle_push(
    data: Dict[str, Any], background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Handle ``push`` event payloads.

    Triggers an index refresh only when:

    - The push targets the repo's default branch (``main``/``master``/etc).
    - The push has at least one commit.
    - At least one changed file is code (``.py``/``.js``/``.ts``/...).
    """
    try:
        repository = data["repository"]
        repo_name = repository["full_name"]
    except (KeyError, TypeError) as e:
        logger.warning(f"Push webhook missing required field: {e}")
        raise HTTPException(status_code=400, detail="Malformed push payload")

    ref = data.get("ref") or ""
    default_branch = repository.get("default_branch") or "main"

    if ref != f"refs/heads/{default_branch}":
        return {"status": "ignored", "reason": "not default branch"}

    commits = data.get("commits")
    if not isinstance(commits, list):
        if commits is not None:
            logger.warning(
                f"Push webhook for {repo_name}: commits is "
                f"{type(commits).__name__}, not list — ignoring"
            )
        return {"status": "ignored", "reason": "no commits"}

    if not commits:
        return {"status": "ignored", "reason": "no commits"}

    changed_files = []
    for commit in commits:
        if not isinstance(commit, dict):
            continue
        for key in ("added", "modified", "removed"):
            items = commit.get(key) or []
            if isinstance(items, list):
                changed_files.extend(items)

    code_files = [
        f for f in changed_files
        if isinstance(f, str) and f.lower().endswith(_CODE_EXTENSIONS)
    ]
    if not code_files:
        return {"status": "ignored", "reason": "no code files changed"}

    logger.info(
        f"Push to {repo_name} default branch with "
        f"{len(code_files)} code files changed — scheduling refresh"
    )
    background_tasks.add_task(refresh_repo_index, repo_name)

    return {
        "status": "accepted",
        "action": "index_refresh_scheduled",
        "repo": repo_name,
    }


async def handle_installation(
    data: Dict[str, Any], background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Handle ``installation`` event payloads.

    On ``created``: register each repo's ``repo → installation`` mapping and
    schedule a background index build per repo. On ``deleted``: drop those
    mappings and invalidate the cached index for each repo.

    Other ``installation`` actions (``suspend``, ``unsuspend``,
    ``new_permissions_accepted``) are ignored.
    """
    action = data.get("action")
    if action not in ("created", "deleted"):
        return {"status": "ignored", "reason": f"installation action {action} not handled"}

    try:
        installation = data["installation"]
    except (KeyError, TypeError) as e:
        logger.warning(f"Installation webhook missing 'installation' field: {e}")
        raise HTTPException(status_code=400, detail="Malformed installation payload")

    installation_id = installation.get("id")
    if not isinstance(installation_id, int) or installation_id <= 0:
        logger.warning(f"Installation webhook: invalid installation.id {installation_id!r}")
        raise HTTPException(status_code=400, detail="Invalid installation.id")

    repositories = data.get("repositories") or []
    if not isinstance(repositories, list):
        logger.warning(
            f"Installation webhook: repositories is "
            f"{type(repositories).__name__}, not list — treating as empty"
        )
        repositories = []

    repo_names = []
    for repo in repositories:
        if not isinstance(repo, dict):
            continue
        name = repo.get("full_name")
        if isinstance(name, str) and name:
            repo_names.append(name)

    if action == "created":
        if not repo_names:
            return {
                "status": "accepted",
                "action": "no_repos_to_index",
                "installation_id": installation_id,
            }
        for name in repo_names:
            register_repo_installation(name, installation_id)
            background_tasks.add_task(refresh_repo_index, name)
        logger.info(
            f"Installation {installation_id} created for {len(repo_names)} repo(s); "
            f"scheduled background index build"
        )
        return {
            "status": "accepted",
            "action": "installation_recorded",
            "installation_id": installation_id,
            "count": len(repo_names),
        }

    # action == "deleted"
    for name in repo_names:
        forget_repo_installation(name)
        background_tasks.add_task(invalidate_cache, name)
    logger.info(
        f"Installation {installation_id} deleted for {len(repo_names)} repo(s); "
        f"invalidated caches"
    )
    return {
        "status": "accepted",
        "action": "installation_removed",
        "installation_id": installation_id,
        "count": len(repo_names),
    }


# ─── Webhook endpoint ─────────────────────────────────────
@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    """Receive a GitHub webhook delivery and dispatch to the right handler."""
    payload = await request.body()

    signature = request.headers.get("X-Hub-Signature-256", "")
    if not validate_signature(payload, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    event = request.headers.get("X-GitHub-Event", "")

    try:
        data = await request.json()
    except Exception as e:
        logger.warning(f"Webhook: malformed JSON body: {e}")
        raise HTTPException(status_code=400, detail="Malformed JSON")

    if event == "pull_request":
        return await handle_pull_request(data)
    if event == "push":
        return await handle_push(data, background_tasks)
    if event == "installation":
        return await handle_installation(data, background_tasks)
    if event == "ping":
        return {"status": "pong", "message": "Webhook configured successfully"}

    logger.debug(f"Ignoring event: {event}")
    return {"status": "ignored", "reason": f"event {event} not handled"}
