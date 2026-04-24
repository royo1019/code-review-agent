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
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# ─── Signature validation ─────────────────────────────────
def validate_signature(payload: bytes, signature: str) -> bool:
    if not WEBHOOK_SECRET:
        return True
    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


# ─── Background task ─────────────────────────────────────
async def process_pr(repo_name: str, pr_number: int):
    try:
        print(f"\nProcessing PR #{pr_number} from {repo_name}")
        repo_path = get_repo_path(repo_name)
        run_agent(repo_name, pr_number, repo_path)
        print(f"Review complete for PR #{pr_number}")
    except Exception as e:
        print(f"Error processing PR #{pr_number}: {e}")


# ─── Health check ─────────────────────────────────────────
@app.get("/")
def health_check():
    return {
        "status": "running",
        "service": "AI Code Review Agent",
        "version": "1.0.0"
    }


# ─── Webhook endpoint ─────────────────────────────────────
@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    # read raw body
    payload = await request.body()

    # validate GitHub signature
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not validate_signature(payload, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # only handle pull_request events
    event = request.headers.get("X-GitHub-Event", "")
    if event != "pull_request":
        return {"status": "ignored", "reason": f"event {event} not handled"}

    # parse payload
    data = await request.json()
    action = data.get("action")

    # only trigger on opened or synchronize (new commits pushed)
    if action not in ["opened", "synchronize"]:
        return {"status": "ignored", "reason": f"action {action} not handled"}

    repo_name = data["repository"]["full_name"]
    pr_number = data["pull_request"]["number"]

    print(f"Received PR event: {action} on {repo_name}#{pr_number}")

    # run agent in background so GitHub doesn't timeout
    background_tasks.add_task(process_pr, repo_name, pr_number)

    return {
        "status": "accepted",
        "repo": repo_name,
        "pr": pr_number
    }
