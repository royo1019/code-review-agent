# Autonomous Code Review Agent — Complete Claude Code Spec

---

## CRITICAL INSTRUCTIONS FOR CLAUDE CODE

1. Read every existing file in this project before writing any code
2. Never break existing functionality — all current tests must still pass
3. Implement each Priority fully and test it before moving to the next
4. Handle every edge case listed — do not skip any
5. Every function must have a docstring
6. Every external call (GitHub API, Groq, ChromaDB) must be wrapped in try/except
7. Use Python logging module throughout — never use print() in new code
8. All new files go in the project root unless specified otherwise
9. After implementing each Priority, run the existing test files to confirm nothing broke

---

## EXISTING PROJECT SUMMARY

### What is built
A multi-step agentic AI system that autonomously reviews GitHub Pull Requests.

### Pipeline (in order)
1. Fetch PR diff from GitHub using PyGithub
2. Run Flake8 + Bandit (Python) or ESLint (JS/TS) on changed files
3. Retrieve semantically similar code from the repo using CodeBERT + ChromaDB RAG
4. Call Groq LLaMA 3.3 70B with diff + lint findings + RAG context
5. Post inline comments on the PR diff
6. Post summary table + APPROVE or REQUEST_CHANGES verdict

### Orchestration
LangGraph 6-node state machine in agent.py:
fetch_pr → run_linters → fetch_rag → call_llm → post_comments → verdict
With conditional retry edge: call_llm → retry (if empty) → call_llm (max 3x)

### Entry points
- CLI: `python3 main.py --repo owner/repo --pr 1 --path /local/repo/path`
- Server: `uvicorn server:app --port 8000` (FastAPI webhook receiver)

### Environment variables required
```
GROQ_API_KEY=gsk_...
GITHUB_TOKEN=ghp_...
WEBHOOK_SECRET=any_random_string
EMBEDDING_MODEL=microsoft/codebert-base  (optional, defaults to codebert)
CHROMA_PATH=./chroma_db  (optional, defaults to ./chroma_db)
LOG_LEVEL=INFO  (optional, defaults to INFO)
```

---

## PRIORITY 1 — AST-Based Code Understanding

### Problem
RAG treats code as plain text. It finds semantic similarity but does not understand:
- Which functions exist in the codebase
- Which functions call which other functions
- Which files import which other files
- What arguments each function takes
- Where a function is defined vs where it is called

This means the agent says "use parameterized queries" generically instead of
"your cancel_ticket() on line 14 should call process_refund() from payment.py
which already handles the 24-hour full refund window on line 8."

### Solution
Create ast_parser.py that builds three data structures:
1. Symbol table — every function/class in the codebase
2. Call graph — which functions call which
3. Import graph — which files import which

Then integrate this context into the LLM prompt alongside RAG chunks.

---

### File to create: `ast_parser.py`

#### Complete function signatures and behavior:

```python
import ast
import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
```

#### Dataclasses to define:

```python
@dataclass
class FunctionInfo:
    name: str                    # function name e.g. "cancel_ticket"
    filepath: str                # absolute path to file
    lineno: int                  # line number where defined
    args: List[str]              # argument names e.g. ["ticket_id", "user_id"]
    returns: Optional[str]       # return type annotation if present e.g. "dict"
    calls: List[str]             # names of functions this function calls
    decorators: List[str]        # decorator names e.g. ["staticmethod", "property"]
    docstring: Optional[str]     # first line of docstring if present
    is_method: bool              # True if inside a class
    class_name: Optional[str]    # class name if is_method is True

@dataclass
class ClassInfo:
    name: str                    # class name
    filepath: str                # absolute path to file
    lineno: int                  # line number where defined
    bases: List[str]             # parent class names
    methods: List[str]           # method names in this class
    docstring: Optional[str]     # first line of docstring if present

@dataclass
class ImportInfo:
    filepath: str                # file doing the importing
    module: str                  # module being imported e.g. "payment"
    names: List[str]             # specific names imported e.g. ["process_refund"]
    is_from_import: bool         # True if "from X import Y", False if "import X"
    lineno: int                  # line number of import statement

@dataclass
class ASTIndex:
    functions: Dict[str, List[FunctionInfo]] = field(default_factory=dict)
    # key = function name, value = list (same name can exist in multiple files)
    
    classes: Dict[str, ClassInfo] = field(default_factory=dict)
    # key = class name
    
    imports: Dict[str, List[ImportInfo]] = field(default_factory=dict)
    # key = filepath, value = list of imports in that file
    
    call_graph: Dict[str, List[str]] = field(default_factory=dict)
    # key = "filepath::function_name", value = list of called function names
    
    reverse_call_graph: Dict[str, List[str]] = field(default_factory=dict)
    # key = function_name, value = list of "filepath::caller_function" that call it
    
    file_to_functions: Dict[str, List[str]] = field(default_factory=dict)
    # key = filepath, value = list of function names defined in that file
```

#### Functions to implement:

**`parse_file(filepath: str) -> Optional[ast.Module]`**
- Read file content with `open(filepath, 'r', errors='ignore')`
- Parse with `ast.parse(content)`
- Return None if SyntaxError or any other exception — never crash on unparseable files
- Log warning if file cannot be parsed: `logger.warning(f"Cannot parse {filepath}: {e}")`
- Return the AST module node on success

**`extract_functions(tree: ast.Module, filepath: str) -> List[FunctionInfo]`**
- Walk the AST tree using `ast.walk(tree)`
- Find all `ast.FunctionDef` and `ast.AsyncFunctionDef` nodes
- For each function:
  - Extract `node.name`
  - Extract `node.lineno`
  - Extract args: `[arg.arg for arg in node.args.args]` — exclude 'self' and 'cls'
  - Extract return annotation: check `node.returns` — if present, use `ast.unparse(node.returns)`, else None
  - Extract decorators: `[ast.unparse(d) for d in node.decorator_list]` — if empty list use []
  - Extract docstring: `ast.get_docstring(node)` — take first line only if present
  - Extract calls: walk the function body, find all `ast.Call` nodes
    - If `call.func` is `ast.Name`: add `call.func.id` to calls list
    - If `call.func` is `ast.Attribute`: add `call.func.attr` to calls list
    - Deduplicate calls list
  - Detect if method: check if parent node is `ast.ClassDef`
  - Extract class name if method
- Edge cases:
  - Lambda functions: skip (they have no name)
  - Nested functions: include them, mark filepath correctly
  - Functions with *args/**kwargs: include 'args' and 'kwargs' in args list
  - Functions with no body (just `pass`): include, calls will be empty
  - `ast.unparse` not available in Python < 3.9: use `ast.dump` as fallback

**`extract_classes(tree: ast.Module, filepath: str) -> List[ClassInfo]`**
- Walk the AST tree
- Find all `ast.ClassDef` nodes
- For each class:
  - Extract name, lineno
  - Extract bases: `[ast.unparse(b) for b in node.bases]` — handle empty list
  - Extract methods: names of FunctionDef nodes directly inside this ClassDef
  - Extract docstring: first line only
- Edge cases:
  - Classes with no bases: bases = []
  - Classes with no methods: methods = []
  - Metaclasses: include in bases list
  - Dataclasses: treat like normal classes

**`extract_imports(tree: ast.Module, filepath: str) -> List[ImportInfo]`**
- Walk the AST tree
- Find all `ast.Import` nodes:
  - For each alias in node.names: create ImportInfo with is_from_import=False
  - module = alias.name, names = [alias.asname or alias.name]
- Find all `ast.ImportFrom` nodes:
  - module = node.module or '' (handle relative imports where module is None)
  - names = [alias.name for alias in node.names]
  - is_from_import = True
- Edge cases:
  - `from . import something` (relative import): module = '.' + (node.module or '')
  - `import os, sys` (multiple imports): create separate ImportInfo per alias
  - `from module import *`: names = ['*']
  - `__future__` imports: include them
  - Try/except imports (conditional): still include them

**`build_index(repo_path: str) -> ASTIndex`**
- Walk repo_path with os.walk
- Skip directories: `.git`, `venv`, `__pycache__`, `node_modules`, `.tox`, `dist`, `build`, `eggs`, `.eggs`
- Skip files: any file not ending in `.py`
- Skip files larger than 1MB (likely generated files): check `os.path.getsize(filepath) > 1_000_000`
- For each .py file:
  - Call `parse_file(filepath)` — skip if returns None
  - Call `extract_functions(tree, filepath)` — add to index
  - Call `extract_classes(tree, filepath)` — add to index
  - Call `extract_imports(tree, filepath)` — add to index
- Build call_graph: for each function, key = "filepath::function_name"
- Build reverse_call_graph: for each called function name, record who calls it
- Build file_to_functions: {filepath: [function_name, ...]}
- Log summary: `logger.info(f"AST index built: {len(functions)} functions, {len(classes)} classes across {file_count} files")`
- Return ASTIndex
- Edge cases:
  - Empty repo: return empty ASTIndex
  - Repo with no .py files: return empty ASTIndex
  - File that exists in os.walk but is deleted by the time we open it: catch FileNotFoundError

**`get_function_context(index: ASTIndex, function_name: str) -> Dict[str, Any]`**
- Look up function_name in index.functions
- If not found: return {"found": False, "function_name": function_name}
- If found (may be in multiple files): return all occurrences
- For each occurrence:
  - What it calls: index.call_graph.get(f"{filepath}::{function_name}", [])
  - What calls it: index.reverse_call_graph.get(function_name, [])
  - Its class if method: class_name field
- Return structure:
```python
{
    "found": True,
    "function_name": function_name,
    "definitions": [
        {
            "filepath": str,
            "lineno": int,
            "args": List[str],
            "calls": List[str],
            "called_by": List[str],
            "is_method": bool,
            "class_name": Optional[str],
            "docstring": Optional[str]
        }
    ]
}
```

**`get_file_dependencies(index: ASTIndex, filepath: str) -> Dict[str, Any]`**
- Look up filepath in index.imports
- Find all files that import FROM this file:
  - Walk index.imports for all files
  - Check if any ImportInfo.module matches this filepath (by basename without .py)
- Return structure:
```python
{
    "filepath": filepath,
    "imports": [ImportInfo objects as dicts],
    "imported_by": [filepath strings that import this file],
    "functions_defined": index.file_to_functions.get(filepath, [])
}
```

**`format_ast_context_for_llm(index: ASTIndex, diff_text: str) -> str`**
- Parse the diff_text to extract function names mentioned (lines starting with + or -)
- Use simple regex: find all words that match `[a-zA-Z_][a-zA-Z0-9_]*\(` pattern (function calls)
- Deduplicate function names found
- For each function name found in diff:
  - Call `get_function_context(index, function_name)`
  - If found: format as readable text
- Also find file imports mentioned in diff
- Return a formatted string like:
```
=== AST ANALYSIS ===

Functions referenced in this PR:

cancel_ticket (defined in cancel.py, line 4)
  - Arguments: ticket_id, user_id
  - Calls: sqlite3.connect, cursor.execute, process_refund
  - Called by: (nothing calls this yet)

process_refund (defined in payment.py, line 3)  
  - Arguments: ticket_id, amount
  - Calls: datetime.now
  - Called by: cancel_ticket

=== IMPORT RELATIONSHIPS ===
cancel.py imports: sqlite3, datetime
payment.py imports: datetime
```
- Edge cases:
  - No functions found in diff: return "=== AST ANALYSIS ===\nNo function references detected in diff."
  - Function name found in diff but not in index: note it as "not found in codebase"
  - Very long output (>2000 chars): truncate with "... (truncated for brevity)"

---

### Changes to `rag.py`

Add at the top:
```python
from ast_parser import build_index, format_ast_context_for_llm, ASTIndex
```

Add a global AST index variable:
```python
AST_INDEX: Optional[ASTIndex] = None
```

In `index_codebase()`, after building ChromaDB index, also build AST index:
```python
global AST_INDEX
AST_INDEX = build_index(repo_path)
logger.info("AST index built successfully")
```

Add new function:
```python
def get_ast_context(diff_text: str) -> str:
    """Get AST-based code structure context for a PR diff."""
    if AST_INDEX is None:
        return ""
    return format_ast_context_for_llm(AST_INDEX, diff_text)
```

Export `get_ast_context` so other modules can import it.

---

### Changes to `llm_reviewer.py`

Import:
```python
from rag import get_ast_context
```

Update `build_prompt()` to accept `ast_context: str` parameter:

Add new section to the prompt between RAG context and instructions:
```
=== CODE STRUCTURE ANALYSIS (AST) ===
{ast_context if ast_context else "AST analysis not available."}
```

Update `review_pr()` to:
```python
def review_pr(diff, lint_findings, rag_chunks):
    ast_context = get_ast_context(diff)
    comments = call_llm(build_prompt(diff, lint_findings, rag_chunks, ast_context))
    return comments
```

Update the prompt instructions to add:
```
- If AST analysis shows a function already exists in the codebase that does what 
  the new code is trying to do, explicitly recommend using it
- If AST analysis shows the new code calls a function that doesn't exist in the 
  codebase, flag it as a bug
- If AST analysis shows the new code duplicates logic from another file, 
  recommend DRY refactoring
```

---

### Edge cases for Priority 1

- Repo has no Python files: AST_INDEX is empty, get_ast_context returns ""
- File has syntax errors: parse_file returns None, file is skipped silently
- Function name in diff matches a Python builtin (len, print, etc.): include it but note "builtin"
- Circular imports in codebase: build_index handles this gracefully (just records the imports, no recursion)
- Very large function (500+ lines): include it but truncate docstring to first line only
- Function with same name in multiple files: show all occurrences in context
- Async functions: treat same as regular functions
- Class methods vs standalone functions: distinguish with is_method flag

---

## PRIORITY 2 — Persistent ChromaDB with Per-Repo Collections

### Problem
ChromaDB uses in-memory storage. Server restart = all indexes lost.
One global "codebase" collection = multiple repos overwrite each other.

### Solution
- Switch to PersistentClient
- Use per-repo collections (one collection per repo)
- Collection names are sanitized repo names

---

### Changes to `rag.py`

#### Remove these lines:
```python
CLIENT = chromadb.Client()
COLLECTION = CLIENT.get_or_create_collection("codebase")
```

#### Add these lines:
```python
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
CLIENT = chromadb.PersistentClient(path=CHROMA_PATH)
_collections: Dict[str, Any] = {}  # repo_name → collection object
```

#### Add new function `get_collection(repo_name: str)`**
```python
def get_collection(repo_name: str):
    """
    Get or create a ChromaDB collection for a specific repo.
    Collection name is sanitized repo name (slashes replaced with underscores).
    
    Args:
        repo_name: GitHub repo name e.g. "royo1019/ticketing-app"
    
    Returns:
        ChromaDB collection object
    
    Edge cases:
        - repo_name with special characters: sanitize to alphanumeric + underscore only
        - repo_name longer than 63 chars (ChromaDB limit): truncate to 63 chars
        - Empty repo_name: raise ValueError
    """
```

Sanitization rules:
- Replace `/` with `_`
- Replace `-` with `_`
- Replace `.` with `_`
- Remove any character that is not alphanumeric or underscore
- If result starts with a number: prepend `r_`
- Truncate to 63 characters max
- If result is empty after sanitization: raise ValueError("Invalid repo name")

#### Update `index_codebase(repo_path: str, repo_name: str = "default")`
- Add `repo_name` parameter
- Get collection via `get_collection(repo_name)`
- Check `collection.count() > 0` to skip re-indexing
- Use this collection instead of global COLLECTION
- Store collection in `_collections[repo_name]`

#### Update `retrieve_context(diff_text: str, repo_name: str = "default", n_results: int = 3)`
- Add `repo_name` parameter
- Get collection via `get_collection(repo_name)`
- Use this collection for querying
- Edge cases:
  - Collection doesn't exist yet (repo not indexed): return [] with warning log
  - n_results > collection.count(): set n_results = collection.count()
  - collection.count() == 0: return [] immediately
  - diff_text is empty string: return [] immediately
  - diff_text is None: return [] immediately

#### Add `delete_collection(repo_name: str)`
```python
def delete_collection(repo_name: str):
    """
    Delete the ChromaDB collection for a repo.
    Used when invalidating cache (repo updated).
    
    Args:
        repo_name: GitHub repo name
    
    Edge cases:
        - Collection doesn't exist: log warning, don't raise
        - ChromaDB error on delete: log error, don't raise
    """
```

#### Add `list_indexed_repos() -> List[str]`
```python
def list_indexed_repos() -> List[str]:
    """
    List all repos currently indexed in ChromaDB.
    Returns list of sanitized collection names.
    """
```

---

### Changes to `repo_cache.py`

Update `get_repo_path(repo_name: str)` to pass `repo_name` to `index_codebase`:
```python
index_codebase(repo_path, repo_name=repo_name)
```

Update `invalidate_cache(repo_name: str)`:
- Call `delete_collection(repo_name)` from rag.py
- Remove from `_cache` dict
- Delete the cloned temp directory
- Log: `logger.info(f"Cache and index invalidated for {repo_name}")`

---

### Changes to `agent.py`

Add `repo_name` to `AgentState`:
```python
repo_name: str  # already exists
```

Update `fetch_rag_node` to pass `repo_name` to `retrieve_context`:
```python
chunks = retrieve_context(f["patch"], repo_name=state["repo_name"])
```

Update `run_agent` to pass `repo_name` to `index_codebase`:
```python
index_codebase(repo_path, repo_name=repo_name)
```

---

### Edge cases for Priority 2

- ChromaDB path doesn't exist: PersistentClient creates it automatically
- ChromaDB path is read-only: catch PermissionError, log error, fall back to in-memory
- Two processes write to same ChromaDB path simultaneously: ChromaDB handles this with file locking
- Disk full when writing to ChromaDB: catch OSError, log error, continue without persisting
- Collection count returns 0 but files exist on disk: re-index (trust count over disk state)
- Repo name is just "/" or empty: raise ValueError before attempting collection creation
- Collection name collision (two different repo names sanitize to same string): append hash of original name to disambiguate

---

## PRIORITY 3 — Async Queue for Concurrent PRs

### Problem
FastAPI BackgroundTasks runs tasks sequentially in a single thread.
If PR #1 is being reviewed (60 seconds), PR #2 for a different repo must wait.
PRs from different repos should be processed in parallel.
PRs from the same repo should be processed sequentially (avoid RAG race conditions).

### Solution
Build an async queue manager that:
- Maintains one queue per repo
- Processes each repo's queue sequentially
- Processes multiple repos' queues concurrently

---

### File to create: `queue_manager.py`

```python
import asyncio
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PRJob:
    repo_name: str
    pr_number: int
    queued_at: datetime = field(default_factory=datetime.now)
    attempts: int = 0
    max_attempts: int = 3
```

#### Class `QueueManager`:

**`__init__(self)`**
```python
self._queues: Dict[str, asyncio.Queue] = {}
# key = repo_name, value = asyncio.Queue of PRJob objects

self._workers: Dict[str, asyncio.Task] = {}
# key = repo_name, value = running asyncio.Task for that repo's worker

self._processing: Dict[str, Optional[PRJob]] = {}
# key = repo_name, value = currently processing job (or None)

self._stats = {
    "total_queued": 0,
    "total_processed": 0,
    "total_failed": 0,
    "total_retried": 0
}

self._lock = asyncio.Lock()
# protect _queues, _workers, _processing from concurrent modification
```

**`async enqueue_pr(self, repo_name: str, pr_number: int) -> bool`**
- Acquire lock
- Create PRJob
- If no queue for repo_name: create `asyncio.Queue(maxsize=50)`
- If queue is full (50 jobs): log warning, return False
- Put job in queue
- Increment `_stats["total_queued"]`
- If no worker for repo_name or worker is done: start new worker
- Release lock
- Log: `logger.info(f"Queued PR #{pr_number} for {repo_name} (queue depth: {queue.qsize()})")`
- Return True
- Edge cases:
  - repo_name is None or empty: log error, return False
  - pr_number <= 0: log error, return False
  - asyncio.Queue creation fails: log error, return False

**`async _worker(self, repo_name: str)`**
- Loop forever:
  - Try to get job from queue with `asyncio.wait_for(queue.get(), timeout=300)`
  - If timeout (5 minutes of empty queue): break out of loop, clean up worker
  - Set `_processing[repo_name] = job`
  - Log: `logger.info(f"Processing PR #{job.pr_number} for {repo_name} (attempt {job.attempts + 1})")`
  - Try to process the job:
    - Import and call `process_pr(repo_name, pr_number)` from server.py
    - On success: increment `_stats["total_processed"]`, log success
    - On exception: increment job.attempts
      - If job.attempts < job.max_attempts: re-queue the job, increment `_stats["total_retried"]`
      - If job.attempts >= job.max_attempts: increment `_stats["total_failed"]`, log error with full traceback
  - Set `_processing[repo_name] = None`
  - Call `queue.task_done()`
  - Add small delay between jobs: `await asyncio.sleep(1)`
- After loop exits: clean up `_workers[repo_name]` and `_queues[repo_name]`
- Edge cases:
  - process_pr raises SystemExit: don't catch, let it propagate
  - process_pr raises KeyboardInterrupt: don't catch, let it propagate
  - process_pr hangs forever: add timeout of 300 seconds (5 minutes)
  - Worker task is cancelled externally: handle CancelledError gracefully

**`async get_status(self) -> Dict`**
- Return:
```python
{
    "queues": {
        repo_name: {
            "depth": queue.qsize(),
            "currently_processing": job.pr_number if job else None,
            "worker_alive": not worker.done()
        }
        for repo_name, queue in self._queues.items()
    },
    "stats": self._stats,
    "total_active_repos": len([w for w in self._workers.values() if not w.done()])
}
```

**`async shutdown(self)`**
- Cancel all worker tasks gracefully
- Wait for all queues to be empty (with 30 second timeout)
- Log: `logger.info("QueueManager shutdown complete")`

#### Module-level singleton:
```python
queue_manager = QueueManager()
```

---

### Changes to `server.py`

Add import:
```python
from queue_manager import queue_manager
```

Add startup event:
```python
@app.on_event("startup")
async def startup_event():
    logger.info("Server starting up")
    # queue_manager is ready to use immediately
```

Add shutdown event:
```python
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Server shutting down")
    await queue_manager.shutdown()
```

Replace in webhook handler:
```python
# OLD:
background_tasks.add_task(process_pr, repo_name, pr_number)

# NEW:
success = await queue_manager.enqueue_pr(repo_name, pr_number)
if not success:
    return {"status": "rejected", "reason": "queue full for this repo"}
```

Add new endpoint:
```python
@app.get("/status")
async def get_status():
    """
    Returns current queue status and processing stats.
    Useful for monitoring and debugging.
    """
    return await queue_manager.get_status()
```

---

### Changes to `process_pr` function in `server.py`

Move it out of async and make it a regular function so it can be called from queue:
```python
def process_pr(repo_name: str, pr_number: int):
    """
    Process a single PR review.
    Called by QueueManager worker.
    
    Args:
        repo_name: GitHub repo full name e.g. "owner/repo"
        pr_number: PR number
    
    Raises:
        Exception: any error during processing (caller handles retry)
    """
    try:
        logger.info(f"Starting review of PR #{pr_number} for {repo_name}")
        repo_path = get_repo_path(repo_name)
        run_agent(repo_name, pr_number, repo_path)
        logger.info(f"Completed review of PR #{pr_number} for {repo_name}")
    except Exception as e:
        logger.error(f"Failed to review PR #{pr_number} for {repo_name}: {e}", exc_info=True)
        raise  # let queue_manager handle retry
```

---

### Edge cases for Priority 3

- Queue full (50 jobs for same repo): return 429-like response, do not crash
- Worker crashes mid-job: catch exception, retry job, restart worker
- Server receives 100 webhook pings simultaneously: all get queued, processed in order per repo
- Same PR number queued twice for same repo: allow it (GitHub may send duplicate webhooks)
- Worker timeout (job takes > 5 minutes): cancel the job, log timeout error, mark as failed
- asyncio event loop not running when enqueue_pr called: catch RuntimeError, log error
- Queue manager shutdown called while jobs are running: wait for current job to finish, cancel queued jobs

---

## PRIORITY 4 — RAG Index Refresh on Push

### Problem
When developers merge PRs and push to main, the RAG index becomes stale.
The agent keeps recommending patterns from old deleted code.

### Solution
Handle GitHub `push` webhook events.
When a push to the default branch is detected, invalidate and rebuild the index.

---

### Changes to `server.py`

Update webhook handler to also handle push events:

```python
@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.body()
    
    # validate signature
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not validate_signature(payload, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    event = request.headers.get("X-GitHub-Event", "")
    data = await request.json()
    
    if event == "pull_request":
        return await handle_pull_request(data)
    elif event == "push":
        return await handle_push(data)
    elif event == "ping":
        return {"status": "pong", "message": "Webhook configured successfully"}
    else:
        logger.debug(f"Ignoring event: {event}")
        return {"status": "ignored", "reason": f"event {event} not handled"}
```

**`async handle_pull_request(data: dict) -> dict`**
- Extract action = data.get("action")
- If action not in ["opened", "synchronize", "reopened"]: return ignored
- Extract repo_name = data["repository"]["full_name"]
- Extract pr_number = data["pull_request"]["number"]
- Extract pr_state = data["pull_request"]["state"]
- If pr_state == "closed": return ignored
- Extract is_draft = data["pull_request"].get("draft", False)
- If is_draft: return {"status": "ignored", "reason": "draft PRs not reviewed"}
- Enqueue via queue_manager
- Edge cases:
  - data["repository"] missing: catch KeyError, return 400
  - data["pull_request"] missing: catch KeyError, return 400
  - pr_number is not an integer: catch ValueError, return 400

**`async handle_push(data: dict) -> dict`**
- Extract repo_name = data["repository"]["full_name"]
- Extract ref = data.get("ref", "")  # e.g. "refs/heads/main"
- Extract default_branch = data["repository"].get("default_branch", "main")
- Check if push is to default branch: `ref == f"refs/heads/{default_branch}"`
- If not default branch: return {"status": "ignored", "reason": "not default branch"}
- Extract commits = data.get("commits", [])
- If no commits (e.g. tag push): return ignored
- Extract changed_files from commits:
  ```python
  changed_files = []
  for commit in commits:
      changed_files.extend(commit.get("added", []))
      changed_files.extend(commit.get("modified", []))
      changed_files.extend(commit.get("removed", []))
  ```
- Check if any .py, .js, .ts files changed (only re-index if code changed, not just README)
- If no code files changed: return {"status": "ignored", "reason": "no code files changed"}
- Schedule cache invalidation as background task:
  ```python
  background_tasks.add_task(refresh_repo_index, repo_name)
  ```
- Return {"status": "accepted", "action": "index_refresh_scheduled", "repo": repo_name}
- Edge cases:
  - data["repository"] missing: catch KeyError, return 400
  - ref is None or empty: treat as non-default branch push, ignore
  - commits is not a list: catch TypeError, log warning, ignore

**`def refresh_repo_index(repo_name: str)`**
```python
def refresh_repo_index(repo_name: str):
    """
    Invalidate cached index for a repo and rebuild it.
    Called when push to default branch detected.
    
    Args:
        repo_name: GitHub repo full name
    
    Edge cases:
        - Clone fails (repo deleted or private): log error, clear cache anyway
        - Index build fails: log error, leave cache empty (next PR will trigger rebuild)
        - Repo not in cache (never been reviewed): just log, do nothing
    """
    try:
        logger.info(f"Refreshing index for {repo_name}")
        invalidate_cache(repo_name)
        # get_repo_path will re-clone and re-index
        get_repo_path(repo_name)
        logger.info(f"Index refresh complete for {repo_name}")
    except Exception as e:
        logger.error(f"Failed to refresh index for {repo_name}: {e}", exc_info=True)
```

---

### Edge cases for Priority 4

- Push event with 0 commits (force push that deletes branch): ignore
- Push event to a branch that doesn't exist in default_branch field: ignore
- Two push events arrive simultaneously for same repo: queue the refresh, process sequentially
- Repo is private and GITHUB_TOKEN doesn't have access: catch 403, log error clearly
- Push event with 1000 files changed: still refresh (don't check file count limit)
- Default branch is "master" not "main": handle both (use data["repository"]["default_branch"])
- data["repository"]["default_branch"] is missing: default to "main"

---

## PRIORITY 5 — Complete Test Suite

### Directory structure to create:
```
tests/
├── __init__.py
├── conftest.py          ← pytest fixtures
├── test_linter.py       ← unit tests for linter.py
├── test_rag.py          ← unit tests for rag.py
├── test_ast_parser.py   ← unit tests for ast_parser.py
├── test_llm.py          ← unit tests for llm_reviewer.py
├── test_github.py       ← unit tests for github_client.py
├── test_queue.py        ← unit tests for queue_manager.py
├── test_server.py       ← unit tests for server.py (webhook handling)
├── test_agent.py        ← integration tests for full pipeline
└── eval_suite.py        ← LLM quality evaluation
```

---

### `tests/conftest.py`

```python
import pytest
import os
import tempfile

@pytest.fixture
def temp_python_file():
    """Creates a temporary Python file with content."""
    def _make_file(content: str, filename: str = "test.py"):
        temp_dir = tempfile.mkdtemp()
        filepath = os.path.join(temp_dir, filename)
        with open(filepath, "w") as f:
            f.write(content)
        return filepath
    return _make_file

@pytest.fixture
def sql_injection_code():
    return '''
def get_user(username, password):
    conn = sqlite3.connect("users.db")
    query = "SELECT * FROM users WHERE username='" + username + "'"
    user = conn.execute(query).fetchone()
    if user and user[2] == password:
        return user
    return None
'''

@pytest.fixture
def clean_code():
    return '''
def get_user(username, password):
    conn = sqlite3.connect("users.db")
    user = conn.execute(
        "SELECT * FROM users WHERE username=?", 
        (username,)
    ).fetchone()
    if user and verify_password(password, user[2]):
        return user
    return None
'''

@pytest.fixture
def mock_pr_diff():
    return """+def cancel_ticket(ticket_id, user_id):
+    query = "SELECT * FROM tickets WHERE id=" + ticket_id
+    ticket = db.execute(query).fetchone()
+    if ticket == None:
+        return False
+    return True"""

@pytest.fixture
def sample_repo(tmp_path):
    """Creates a minimal fake repo for testing."""
    # auth.py
    (tmp_path / "auth.py").write_text('''
import sqlite3

def get_user(username, password):
    conn = sqlite3.connect("users.db")
    user = conn.execute(
        "SELECT * FROM users WHERE username=?",
        (username,)
    ).fetchone()
    return user
''')
    # payment.py
    (tmp_path / "payment.py").write_text('''
def process_refund(ticket_id, amount):
    refund_pct = 0.9
    return {"refund": amount * refund_pct}
''')
    # models.py
    (tmp_path / "models.py").write_text('''
TICKETS_TABLE = "tickets"
REFUNDS_TABLE = "refunds"
''')
    return str(tmp_path)
```

---

### `tests/test_linter.py` — Complete test cases

```
Test: test_bandit_catches_sql_injection_string_concat
  Input: code with "SELECT * FROM users WHERE id='" + user_id + "'"
  Expected: at least one finding with tool="bandit" and "SQL" in message

Test: test_bandit_catches_sql_injection_fstring
  Input: code with f"SELECT * FROM users WHERE id={user_id}"
  Expected: at least one bandit finding

Test: test_bandit_catches_hardcoded_password
  Input: code with PASSWORD = "admin123"
  Expected: at least one bandit finding mentioning "password" or "hardcoded"

Test: test_bandit_catches_eval
  Input: code with eval(user_input)
  Expected: at least one bandit finding

Test: test_flake8_catches_unused_import
  Input: "import os\nimport sys\n\ndef foo():\n    pass\n"
  Expected: findings with "imported but unused" in message

Test: test_flake8_catches_none_comparison
  Input: code with "if x == None:"
  Expected: finding with "E711" in message

Test: test_flake8_catches_bare_except
  Input: code with "except:"
  Expected: finding mentioning bare except

Test: test_clean_code_has_no_bandit_findings
  Input: properly written code with parameterized queries
  Expected: zero bandit findings

Test: test_unsupported_extension_returns_empty
  Input: filename="test.java", any content
  Expected: empty list returned (Java not yet supported by linters)

Test: test_empty_file_returns_empty
  Input: empty string content
  Expected: empty list or only whitespace-related findings

Test: test_linter_handles_syntax_error_gracefully
  Input: "def foo(:\n    pass" (invalid syntax)
  Expected: does not raise, returns list (may be empty or have syntax error finding)

Test: test_linter_handles_none_content_gracefully
  Input: content=None
  Expected: does not raise, returns empty list

Test: test_linter_handles_very_large_file
  Input: file with 10000 lines
  Expected: completes without timeout or crash

Test: test_flake8_line_numbers_are_correct
  Input: code where unused import is on line 3
  Expected: finding with line=3
```

---

### `tests/test_ast_parser.py` — Complete test cases

```
Test: test_extract_functions_basic
  Input: simple function def
  Expected: FunctionInfo with correct name, lineno, args

Test: test_extract_functions_with_args
  Input: def foo(a, b, c=None, *args, **kwargs)
  Expected: args = ["a", "b", "c", "args", "kwargs"] (excluding self/cls)

Test: test_extract_functions_detects_calls
  Input: function that calls bar() and baz()
  Expected: calls = ["bar", "baz"]

Test: test_extract_functions_method_detection
  Input: class with method
  Expected: is_method=True, class_name set correctly

Test: test_extract_functions_async
  Input: async def foo()
  Expected: extracted same as regular function

Test: test_extract_classes_basic
  Input: simple class definition
  Expected: ClassInfo with correct name, lineno

Test: test_extract_classes_with_bases
  Input: class Foo(Bar, Baz)
  Expected: bases = ["Bar", "Baz"]

Test: test_extract_imports_regular
  Input: import os
  Expected: ImportInfo with module="os", is_from_import=False

Test: test_extract_imports_from
  Input: from payment import process_refund
  Expected: ImportInfo with module="payment", names=["process_refund"], is_from_import=True

Test: test_extract_imports_relative
  Input: from . import utils
  Expected: ImportInfo with module="." or ".utils"

Test: test_parse_file_syntax_error
  Input: file with invalid Python syntax
  Expected: returns None, does not raise

Test: test_parse_file_empty
  Input: empty file
  Expected: returns ast.Module (valid empty module)

Test: test_build_index_empty_dir
  Input: directory with no .py files
  Expected: returns ASTIndex with empty dicts

Test: test_build_index_skips_venv
  Input: directory with venv/ subfolder containing .py files
  Expected: venv files not in index

Test: test_build_index_skips_large_files
  Input: .py file larger than 1MB
  Expected: file not in index

Test: test_get_function_context_not_found
  Input: function name that doesn't exist
  Expected: returns {"found": False, ...}

Test: test_get_function_context_found
  Input: function name that exists
  Expected: returns {"found": True, "definitions": [...]}

Test: test_call_graph_built_correctly
  Input: function A calls function B
  Expected: call_graph["filepath::A"] contains "B"

Test: test_reverse_call_graph_built_correctly
  Input: function A calls function B
  Expected: reverse_call_graph["B"] contains "filepath::A"

Test: test_format_ast_context_detects_functions_in_diff
  Input: diff containing "cancel_ticket(" 
  Expected: output string contains "cancel_ticket"

Test: test_format_ast_context_empty_diff
  Input: empty diff string
  Expected: returns non-empty string with header at minimum

Test: test_format_ast_context_truncates_long_output
  Input: diff with 50 different function names all in index
  Expected: output length <= 2000 chars
```

---

### `tests/test_llm.py` — Complete test cases

```
Test: test_review_pr_returns_list
  Input: valid diff, empty lint findings, empty rag chunks
  Expected: returns list (may be empty)

Test: test_review_pr_returns_correct_structure
  Input: diff with obvious SQL injection
  Expected: each comment has "line", "severity", "comment" keys

Test: test_review_pr_severity_values_valid
  Input: any valid diff
  Expected: all severities are one of "critical", "warning", "suggestion"

Test: test_review_pr_line_numbers_are_integers
  Input: any valid diff
  Expected: all line values are integers >= 1

Test: test_review_pr_catches_sql_injection
  Input: diff with string-concatenated SQL query
  Expected: at least one comment with severity="critical" mentioning SQL

Test: test_review_pr_handles_empty_diff
  Input: empty string diff
  Expected: returns list (may be empty), does not raise

Test: test_review_pr_handles_none_lint_findings
  Input: lint_findings=None
  Expected: does not raise (handle None gracefully)

Test: test_review_pr_handles_none_rag_chunks
  Input: rag_chunks=None
  Expected: does not raise

Test: test_call_llm_retries_on_json_error
  Input: mock Groq to return invalid JSON twice, then valid JSON
  Expected: returns the valid JSON result on third attempt

Test: test_call_llm_returns_empty_after_max_retries
  Input: mock Groq to always return invalid JSON
  Expected: returns [] after 3 attempts

Test: test_build_prompt_includes_diff
  Input: diff="+ some code here"
  Expected: prompt string contains "some code here"

Test: test_build_prompt_includes_lint_findings
  Input: lint_findings with one bandit finding
  Expected: prompt string contains the finding message

Test: test_build_prompt_includes_rag_context
  Input: rag_chunks with one chunk from auth.py
  Expected: prompt string contains "auth.py"
```

---

### `tests/test_server.py` — Complete test cases

Use FastAPI TestClient for all tests.

```
Test: test_health_check
  GET /
  Expected: 200, body has "status": "running"

Test: test_webhook_invalid_signature
  POST /webhook with wrong X-Hub-Signature-256
  Expected: 401

Test: test_webhook_ping_event
  POST /webhook with X-GitHub-Event: ping
  Expected: 200, body has "status": "pong"

Test: test_webhook_pr_opened
  POST /webhook with valid PR opened payload
  Expected: 200, body has "status": "accepted"

Test: test_webhook_pr_closed_ignored
  POST /webhook with PR closed action
  Expected: 200, body has "status": "ignored"

Test: test_webhook_pr_draft_ignored
  POST /webhook with draft=true in payload
  Expected: 200, body has "status": "ignored", reason mentions "draft"

Test: test_webhook_push_to_main
  POST /webhook with push to refs/heads/main with .py files changed
  Expected: 200, body has "action": "index_refresh_scheduled"

Test: test_webhook_push_to_branch_ignored
  POST /webhook with push to refs/heads/feature-branch
  Expected: 200, body has "status": "ignored"

Test: test_webhook_push_no_code_files
  POST /webhook with push that only changes README.md
  Expected: 200, body has "status": "ignored", reason mentions "no code files"

Test: test_webhook_unknown_event_ignored
  POST /webhook with X-GitHub-Event: release
  Expected: 200, body has "status": "ignored"

Test: test_webhook_missing_repository_field
  POST /webhook with PR payload missing "repository" key
  Expected: 400

Test: test_status_endpoint
  GET /status
  Expected: 200, body has "queues" and "stats" keys
```

---

### `eval_suite.py` — LLM Quality Evaluation

```python
"""
Evaluation suite for measuring agent review quality.
Run with: python eval_suite.py

Measures:
- Critical issue detection rate
- False positive rate  
- Average comments per PR
- Consistency across 3 runs
"""
```

#### Test cases to define (10 minimum):

```python
TEST_CASES = [
    {
        "name": "SQL injection via string concat",
        "code": '...',
        "must_catch": ["SQL", "injection", "parameterized"],  # any of these in comment
        "expected_severity": "critical",
        "should_not_flag": []  # false positive check
    },
    {
        "name": "SQL injection via f-string",
        "code": '...',
        "must_catch": ["SQL", "injection"],
        "expected_severity": "critical",
        "should_not_flag": []
    },
    {
        "name": "Hardcoded password",
        "code": 'PASSWORD = "admin123"',
        "must_catch": ["password", "hardcoded", "secret"],
        "expected_severity": "critical",
        "should_not_flag": []
    },
    {
        "name": "Eval with user input",
        "code": 'eval(user_input)',
        "must_catch": ["eval", "dangerous", "arbitrary"],
        "expected_severity": "critical",
        "should_not_flag": []
    },
    {
        "name": "None comparison with ==",
        "code": 'if x == None:',
        "must_catch": ["None", "is None"],
        "expected_severity": "warning",
        "should_not_flag": []
    },
    {
        "name": "Bare except clause",
        "code": 'try:\n    foo()\nexcept:\n    pass',
        "must_catch": ["except", "specific"],
        "expected_severity": "warning",
        "should_not_flag": []
    },
    {
        "name": "Division by zero risk",
        "code": 'def discount(price, pct):\n    return price * pct / pct',
        "must_catch": ["division", "zero", "ZeroDivision"],
        "expected_severity": "critical",
        "should_not_flag": []
    },
    {
        "name": "Unused variable",
        "code": 'def foo():\n    x = 5\n    return 10',
        "must_catch": ["unused", "x"],
        "expected_severity": "warning",
        "should_not_flag": []
    },
    {
        "name": "Clean code should not be flagged critical",
        "code": '''
def get_user(username):
    conn = sqlite3.connect("users.db")
    user = conn.execute(
        "SELECT * FROM users WHERE username=?",
        (username,)
    ).fetchone()
    return user
''',
        "must_catch": [],  # nothing must be caught
        "expected_severity": None,
        "should_not_flag": ["SQL injection", "critical"]  # must not say SQL injection
    },
    {
        "name": "Missing input validation",
        "code": '''
def transfer_money(amount, to_account):
    db.execute("UPDATE accounts SET balance = balance - ? WHERE id=1", (amount,))
    db.execute("UPDATE accounts SET balance = balance + ? WHERE id=?", (amount, to_account))
''',
        "must_catch": ["validation", "negative", "amount"],
        "expected_severity": "warning",
        "should_not_flag": []
    }
]
```

#### Evaluation logic:

```python
def run_eval(runs: int = 3) -> Dict:
    """
    Run evaluation suite multiple times to account for LLM non-determinism.
    
    Args:
        runs: number of times to run each test case
    
    Returns:
        {
            "overall_score": float,  # 0.0 to 1.0
            "critical_detection_rate": float,
            "false_positive_rate": float,
            "per_case_results": [...],
            "runs": runs
        }
    """
```

For each test case across all runs:
- Run `review_pr(code, lint_findings, [])`
- Check if any comment contains any word from `must_catch`
- Check if no comment contains words from `should_not_flag`
- Score = (cases passed) / (total cases * runs)
- Print detailed report showing which cases fail and why
- Target: 80%+ overall score, 90%+ critical detection rate

---

### Install pytest and run:

Add to requirements.txt:
```
pytest
pytest-asyncio
pytest-mock
httpx  # for FastAPI TestClient
```

Add `pytest.ini` to project root:
```ini
[pytest]
testpaths = tests
asyncio_mode = auto
log_cli = true
log_cli_level = INFO
```

---

## PRIORITY 6 — GitHub App (One-Click Installation)

### Problem
Users must manually:
1. Go to repo Settings → Webhooks
2. Add your URL
3. Configure secret
4. Select events

A GitHub App allows:
1. User goes to your app page
2. Clicks Install
3. Selects repos
4. Done — fully automated

### What to build

#### Register GitHub App (manual step — document in README)
```
Name: CodeReviewBot
Homepage URL: https://your-app.fly.dev
Webhook URL: https://your-app.fly.dev/webhook
Webhook secret: (same as WEBHOOK_SECRET env var)
Permissions needed:
  - Pull requests: Read & Write
  - Contents: Read
  - Metadata: Read
Events to subscribe:
  - Pull request
  - Push
```

#### File to create: `github_app.py`

**`generate_jwt(app_id: str, private_key: str) -> str`**
- Generate a JWT token for GitHub App authentication
- Use PyJWT library
- Payload: {"iat": now-60, "exp": now+600, "iss": app_id}
- Algorithm: RS256
- Edge cases:
  - private_key is None: raise ValueError("GitHub App private key not configured")
  - private_key is invalid PEM: catch jwt.exceptions.InvalidKeyError, raise with clear message
  - app_id is None or empty: raise ValueError

**`get_installation_token(app_id: str, private_key: str, installation_id: int) -> str`**
- Use JWT to get an installation access token
- POST https://api.github.com/app/installations/{installation_id}/access_tokens
- Returns token that expires in 1 hour
- Cache tokens per installation_id (avoid regenerating on every PR)
- Edge cases:
  - installation_id not found (app uninstalled): raise clear error
  - GitHub API rate limit hit: retry with exponential backoff (1s, 2s, 4s)
  - Token generation fails: raise with full GitHub error message
  - Network timeout: retry 3 times

**`get_github_client_for_installation(installation_id: int) -> Github`**
- Generate JWT → get installation token → return Github(token)
- Cache the Github object per installation_id for 50 minutes (token lasts 60)
- Edge cases:
  - installation_id is None: raise ValueError
  - installation_id is 0 or negative: raise ValueError

#### New environment variables to add:
```
GITHUB_APP_ID=123456
GITHUB_APP_PRIVATE_KEY=-----BEGIN RSA PRIVATE KEY-----\n...
GITHUB_APP_INSTALLATION_ID=  # optional, for single-repo mode
```

#### Changes to `server.py`

Handle installation event:
```python
elif event == "installation":
    return await handle_installation(data)
```

**`async handle_installation(data: dict) -> dict`**
- Extract action = data["action"]
- If action == "created":
  - Log new installation
  - Extract installation_id = data["installation"]["id"]
  - Extract repos = data["repositories"] (list of repos just installed on)
  - For each repo: schedule background index build
  - Return {"status": "accepted", "action": "installation_recorded"}
- If action == "deleted":
  - Extract repos previously installed on
  - For each repo: invalidate cache
  - Return {"status": "accepted", "action": "installation_removed"}
- Other actions: return ignored
- Edge cases:
  - data["installation"] missing: return 400
  - data["repositories"] missing or empty: return accepted with note "no repos to index"

#### Changes to `github_client.py`

Support both PAT and GitHub App authentication:
```python
def get_github_client(installation_id: int = None) -> Github:
    """
    Get GitHub client.
    Uses GitHub App if GITHUB_APP_ID is set and installation_id provided.
    Falls back to Personal Access Token.
    
    Args:
        installation_id: GitHub App installation ID (optional)
    
    Returns:
        Authenticated Github client
    """
    app_id = os.getenv("GITHUB_APP_ID")
    private_key = os.getenv("GITHUB_APP_PRIVATE_KEY")
    
    if app_id and private_key and installation_id:
        return get_github_client_for_installation(installation_id)
    
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("No GitHub authentication configured. Set GITHUB_TOKEN or GITHUB_APP_* vars.")
    return Github(token)
```

---

### Edge cases for Priority 6

- Private key has Windows line endings (\r\n): normalize to \n before use
- Private key stored as base64 in env var: detect and decode
- Installation token expires mid-review: refresh token, retry the failed API call
- App installed on org with 500 repos: index all in background, log progress
- App uninstalled while review is in progress: catch 401, abort gracefully
- Multiple installations for same app: handle each installation_id independently

---

## PRIORITY 7 — Deployment on Fly.io

### Steps to automate (create `deploy.sh`):

```bash
#!/bin/bash
set -e

echo "Deploying to Fly.io..."

# Check flyctl installed
if ! command -v flyctl &> /dev/null; then
    echo "flyctl not installed. Run: brew install flyctl"
    exit 1
fi

# Check logged in
flyctl auth whoami || { echo "Not logged in. Run: flyctl auth login"; exit 1; }

# Create volume for ChromaDB persistence
flyctl volumes create chroma_data --size 1 --region bom 2>/dev/null || echo "Volume already exists"

# Set secrets
flyctl secrets set \
    GROQ_API_KEY="$GROQ_API_KEY" \
    GITHUB_TOKEN="$GITHUB_TOKEN" \
    WEBHOOK_SECRET="$WEBHOOK_SECRET" \
    CHROMA_PATH="/data/chroma_db" \
    LOG_LEVEL="INFO"

# Deploy
flyctl deploy --remote-only

echo "Deployment complete!"
flyctl status
```

### Update `fly.toml`:

```toml
app = 'code-review-agent'
primary_region = 'bom'

[build]
  dockerfile = 'Dockerfile'

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = 'stop'
  auto_start_machines = true
  min_machines_running = 1

[http_service.concurrency]
  type = "requests"
  hard_limit = 25
  soft_limit = 20

[[mounts]]
  source = "chroma_data"
  destination = "/data"

[[vm]]
  memory = '2gb'
  cpu_kind = 'shared'
  cpus = 2

[checks]
  [checks.health]
    grace_period = "30s"
    interval = "15s"
    method = "GET"
    path = "/"
    port = 8000
    timeout = "10s"
    type = "http"
```

### Update `Dockerfile` for production:

```dockerfile
FROM python:3.11-slim

# install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g eslint

WORKDIR /app

# install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pre-download embedding model so first request isn't slow
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('microsoft/codebert-base')"

COPY . .

# create directory for ChromaDB (overridden by Fly volume mount)
RUN mkdir -p /data/chroma_db

EXPOSE 8000

# use gunicorn with uvicorn workers for production
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "info"]
```

---

## PRIORITY 8 — README

### File: `README.md`

Must contain these sections in order:

#### 1. Header
- Project name: **CodeReviewBot**
- One-line description: "An autonomous AI agent that reviews GitHub Pull Requests like a senior engineer."
- Badges: Python version, License, Docker

#### 2. Demo screenshot
- Show a real GitHub PR with inline comments posted by the agent
- Use screenshots from your test PRs

#### 3. What it does (3 bullets)
- Detects security vulnerabilities (SQL injection, hardcoded secrets, eval)
- Retrieves codebase context using RAG — gives specific advice referencing your existing code
- Posts inline comments + summary directly on GitHub PR within 60 seconds

#### 4. Architecture diagram (ASCII)
```
GitHub PR Opened
      │
      ▼
FastAPI Webhook Server
      │
      ▼
LangGraph State Machine
      │
  ┌───┴───────────────────┐
  │                       │
  ▼                       ▼
Flake8 + Bandit      CodeBERT + ChromaDB
(Static Analysis)    (RAG Retrieval)
  │                       │
  └───────────┬───────────┘
              │
              ▼
    Groq LLaMA 3.3 70B
    (Review Generation)
              │
              ▼
    GitHub PR Comments
```

#### 5. Tech stack table
| Layer | Technology |
|-------|-----------|
| Agent framework | LangGraph |
| LLM | Groq LLaMA 3.3 70B (free) |
| Embeddings | Microsoft CodeBERT |
| Vector DB | ChromaDB |
| Static analysis | Flake8 + Bandit + ESLint |
| Web server | FastAPI |
| Container | Docker |

#### 6. How to use it (two methods)

**Method 1: CLI**
```bash
git clone https://github.com/royo1019/code-review-agent
cd code-review-agent
pip install -r requirements.txt
cp .env.example .env  # fill in your keys
python3 main.py --repo owner/repo --pr 1 --path /local/clone/of/repo
```

**Method 2: Webhook server**
```bash
# Start server
uvicorn server:app --port 8000

# Expose publicly (for testing)
ngrok http 8000

# Add webhook in GitHub repo settings:
# URL: https://your-url/webhook
# Secret: your WEBHOOK_SECRET
# Events: Pull requests, Pushes
```

#### 7. Environment variables table
| Variable | Required | Description |
|----------|---------|-------------|
| GROQ_API_KEY | Yes | Free at console.groq.com |
| GITHUB_TOKEN | Yes | PAT with repo scope |
| WEBHOOK_SECRET | Yes | Any random string |
| EMBEDDING_MODEL | No | Defaults to microsoft/codebert-base |
| CHROMA_PATH | No | Defaults to ./chroma_db |

#### 8. Limitations
- Python static analysis only (Flake8 + Bandit); JS/TS via ESLint; other languages LLM-only
- RAG retrieval limited to text similarity; does not parse import graphs (AST parser in progress)
- Webhook server requires manual setup (GitHub App coming soon)

#### 9. Roadmap
- [ ] GitHub App for one-click installation
- [ ] AST-based call graph analysis
- [ ] Java, Go, Ruby linter support
- [ ] Per-team custom rule configuration
- [ ] Slack/Teams notifications

#### 10. Contributing + License
MIT License. PRs welcome.

---

## LOGGING REQUIREMENTS (applies to all priorities)

Replace all `print()` calls in existing files with proper logging.

Setup in each file:
```python
import logging
logger = logging.getLogger(__name__)
```

In `server.py` startup, configure root logger:
```python
import logging
import sys

def setup_logging():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    # silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
```

Log levels to use:
- `logger.debug()` — detailed internal state, loop iterations
- `logger.info()` — key events (PR received, review posted, index built)
- `logger.warning()` — recoverable issues (file skipped, retry attempted)
- `logger.error()` — failures with exc_info=True for full traceback
- Never use `logger.exception()` outside except blocks

---

## ERROR HANDLING REQUIREMENTS (applies to all priorities)

Every function that calls external services must follow this pattern:
```python
try:
    result = external_call()
    return result
except SpecificException as e:
    logger.error(f"Specific thing failed: {e}", exc_info=True)
    return default_value  # never return None unless documented
except Exception as e:
    logger.error(f"Unexpected error in function_name: {e}", exc_info=True)
    return default_value
```

Never:
- Swallow exceptions silently (bare `except: pass`)
- Return None without documenting it
- Let an exception from one PR review crash the entire server
- Expose internal error details in HTTP responses (return generic message, log detail)

---

## FINAL CHECKLIST FOR CLAUDE CODE

Before marking any Priority complete:

- [ ] All new functions have docstrings
- [ ] All edge cases listed are handled
- [ ] No print() statements — all logging uses logger
- [ ] All external calls wrapped in try/except
- [ ] Existing test files still pass (test_linter.py, test_rag.py, test_llm.py, test_post.py)
- [ ] New test file for the priority passes
- [ ] No hardcoded values — all config via environment variables
- [ ] Type hints on all function signatures
- [ ] requirements.txt updated with any new dependencies
- [ ] README updated to reflect new capabilities
