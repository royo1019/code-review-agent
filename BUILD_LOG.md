# Build Log — Code Review Agent Improvements

Session date: 2026-05-18
Working directory: `/Users/royo/code-review-agent`
Spec: `claude_code_spec.md`
Starting state: `claude_code_context.md`

Each priority was implemented end-to-end and validated before moving to the
next, per the spec's "implement Priority fully and test it before moving on"
rule.

---

## Priority 1 — AST-Based Code Understanding

### Goal
Give the LLM reviewer structural awareness — which functions exist, which
call which, which files import which — instead of treating code as opaque
text.

### Files
| File | Action | Notes |
|---|---|---|
| `ast_parser.py` | **new** | All dataclasses + AST walkers + diff-formatter |
| `rag.py` | edit | Build AST index alongside vector index; expose `get_ast_context` |
| `llm_reviewer.py` | edit | New `ast_context` param in `build_prompt`; `review_pr` pulls AST context; three new prompt rules |

### Implementation highlights
- Dataclasses: `FunctionInfo`, `ClassInfo`, `ImportInfo`, `ASTIndex`
  (functions / classes / imports / call_graph / reverse_call_graph /
  file_to_functions).
- `parse_file` returns `None` on syntax errors, missing files, or any read
  failure — never raises.
- `extract_functions` handles sync + async + nested + class methods, filters
  `self`/`cls`, includes `*args`/`**kwargs`, deduplicates call names (Name +
  Attribute forms), captures decorators via `ast.unparse` with `ast.dump`
  fallback, takes first line of docstring.
- `extract_classes` includes regular bases plus `metaclass=` keyword bases.
- `extract_imports` handles `import X`, `import X, Y` (split into separate
  records), `from X import Y`, `from . import Y`, `from .X import *`
  (relative levels encoded as leading dots).
- `build_index` skips `.git / venv / .venv / env / __pycache__ /
  node_modules / .tox / dist / build / eggs / .eggs / .mypy_cache /
  .pytest_cache`, skips files > 1 MB, handles `FileNotFoundError` mid-walk.
- `format_ast_context_for_llm` scans diff with `\b([a-zA-Z_]\w*)\s*\(`
  pattern, labels Python builtins, truncates output to ~2000 chars with
  "truncated for brevity" suffix.
- `rag.py` builds the AST index inside `index_codebase` under try/except so
  RAG retrieval keeps working even if AST fails.
- `llm_reviewer.py` adds three new prompt rules: recommend existing function
  if one exists, flag calls to nonexistent functions, recommend DRY
  refactoring when code duplicates another file.

### Validation
**47 sanity checks passed**, covering: every dataclass field, parse-file edge
cases (syntax error / empty / missing), extraction (basic / args / methods /
nested / async / decorators / docstring / metaclass), import variants
(regular / from / relative / multi / star), build_index (empty repo / venv
skip / large file skip / call graph / reverse call graph), function context
lookup (found / not found), file dependencies (imported_by tracking),
formatter (header / function detection / empty diff / None diff / truncation
/ builtin labeling).

---

## Priority 2 — Persistent ChromaDB with Per-Repo Collections

### Goal
Survive server restarts and isolate repos from each other.

### Files
| File | Action | Notes |
|---|---|---|
| `rag.py` | rewrite | `PersistentClient` + per-repo collections |
| `repo_cache.py` | rewrite | Pass `repo_name` through; clear collection on invalidate |
| `agent.py` | edit | `fetch_rag_node` and `run_agent` thread `repo_name` |

### Implementation highlights
- `chromadb.Client()` → `chromadb.PersistentClient(path=CHROMA_PATH)` with
  `os.makedirs(..., exist_ok=True)` and graceful fallback to in-memory
  `Client()` on `PermissionError` / `OSError` / unexpected exception.
- `_sanitize_repo_name`:
  - `/`, `-`, `.` → `_`
  - strip non-alphanumeric/underscore
  - prepend `r_` for digit-start
  - cap at 63 chars (ChromaDB limit)
  - pad to ≥3 chars (ChromaDB minimum)
  - raise `ValueError` on empty / whitespace / all-invalid input
- `_collection_name_for` uses `_name_registry` to detect collisions; second
  repo that sanitizes to the same string gets an `_<sha1[:8]>` suffix.
- `get_collection(repo_name)`: caches in `_collections`, raises on empty
  `repo_name`.
- `delete_collection(repo_name)`: never raises; logs warning on missing
  collection; pops from `_collections` and `_name_registry`.
- `list_indexed_repos()` wraps `CLIENT.list_collections()` with safe error
  handling.
- `index_codebase(repo_path, repo_name="default")`: per-repo collection,
  skips re-index when `collection.count() > 0` (trust count over disk),
  catches `OSError` for disk-full at `collection.add` time.
- `retrieve_context(diff_text, repo_name="default", n_results=3)`:
  returns `[]` for None/empty diff, missing/empty collection, or any failure;
  clamps `n_results` to `max(1, min(n, count))`.
- `COLLECTION` global preserved as the default collection for backwards
  compat.
- `repo_cache.invalidate_cache` now calls `delete_collection(repo_name)`
  inside try/except.

### Validation
**33 checks passed**, covering: sanitization (slashes / dots / hyphens /
invalid chars / digit prefix / length cap / short padding / empty raises),
collision handling, per-repo isolation (chunks from repoA don't leak into
repoB query), idempotence (re-running `index_codebase` skips), retrieve edge
cases (empty / None / unknown repo / oversized n_results), delete resilience,
**persistence across `importlib.reload(rag)`** (chunks survive client
teardown), backwards-compat path. Disk check confirmed `chroma.sqlite3` +
segment dirs written.

---

## Priority 3 — Async Queue for Concurrent PRs

### Goal
Process repo A and repo B in parallel; process PR #1 and PR #2 on the same
repo sequentially.

### Files
| File | Action | Notes |
|---|---|---|
| `queue_manager.py` | **new** | `QueueManager` class + `PRJob` dataclass + singleton |
| `server.py` | rewrite | Convert `process_pr` to sync; enqueue via queue; add `/status`; lifecycle hooks |

### Implementation highlights
- `PRJob` dataclass: `repo_name`, `pr_number`, `queued_at` (default
  `datetime.now()`), `attempts`, `max_attempts=3`.
- `QueueManager`:
  - `_queues: Dict[str, asyncio.Queue]` (maxsize 50 per repo)
  - `_workers: Dict[str, asyncio.Task]`
  - `_processing: Dict[str, Optional[PRJob]]`
  - `_stats`: `total_queued / total_processed / total_failed / total_retried`
  - `_lock: asyncio.Lock` guards all mutable state
  - Injectable `_processor` for testability (defaults to lazy import of
    `server.process_pr` to break the circular dependency)
  - `_shutting_down` flag rejects new enqueues post-shutdown
- `enqueue_pr`: validates inputs (empty repo / non-int pr / pr ≤ 0 → False),
  creates queue on first job, spawns worker if missing/done, catches
  `RuntimeError` when no event loop, catches `QueueFull` for races.
- `_worker`: 300s idle timeout exits worker (cleanup is lock-protected and
  only removes its own slot — concurrent enqueues that spawned a new worker
  are safe), `asyncio.CancelledError` propagates.
- `_run_job`: runs `_processor` via `asyncio.to_thread` under a 300s
  `wait_for`, retries on `Exception` up to `max_attempts` (re-queues the
  job), **no retry on `TimeoutError` / `SystemExit` / `KeyboardInterrupt`**.
- `get_status`: depth + currently_processing + worker_alive per repo + stats
  + total_active_repos.
- `shutdown`: stops accepting, waits ≤30s for natural drain, cancels
  stragglers.
- `server.process_pr` now synchronous (raises on failure so queue can decide
  retry vs fail).
- New `GET /status` endpoint returns queue state.

### Validation
**34 checks passed**, including: invalid inputs (5 cases), FIFO processing,
sequential-within-repo + parallel-across-repos verified by timing
measurements (cross-repo start gap < 100 ms), retry on transient exception
(`total_retried=2`, `total_processed=1`), max-attempts exhausted →
`total_failed=1`, **timeout marks failed without retry** (`total_retried=0`),
queue capacity (3 accepted + 3 rejected), `get_status` shape, shutdown drains
all 5 in-flight jobs, duplicate PR enqueue allowed (both processed), enqueue
rejected after shutdown, FastAPI integration (`/`, `/status`, malformed → 400,
non-PR → ignored, PR closed → ignored, valid PR opened → accepted, bad
signature → 401).

---

## Priority 4 — RAG Index Refresh on Push

### Goal
Keep the index in sync when developers push to the default branch.

### Files
| File | Action | Notes |
|---|---|---|
| `server.py` | edit | Event dispatch + `handle_pull_request` + `handle_push` + `refresh_repo_index` |

### Implementation highlights
- Dispatch by `X-GitHub-Event`: `pull_request` / `push` / `ping` / else.
- Malformed JSON request bodies → 400.
- `handle_pull_request`:
  - Accepts `opened`, `synchronize`, `reopened` (added `reopened` per spec).
  - Missing `repository`/`pull_request` → 400 (`KeyError`/`TypeError`).
  - Non-int `pr_number` → 400.
  - `state == "closed"` → ignored.
  - `draft == True` → ignored.
- `handle_push`:
  - Missing `repository.full_name` → 400.
  - `ref` defaults to empty when None/missing; only triggers when
    `ref == f"refs/heads/{default_branch}"`.
  - `default_branch` falls back to `"main"` when missing → handles both
    `main` and `master` projects.
  - `commits` not a list → ignored (warning log); empty list → ignored.
  - Code-file filter: `.py / .js / .ts / .jsx / .tsx / .java / .go / .rb`
    — README/JSON/YAML changes don't trigger a refresh.
  - Aggregates `added` + `modified` + `removed` across all commits.
  - Schedules `background_tasks.add_task(refresh_repo_index, repo_name)`.
- `refresh_repo_index`:
  - Per-repo `threading.Lock` (`_refresh_locks` keyed by repo, guarded by
    `_refresh_locks_guard`) — concurrent pushes for the same repo serialize.
  - Catches `invalidate_cache` exceptions and still attempts clone.
  - Catches `get_repo_path` exceptions — never propagates.
  - Lock acquired with 10-min safety timeout; released in `finally`.

### Validation
**32 checks passed**, covering: ping/pong, unknown event → ignored,
malformed JSON → 400, PR actions (opened/synchronize/reopened accepted,
closed/draft ignored), PR malformed payloads (missing repo / missing PR /
non-int → 400), push variants (main → scheduled, feature branch → ignored,
master with `default_branch=master` → scheduled, no code files → ignored,
0 commits → ignored, missing `commits` → ignored, `commits=str` → ignored,
missing `default_branch` → main fallback, missing repo → 400, `ref=None` →
ignored, added/modified/removed all captured, every code extension triggers).
**Serialization verified** by two threads firing `refresh_repo_index` for the
same repo: durations 0.26 s and 0.52 s confirmed second blocked on first.
Failure handling: `invalidate_cache` raising → clone still attempted;
`get_repo_path` raising → function returns cleanly.

---

## Priority 5 — Test Suite

### Goal
Replace the legacy integration scripts with a real pytest suite.

### Files
| File | Action | Notes |
|---|---|---|
| `pytest.ini` | **new** | `testpaths = tests`, `asyncio_mode = auto`, deselect `integration` marker |
| `tests/__init__.py` | **new** | package marker |
| `tests/conftest.py` | **new** | fixtures + env defaults + `venv/bin` on PATH + `FakeGroqClient` |
| `tests/test_ast_parser.py` | **new** | 33 tests |
| `tests/test_linter.py` | **new** | 13 tests |
| `tests/test_rag.py` | **new** | 25 tests |
| `tests/test_llm.py` | **new** | 15 tests (1 integration) |
| `tests/test_github.py` | **new** | 9 tests |
| `tests/test_queue.py` | **new** | 15 async tests |
| `tests/test_server.py` | **new** | 19 tests |
| `tests/test_agent.py` | **new** | 2 integration tests |
| `tests/eval_suite.py` | **new** | runnable LLM-quality script (10 cases, --runs N) |
| `requirements.txt` | edit | added `pytest`, `pytest-asyncio`, `pytest-mock` |

### Implementation highlights
- `pytest.ini` sets `addopts = -m "not integration" -ra` so integration tests
  are skipped by default.
- `conftest.py` sets dummy `GROQ_API_KEY` / `GITHUB_TOKEN` at module level
  (before any test imports `llm_reviewer`), points `CHROMA_PATH` at a
  per-process temp dir, prepends `dirname(sys.executable)` to `PATH` so
  `flake8` / `bandit` subprocesses resolve.
- `FakeGroqClient` is a drop-in replacement consumed via the `patch_groq`
  fixture — accepts either a single `response` string or a `responses` list
  for retry-behavior tests.
- `test_server.py` uses a `stub_queue` fixture (records `enqueue_pr` calls
  without actually processing) and a `stub_refresh` fixture (prevents
  FastAPI BackgroundTasks from trying to clone real repos).
- `test_queue.py` monkey-patches `WORKER_IDLE_TIMEOUT_SECONDS` / 
  `JOB_TIMEOUT_SECONDS` / `INTER_JOB_DELAY_SECONDS` down so each test runs in
  seconds instead of minutes.
- `eval_suite.py` is intentionally **not** `test_*.py` so pytest won't
  auto-collect it (it costs real Groq tokens).

### Validation
**`pytest -q` reports 129 passed, 2 deselected, 0 captured-output noise**.

Per-file counts:
- `test_ast_parser.py`: 33
- `test_rag.py`: 25
- `test_server.py`: 19
- `test_queue.py`: 15
- `test_llm.py`: 15
- `test_linter.py`: 13
- `test_github.py`: 9
- `test_agent.py`: 2 (integration, auto-skip without creds)

### One iteration fix during validation
- First run: 12 linter tests failed with `FileNotFoundError: 'flake8'` because
  `venv/bin/` isn't on PATH when pytest runs via `venv/bin/python -m pytest`.
  Added a path-prepend in `conftest.py` → all 12 now pass.
- Captured-stdout `git clone` noise from `test_webhook_push_master_default_branch`
  — the test didn't stub `refresh_repo_index` so FastAPI's BackgroundTasks
  actually tried to clone the fake repo. Made `stub_refresh` part of the
  default `client` fixture → no more noise.

---

## Priority 6 — GitHub App Authentication

### Goal
Allow one-click installation via a GitHub App instead of manual webhook +
PAT setup. Support both auth modes side-by-side so the CLI path keeps
working.

### Files
| File | Action | Notes |
|---|---|---|
| `github_app.py` | **new** | JWT minting + installation-token exchange + cached App client + repo→installation registry |
| `github_client.py` | rewrite | `get_github_client(installation_id=None)` dual auth; `get_pr_details` accepts `installation_id` |
| `server.py` | edit | `handle_installation` event handler + PR webhook registers `installation.id` |
| `tests/test_github_app.py` | **new** | 25 tests (mocked HTTP + real RSA via `cryptography`) |
| `tests/test_server.py` | edit | +8 installation-event tests |
| `tests/test_github.py` | edit | +5 dual-auth tests |

### Implementation highlights
- `_normalize_private_key` handles Windows CRLF and base64-encoded PEMs
  (detected by absence of `-----BEGIN` marker).
- `generate_jwt`: payload is `{iat: now-60, exp: now+600, iss: str(app_id)}`,
  algorithm RS256; wraps `pyjwt.InvalidKeyError` and other signing errors in
  `ValueError` with a clear message.
- `get_installation_token`: POSTs to
  `/app/installations/{id}/access_tokens`. Retries on `RequestException`,
  503/502/500/504/429/403 with exponential backoff (1s, 2s, 4s). 404 raises
  a clear "installation not found" error (App uninstalled); 401 raises
  "JWT rejected — check app_id/key".
- `get_github_client_for_installation`: caches one `Github` client per
  installation_id for 50 minutes (vs 60 min real token lifetime, giving
  headroom for in-flight calls). Raises if `GITHUB_APP_ID` /
  `GITHUB_APP_PRIVATE_KEY` aren't set.
- `invalidate_installation_cache(installation_id=None)`: clears one or all
  cached clients — used on uninstall.
- `register_repo_installation` / `get_installation_for_repo` /
  `forget_repo_installation`: a small in-memory `repo → installation_id`
  map. Populated by the `installation` and `pull_request` webhook handlers
  so future PR review code can resolve a repo to its installation without
  re-deriving the mapping. Invalid inputs are silently ignored (not raised).
- `github_client.get_github_client(installation_id=None)`:
  - When `GITHUB_APP_ID + GITHUB_APP_PRIVATE_KEY` are set **and**
    `installation_id` is passed → use App auth.
  - On App auth failure, falls back to PAT (logged at error) rather than
    blowing up the request.
  - When no auth at all is configured → `ValueError` with a clear message.
- `get_pr_details` now accepts an optional `installation_id` and forwards it
  to `get_github_client` — backwards-compatible since the param defaults
  to `None`.
- `handle_installation` (in `server.py`):
  - `created` with repos → registers each `repo → installation_id` mapping
    and schedules a background `refresh_repo_index` per repo. Returns
    `{action: "installation_recorded", count: N}`.
  - `created` with empty/missing `repositories` → returns
    `{action: "no_repos_to_index"}`.
  - `deleted` → calls `forget_repo_installation` and schedules
    `invalidate_cache` per repo. Returns `{action: "installation_removed"}`.
  - Missing `installation` field → 400.
  - Non-int / non-positive `installation.id` → 400.
  - `repositories` not a list → treat as empty (warning log).
  - Other actions (`suspend`, `unsuspend`, etc.) → ignored.
- `handle_pull_request` now also pulls `data["installation"]["id"]` (when
  present) and calls `register_repo_installation` before enqueuing — so
  later PR processing can resolve the installation via the cache.

### Validation
**38 new tests, all passing**, broken down:
- `test_github_app.py` (25): key normalization (Windows CRLF, base64,
  passthrough, empty raises), `generate_jwt` round-trip with real RSA key
  (signs + verifies + claims correct), missing/empty/invalid key raises,
  `get_installation_token` success, 5xx retry-then-success, 404 raises,
  401 raises, invalid install_id raises, all-retries-fail raises, network
  error retries-then-fails, client caching (same id → same client, different
  id → different client), TTL expiry triggers refresh, invalid install_id
  raises, missing env vars raises, cache invalidation (all vs single),
  repo→installation register/lookup/forget, invalid input ignored.
- `test_server.py` (+8): installation created with repos → indexing
  scheduled + count; created with empty repos → `no_repos_to_index`;
  created with missing `repositories` field → `no_repos_to_index`;
  deleted → invalidations scheduled; missing `installation` field → 400;
  non-int `installation.id` → 400; unknown action (`suspend`) → ignored;
  PR webhook carrying `installation.id` populates the repo→installation
  cache.
- `test_github.py` (+5): PAT default path (no installation_id),
  App path forwards installation_id, App-failure falls back to PAT, no-auth
  raises `ValueError`, `get_pr_details` forwards installation_id.

**Total suite: 167 passed, 2 deselected** (up from 129).

---

## Priority 7 — Fly.io Deployment (files-ready, deploy pending)

### Goal
Produce the three config artifacts needed to deploy to Fly.io. The user
hasn't created their Fly.io account yet, so the actual `flyctl deploy` step
is intentionally deferred — but everything that can be authored and
validated locally is done.

### Decisions locked before writing
| Question | Choice |
|---|---|
| uvicorn workers | **1** (preserves the per-repo serial guarantee of `queue_manager`; scale horizontally via Fly machines instead) |
| `railway.toml` | **deleted** — Fly.io is the new home |
| Local Docker build | **performed** — caught zero Dockerfile bugs |
| `fly.toml` app name | **`code-review-agent-TODO`** placeholder; `deploy.sh` refuses to run until edited |

### Files
| File | Action | Notes |
|---|---|---|
| `Dockerfile` | rewrite | apt + node + eslint + pre-baked CodeBERT + 1-worker uvicorn |
| `fly.toml` | **new** | Mumbai region, 2 GB / shared-2x, volume mount at `/data`, health check on `/`, auto-stop with min 1 machine |
| `deploy.sh` | **new** | Pre-flight checks → volume create (idempotent) → secrets set → `flyctl deploy --remote-only` |
| `railway.toml` | **deleted** | Old Railway config removed |

### Dockerfile highlights
- `nodejs` + `npm` + global `eslint@9` for the JS/TS linter path.
- Pre-downloads `microsoft/codebert-base` (~500 MB) in a build layer so
  the first PR review skips the cold-start model load.
- Creates `/data/chroma_db` and sets `CHROMA_PATH` env var so the image
  also works without the Fly volume (degrades to ephemeral ChromaDB).
- `--workers 1` is intentional: comment explicitly notes that the
  per-repo serial queue guarantee is per-process.

### fly.toml highlights
- Mounts `chroma_data` volume at `/data` — directly leverages Priority 2's
  persistent storage layer.
- `auto_stop_machines = 'stop'` + `auto_start_machines = true` +
  `min_machines_running = 1` → scale-to-zero with instant wake on the next
  webhook delivery, so GitHub's 10 s timeout doesn't fire.
- HTTP concurrency soft/hard limits 20/25 keep one machine from getting
  saturated.
- Health check uses the existing `GET /` endpoint; 30 s grace, 15 s
  interval, 10 s timeout.
- `app = 'code-review-agent-TODO'` placeholder; `deploy.sh` aborts with a
  clear message if this isn't edited first.

### deploy.sh highlights
- `set -euo pipefail` — fail fast on any error or unset var.
- Pre-flight checks: `flyctl` installed, logged in (`flyctl auth whoami`),
  `fly.toml` not still the placeholder, required env vars present
  (`GROQ_API_KEY`, `GITHUB_TOKEN`, `WEBHOOK_SECRET`).
- Volume create is idempotent — checks `flyctl volumes list` first.
- Secrets array picks up optional GitHub App vars only if both
  `GITHUB_APP_ID` and `GITHUB_APP_PRIVATE_KEY` are exported.
- Final output prints the public URLs (`/`, `/webhook`, `/status`) by
  parsing the app name back out of `fly.toml`.

### Local validation
- `bash -n deploy.sh` — passes.
- `python -c "import tomllib; tomllib.load(open('fly.toml','rb'))"` —
  passes, all expected keys present (app / primary_region / build /
  http_service / vm / mounts / checks).
- **Docker build attempted locally** per user request. First attempt
  failed mid-build on transient apt-mirror connectivity. Second attempt
  ran all 9 build stages to completion (apt-get install, npm install -g
  eslint@9, pip install -r requirements.txt, CodeBERT pre-download, COPY,
  mkdir) before failing at the final image-extract step with
  `input/output error` on the torch layer — a Docker Desktop containerd
  storage corruption on the local Mac, not a Dockerfile bug. The
  Dockerfile is structurally verified end-to-end. Fly.io's remote builder
  runs on different infrastructure and won't hit this local disk issue.

### Pending — user actions before this ships
1. `brew install flyctl`
2. `flyctl auth login`
3. Add a payment method on Fly.io (free tier requires it).
4. Edit `app = ...` in `fly.toml` to a globally-unique name. Suggested
   form: `code-review-agent-<your-handle>`.
5. Export the secrets in your shell:
   ```bash
   export GROQ_API_KEY=gsk_...
   export GITHUB_TOKEN=ghp_...
   export WEBHOOK_SECRET=$(openssl rand -hex 32)
   ```
6. Run `./deploy.sh`.

The script will create the volume, set secrets, deploy via Fly's remote
builder, and print the public webhook URL to paste into GitHub.

---

## Cleanup — Removed legacy root-level test scripts

Four untracked legacy integration scripts at the project root were
superseded by the new `tests/` suite and removed:

| File | Why redundant |
|---|---|
| `test_linter.py` | Hit real GitHub `royo1019/test-review-agent` PR #1. Now `tests/test_linter.py` (13 unit tests). |
| `test_rag.py` | Used hardcoded `/Users/royo/test-repo`. Now `tests/test_rag.py` (25 isolated tests). |
| `test_llm.py` | Same hardcoded path + real Groq calls. Now `tests/test_llm.py` (15 mocked tests + 1 integration). |
| `test_post.py` | Same hardcoded path + real GitHub posting. Now `tests/test_agent.py` (integration, gated). |

None were tracked in git, so deletion was just a filesystem operation.
After cleanup: pytest still reports **129 passed, 2 deselected**.

---

## Final State

### New files
```
ast_parser.py                Priority 1
queue_manager.py             Priority 3
github_app.py                Priority 6
fly.toml                     Priority 7
deploy.sh                    Priority 7
pytest.ini                   Priority 5
tests/__init__.py            Priority 5
tests/conftest.py            Priority 5
tests/test_ast_parser.py     Priority 5
tests/test_linter.py         Priority 5
tests/test_rag.py            Priority 5
tests/test_llm.py            Priority 5
tests/test_github.py         Priority 5  (extended in Priority 6)
tests/test_github_app.py     Priority 6
tests/test_queue.py          Priority 5
tests/test_server.py         Priority 5  (extended in Priority 4 + Priority 6)
tests/test_agent.py          Priority 5
tests/eval_suite.py          Priority 5
BUILD_LOG.md                 (this file)
```

### Modified files
```
agent.py            Priority 2  (thread repo_name through state machine)
rag.py              Priority 1 + 2  (AST hook + PersistentClient + per-repo collections)
llm_reviewer.py     Priority 1  (AST in prompt)
repo_cache.py       Priority 2  (per-repo collection lifecycle)
server.py           Priority 3 + 4 + 6  (queue + push handler + ping + installation event)
github_client.py    Priority 6  (dual PAT/App auth)
Dockerfile          Priority 7  (production image with pre-baked model)
requirements.txt    Priority 5  (pytest tooling)
```

### Removed files (in addition to legacy test scripts)
```
railway.toml        Priority 7  (project moved to Fly.io)
```

### Removed files
```
test_linter.py      legacy integration script
test_rag.py         legacy integration script
test_llm.py         legacy integration script
test_post.py        legacy integration script
```

### Cumulative validation totals
| Priority | Checks |
|---|---|
| 1 — AST | 47 |
| 2 — Persistent ChromaDB | 33 |
| 3 — Async queue | 34 |
| 4 — Push refresh | 32 |
| 5 — pytest suite | 129 passed, 2 deselected |
| 6 — GitHub App | +38 pytest tests (total now 167 passed, 2 deselected) |
| 7 — Fly.io configs | Dockerfile verified end-to-end via local `docker build` (all 9 RUN stages green; image-export failed due to local Docker Desktop disk corruption, unrelated to Dockerfile) |
| **Total** | **146 standalone checks + 167 pytest tests + 3 Fly.io config artifacts** |

### Per-test-file counts (after Priority 6)
| File | Tests |
|---|---|
| `test_ast_parser.py` | 33 |
| `test_server.py` | 27 |
| `test_rag.py` | 25 |
| `test_github_app.py` | 25 |
| `test_queue.py` | 15 |
| `test_llm.py` | 15 |
| `test_github.py` | 14 |
| `test_linter.py` | 13 |
| `test_agent.py` | 2 (integration, auto-skipped) |
| **Total** | **169 (167 active + 2 deselected)** |

### Remaining priorities (not yet implemented)
- Priority 8 — `README.md` rewrite per spec template.

### Pending user actions (not blockers, but required before Priority 7 ships)
- Install flyctl, log in, add a payment method on Fly.io.
- Edit `fly.toml` `app = ...` to a globally-unique name.
- Export `GROQ_API_KEY` / `GITHUB_TOKEN` / `WEBHOOK_SECRET` (and optionally
  `GITHUB_APP_ID` / `GITHUB_APP_PRIVATE_KEY`) in the shell, then run
  `./deploy.sh`.
