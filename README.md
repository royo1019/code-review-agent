# AI Code Review Agent

An autonomous pull-request reviewer. It listens for GitHub webhooks, clones and
indexes the target repository, and runs each changed file through a LangGraph
state machine that combines **static analysis**, **semantic retrieval (RAG)**,
and **AST-based structural analysis** into a single LLM prompt. Findings are
posted back as inline review comments plus a summary comment with an
APPROVE / REQUEST_CHANGES verdict.

The core idea: a generic LLM review says "consider adding error handling."
A review that also knows the repo's call graph, import graph, and semantically
similar existing code can say *"`validate_user()` already exists in
`auth/utils.py:41` — call it instead of reimplementing this."*

---

## Table of contents

- [Architecture](#architecture)
- [Request lifecycle](#request-lifecycle)
- [Module reference](#module-reference)
  - [`server.py` — webhook receiver](#serverpy--webhook-receiver)
  - [`queue_manager.py` — per-repo async queue](#queue_managerpy--per-repo-async-queue)
  - [`agent.py` — LangGraph state machine](#agentpy--langgraph-state-machine)
  - [`rag.py` — vector index](#ragpy--vector-index)
  - [`ast_parser.py` — structural analysis](#ast_parserpy--structural-analysis)
  - [`linter.py` — static analysis](#linterpy--static-analysis)
  - [`llm_reviewer.py` — prompt construction](#llm_reviewerpy--prompt-construction)
  - [`github_client.py` / `github_app.py` — GitHub I/O and auth](#github_clientpy--github_apppy--github-io-and-auth)
  - [`repo_cache.py` — clone and index lifecycle](#repo_cachepy--clone-and-index-lifecycle)
- [Data model](#data-model)
- [Configuration](#configuration)
- [Running locally](#running-locally)
- [Deployment](#deployment)
- [Testing](#testing)
- [Design decisions and trade-offs](#design-decisions-and-trade-offs)
- [Known limitations](#known-limitations)

---

## Architecture

```
                    GitHub
                      │
        pull_request / push / installation / ping
                      │  (HMAC-SHA256 signed)
                      ▼
            ┌───────────────────────┐
            │   server.py (FastAPI) │
            │   POST /webhook       │
            │   GET  /  /status     │
            └───────────┬───────────┘
                        │
         ┌──────────────┴──────────────┐
         │                             │
  pull_request                       push / installation
         │                             │
         ▼                             ▼
┌─────────────────────┐    ┌──────────────────────────┐
│  queue_manager.py   │    │  refresh_repo_index()    │
│  one asyncio.Queue  │    │  BackgroundTasks +       │
│  + worker per repo  │    │  per-repo threading.Lock │
└──────────┬──────────┘    └────────────┬─────────────┘
           │                            │
           │  asyncio.to_thread         │ invalidate + re-clone + re-index
           ▼                            ▼
┌─────────────────────────────────────────────────────┐
│              repo_cache.py                          │
│   clone → index_codebase() → cache repo_name→path   │
└──────────┬──────────────────────────┬───────────────┘
           │                          │
           ▼                          ▼
   ┌───────────────┐          ┌────────────────┐
   │   rag.py      │          │ ast_parser.py  │
   │  ChromaDB     │          │ symbol table,  │
   │  per-repo     │          │ call graph,    │
   │  collection   │          │ import graph   │
   │  CodeBERT     │          └────────────────┘
   └───────────────┘

                        ▼
            ┌───────────────────────┐
            │   agent.py            │
            │   LangGraph StateGraph│
            │   (per changed file)  │
            └───────────┬───────────┘
                        ▼
   linter.py ─┐
   rag.py ────┼──▶ llm_reviewer.py ──▶ Groq (llama-3.3-70b-versatile)
   ast_parser ┘                              │
                                             ▼
                                  github_client.py
                            inline comments + summary + verdict
```

**Stack:** FastAPI · LangGraph · ChromaDB · sentence-transformers (CodeBERT) ·
Python `ast` · Flake8 / Bandit / ESLint · Groq (LLaMA 3.3 70B) · PyGithub ·
GitPython · Docker / Railway.

---

## Request lifecycle

A `pull_request` webhook (`opened` / `synchronize` / `reopened`) flows through
the system as follows:

1. **Signature validation** — `server.validate_signature` computes
   `sha256=HMAC(WEBHOOK_SECRET, raw_body)` and compares against
   `X-Hub-Signature-256` using `hmac.compare_digest`. With no secret
   configured, validation passes (dev mode).
2. **Filtering** — closed PRs and drafts are dropped. Non-handled actions
   return `{"status": "ignored"}` with a reason.
3. **Installation capture** — if the delivery carries `installation.id`
   (GitHub App mode), `register_repo_installation` records the
   `repo → installation_id` mapping.
4. **Enqueue** — `queue_manager.enqueue_pr(repo, pr)` places a `PRJob` on the
   repo's queue and lazily spawns that repo's worker task. Returns HTTP 200
   immediately; GitHub's 10-second webhook budget is never at risk.
5. **Worker pickup** — the per-repo worker dequeues the job and dispatches
   `process_pr` via `asyncio.to_thread` under a 300-second timeout.
6. **Repo materialization** — `repo_cache.get_repo_path` returns the cached
   clone, or clones into a temp dir and calls `index_codebase` (builds both
   the ChromaDB collection and the AST index).
7. **Graph execution** — `agent.run_agent` compiles the LangGraph once, then
   invokes it once per changed file.
8. **Posting** — inline comments land on the PR head commit; the summary
   comment carries a severity table and the verdict.

Two other event types are handled:

- **`push`** to the default branch touching a code file schedules
  `refresh_repo_index` as a FastAPI `BackgroundTask`, guarded by a per-repo
  `threading.Lock` (10-minute acquire timeout) so concurrent pushes serialize.
- **`installation`** (`created`) registers each repo and schedules a background
  index build; (`deleted`) forgets the mapping and invalidates caches.

---

## Module reference

### `server.py` — webhook receiver

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Liveness probe — used as Railway's healthcheck path |
| `/status` | GET | Live queue depths, in-flight job per repo, worker liveness, aggregate stats |
| `/webhook` | POST | GitHub event ingress |

Event routing is a flat dispatch on `X-GitHub-Event`: `pull_request`, `push`,
`installation`, `ping` (returns pong so GitHub's redelivery UI shows green),
everything else ignored.

Payload handling is defensive throughout — malformed JSON, missing keys,
non-integer PR numbers, and `commits` arriving as a non-list all produce a
400 or a logged `ignored` rather than a 500.

Index refreshes only fire when the changed-file set contains at least one of
`.py .js .ts .jsx .tsx .java .go .rb`; README/JSON/YAML pushes don't invalidate
the vector index.

### `queue_manager.py` — per-repo async queue

The concurrency model is one `asyncio.Queue` and one worker coroutine **per
repository**:

- **Parallel across repos** — repo A and repo B review simultaneously.
- **Serial within a repo** — PR #1 finishes before PR #2 starts, which keeps
  two workers from mutating the same ChromaDB collection concurrently.

Tunables (module-level so tests can patch them down):

| Constant | Default | Meaning |
|---|---|---|
| `QUEUE_MAX_SIZE` | 50 | Per-repo backlog; enqueue returns `False` when full |
| `WORKER_IDLE_TIMEOUT_SECONDS` | 300 | Idle worker exits and is respawned on demand |
| `JOB_TIMEOUT_SECONDS` | 300 | Per-job wall clock; timeout is terminal, no retry |
| `INTER_JOB_DELAY_SECONDS` | 1 | Breather between jobs |

`_run_job` returns one of `"ok" | "retry" | "failed"`. Exceptions re-queue the
job up to `PRJob.max_attempts` (3); timeouts do not retry, on the theory that a
job that blew the wall clock once will blow it again. `asyncio.CancelledError`,
`SystemExit`, and `KeyboardInterrupt` propagate rather than being swallowed as
job failures.

Worker teardown checks `self._workers.get(repo) is asyncio.current_task()`
before cleaning its slot, so a worker exiting on idle timeout can't delete a
replacement worker that a concurrent `enqueue_pr` just spawned.

`shutdown()` stops accepting jobs, waits up to 30 seconds for natural drain,
then cancels stragglers. It's wired to FastAPI's shutdown event.

The default processor imports `server.process_pr` **lazily inside the
function** — `server.py` imports the `queue_manager` singleton, so a top-level
import would be circular.

### `agent.py` — LangGraph state machine

`AgentState` is a `TypedDict` carrying the repo/PR handles, the file list, the
current file, per-node outputs (`lint_findings`, `rag_chunks`, `comments`), the
accumulator `all_comments`, `retry_count`, and the final `verdict`.

```
fetch_pr → run_linters → fetch_rag → call_llm ─┬─(no comments, <3 tries)→ retry ─┐
                                               │                                 │
                                               └─────────────→ post_comments ◀───┘
                                                                     │
                                                                  verdict → END
```

| Node | Responsibility |
|---|---|
| `fetch_pr_node` | Resolve repo + PR objects, list changed files |
| `run_linters_node` | Fetch file content at PR head SHA, run the language-appropriate linters |
| `fetch_rag_node` | Query the repo's ChromaDB collection with the diff text |
| `call_llm_node` | Build the prompt (diff + lint + RAG + AST) and call Groq |
| `retry_node` | Increment `retry_count` and loop back to `call_llm` |
| `post_comments_node` | Post inline comments, append to `all_comments` |
| `verdict_node` | Post the summary and return APPROVE / REQUEST_CHANGES |

The conditional edge `should_retry` re-runs the LLM when it returned zero
comments and fewer than 3 attempts have been made. This is a cheap guard
against a transient empty/malformed response — distinct from the JSON-parse
retry inside `llm_reviewer.call_llm`, which handles the same failure one layer
down.

The graph is compiled **once** and invoked **once per changed file**, with
`all_comments` carried in the initial state. Per-file invocation keeps each
prompt scoped to a single diff (better signal, smaller context) at the cost of
one LLM call per file.

### `rag.py` — vector index

Wraps ChromaDB with a **per-repo collection** model. Also owns the global AST
index, which is built inside `index_codebase` so both indexes stay in lockstep.

**Embeddings** — `microsoft/codebert-base` via `sentence-transformers`,
overridable with `EMBEDDING_MODEL`. A code-pretrained encoder is used rather
than a general-purpose text model because the corpus and the queries are both
source code.

**Chunking** — `chunk_file` produces 50-line windows with 10 lines of overlap.
The overlap keeps a function that straddles a boundary retrievable from either
side. Indexed extensions: `.py .js .ts .java .md`.

**Collection naming** — `_sanitize_repo_name` maps `owner/repo` to a
ChromaDB-legal name: `/`, `-`, `.` → `_`, all other non-alphanumerics dropped,
leading digit prefixed with `r_`, truncated to 63 chars, padded to a minimum of
3. Because that mapping is lossy (`owner/foo-bar` and `owner/foo_bar` collide),
`_collection_name_for` maintains a registry and appends an 8-char SHA-1 suffix
of the original name when a different repo already claimed a base name.

**Persistence** — `PersistentClient(CHROMA_PATH)`, with a graceful fallback to
the in-memory `Client` on `PermissionError` / `OSError` so a read-only or full
disk degrades to "no cross-restart persistence" instead of a crash loop.

**Idempotence** — `index_codebase` skips the vector build when
`collection.count() > 0`. Forcing a rebuild means calling `delete_collection`
first, which is exactly what `repo_cache.invalidate_cache` does.

**Failure posture** — `retrieve_context` returns `[]` (never raises) for empty
queries, missing collections, empty collections, embedding failures, and query
failures. `n_results` is clamped to the collection size. RAG is an enhancement,
so its failure must not take down a review.

### `ast_parser.py` — structural analysis

Pure-stdlib `ast` walk over the repo, producing three indexes inside an
`ASTIndex` dataclass:

1. **Symbol table** — `functions: name → [FunctionInfo]` (a list, because names
   repeat across files) and `classes: name → ClassInfo`.
2. **Call graph** — `call_graph: "filepath::function" → [callee names]`, plus
   `reverse_call_graph: callee → [caller keys]` for "who calls this".
3. **Import graph** — `imports: filepath → [ImportInfo]`.

`FunctionInfo` captures name, filepath, lineno, args (including defaults,
`*args`/`**kwargs`, and annotations), return annotation, calls, decorators,
first docstring line, and method/class attribution via `_build_method_map`.

Robustness rules: `SKIP_DIRS` excludes `.git`, `venv`, `node_modules`,
`__pycache__`, build dirs, etc.; files over `MAX_FILE_SIZE_BYTES` (1 MB) are
skipped as likely generated; syntax errors, vanished files, and stat failures
are logged and skipped. `build_index` never raises on a bad repo.

**Diff → prompt context.** `format_ast_context_for_llm` extracts `name(`
patterns from the diff, resolves each against the index, and emits a text block
listing, per function: definition site and line, arguments, callees, callers,
and docstring — followed by an import-relationship section. Unresolved names
are labeled either `Python builtin` (checked against a `PYTHON_BUILTINS` set)
or `not found in codebase`, which is what lets the LLM flag typos and missing
imports as likely bugs. Output is truncated to roughly 2000 characters to bound
prompt growth.

This module is Python-only; other languages fall through to lint + RAG + LLM.

### `linter.py` — static analysis

Routes on file extension after writing the fetched content to a temp file:

| Extension | Tools |
|---|---|
| `.py` | Flake8 (style/correctness) + Bandit (security, JSON output) |
| `.js` `.ts` `.jsx` `.tsx` | ESLint, if installed — flat rule set passed via `--rule` with no project config, so it works on any repo |
| `.java` `.go` `.rb` | LLM-only |
| `.md` `.txt` `.json` `.yaml` | LLM-only |

Findings are normalized to `{tool, line, severity, message}` regardless of
source. `is_tool_installed` gates ESLint so a missing Node toolchain logs a
hint instead of failing the review.

### `llm_reviewer.py` — prompt construction

`build_prompt` assembles four labeled sections — the diff, static-analysis
findings, RAG chunks (truncated to 300 chars each), and the AST block — and
specifies the output contract: a JSON array of
`{line, severity, comment}` where severity is `critical | warning | suggestion`,
with explicit definitions for each tier.

The instruction set is what turns context into leverage:

- If RAG shows a better pattern already in the codebase, cite it.
- If AST shows a function already exists that does what the new code does,
  recommend it by filename and line.
- If the new code calls a function that doesn't exist in the index, flag it as
  a likely typo / missing import / wrong name.
- If AST shows duplicated logic, recommend DRY refactoring and name the
  existing function.

`call_llm` targets `llama-3.3-70b-versatile` at `temperature=0.1` (review
output should be near-deterministic), strips ``` fences, and retries up to 3
times on `JSONDecodeError` or any API exception. It returns `[]` after
exhausting retries rather than raising — "no comments" is a valid outcome the
graph knows how to handle.

`review_pr` pulls AST context from the global index and tolerates its absence.

### `github_client.py` / `github_app.py` — GitHub I/O and auth

Two authentication modes, resolved at call time:

- **Personal Access Token** — `GITHUB_TOKEN`. The default; keeps the CLI
  entrypoint working with zero App setup.
- **GitHub App** — `GITHUB_APP_ID` + `GITHUB_APP_PRIVATE_KEY`, used when an
  `installation_id` is supplied. Falls back to PAT if App auth throws.

`github_app.py` handles the full App flow:

- `_normalize_private_key` copes with two real deployment quirks: CRLF line
  endings, and base64-wrapped PEMs (some secret stores require them) — detected
  by the absence of a `-----BEGIN` marker and accepted only if the decoded
  bytes contain one.
- `generate_jwt` mints an RS256 JWT with `iat` backdated 60 s for clock skew
  and `exp` at GitHub's 10-minute ceiling.
- `get_installation_token` exchanges the JWT with retry and exponential backoff
  (1 s / 2 s / 4 s) on network errors and 403/429/5xx, and raises distinct,
  actionable errors on 401 (bad App credentials) and 404 (app uninstalled).
- `get_github_client_for_installation` caches the client for **50 minutes**
  against a 60-minute token lifetime, leaving headroom for in-flight calls.
- `_repo_to_installation` maps repos to installations, populated by the
  webhook handlers.

`github_client.py` also owns the output side: `post_inline_comments` attaches
each comment to the PR's latest commit and counts successes (a single bad line
number can't abort the batch), and `post_summary_and_verdict` renders the
severity table, lists critical and warning items, posts it, and returns
`REQUEST_CHANGES` if any critical finding exists, else `APPROVE`.

### `repo_cache.py` — clone and index lifecycle

In-memory `repo_name → local_path` map. First review clones over HTTPS with the
token embedded in the URL into a `tempfile.mkdtemp()` directory and triggers
`index_codebase`; later reviews reuse it. `invalidate_cache` removes the clone
tree and calls `rag.delete_collection`, tolerating a repo that was never
cached.

---

## Data model

```python
# agent.py
class AgentState(TypedDict):
    repo_name: str;  pr_number: int
    repo: Any;       pr: Any
    files: List[Dict];  current_file: Dict
    lint_findings: List[Dict]   # {tool, line, severity, message}
    rag_chunks: List[Dict]      # {text, filename, start_line}
    comments: List[Dict]        # {line, severity, comment}
    all_comments: List[Dict]
    retry_count: int;  verdict: str

# queue_manager.py
@dataclass
class PRJob:
    repo_name: str;  pr_number: int
    queued_at: datetime;  attempts: int = 0;  max_attempts: int = 3

# ast_parser.py
@dataclass
class ASTIndex:
    functions: Dict[str, List[FunctionInfo]]
    classes: Dict[str, ClassInfo]
    imports: Dict[str, List[ImportInfo]]
    call_graph: Dict[str, List[str]]          # "path::func" → callees
    reverse_call_graph: Dict[str, List[str]]  # callee → ["path::func", ...]
    file_to_functions: Dict[str, List[str]]
```

---

## Configuration

All configuration is environment variables, loaded from `.env` via
`python-dotenv`.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | yes | — | Groq inference |
| `GITHUB_TOKEN` | yes (PAT mode) | — | Repo access, cloning, comment posting |
| `WEBHOOK_SECRET` | recommended | unset | HMAC validation; unset disables it (dev only) |
| `GITHUB_APP_ID` | App mode | — | GitHub App identity |
| `GITHUB_APP_PRIVATE_KEY` | App mode | — | RSA PEM (raw or base64) for JWT signing |
| `CHROMA_PATH` | no | `./chroma_db` | ChromaDB persistence directory |
| `EMBEDDING_MODEL` | no | `microsoft/codebert-base` | sentence-transformers model id |
| `PORT` | no | `8000` | Bind port (set by Railway) |

---

## Running locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Optional — enables JS/TS linting
npm install -g eslint

# Webhook server
python -m uvicorn server:app --reload
```

Expose the local server to GitHub (e.g. `ngrok http 8000`) and point a webhook
at `https://<host>/webhook` with content type `application/json`, subscribing to
**Pull requests** and **Pushes**.

There is also a CLI entrypoint that skips webhooks entirely — useful for
iterating on prompts against a known PR:

```bash
python main.py --repo owner/repo --pr 42 --path /local/clone/of/repo
```

---

## Deployment

Dockerized for Railway (`Dockerfile` + `railway.toml`):

- **Base** `python:3.11-slim`, plus `git` (cloning target repos) and
  `nodejs`/`npm` with ESLint 9 installed globally.
- **Pre-baked model** — CodeBERT (~500 MB) is downloaded at build time so the
  first webhook after a deploy doesn't pay a ~30-second model-load penalty.
- **Single worker, single replica** — `--workers 1` and `numReplicas = 1` are
  load-bearing, not conservatism. The per-repo serial guarantee in
  `queue_manager.py` is *in-process*; a second worker or replica would race two
  reviews onto the same ChromaDB collection. Scaling out requires a shared
  queue (e.g. Redis) first.
- **Healthcheck** — `GET /` with a 30-second timeout, `ON_FAILURE` restart
  policy capped at 3 retries.
- **Ephemeral index** — `CHROMA_PATH=/app/chroma_db` inside the container
  filesystem, so the index is wiped on every restart or deploy and the first PR
  per repo re-indexes. Making it durable requires only mounting a Railway
  volume at `/app/chroma_db` — no code change.

---

## Testing

167 tests across 10 modules, plus a separate LLM evaluation suite.

```bash
pytest                    # unit + integration-mocked suite
pytest tests/test_rag.py  # one module
pytest -m integration     # live Groq/GitHub tests (deselected by default)
python tests/eval_suite.py --runs 5
```

`pytest.ini` sets `asyncio_mode = auto` (queue tests are coroutine-heavy) and
`addopts = -m "not integration"` so a bare `pytest` never spends API tokens or
hits the network.

| Module | Coverage |
|---|---|
| `test_server.py` | Signature validation, event routing, payload malformation, install lifecycle |
| `test_queue.py` | Per-repo isolation, backpressure, retries, timeouts, graceful shutdown |
| `test_ast_parser.py` | Symbol/call/import extraction, method attribution, malformed sources |
| `test_rag.py` | Collection naming, collisions, chunking, retrieval failure paths |
| `test_github_app.py` | JWT minting, key normalization, token exchange, retry/backoff, caching |
| `test_github.py` | PR fetch, diff parsing, comment posting, verdict logic |
| `test_llm.py` | Prompt assembly, JSON parsing, retry behavior |
| `test_linter.py` | Extension routing, finding normalization, missing-tool handling |
| `test_agent.py` | Graph wiring, node transitions, retry edge |

`tests/eval_suite.py` is deliberately named `eval_*` so pytest won't collect
it — it hits the real Groq endpoint. It runs each case N times (default 3) to
smooth LLM non-determinism and reports overall pass rate, critical-issue
detection rate, and **false-positive rate on clean code**, which is the metric
that decides whether a reviewer is actually usable.

`pyproject.toml` configures Ruff (py311, line length 100) with pycodestyle,
pyflakes, isort, bugbear, pyupgrade, bandit, and simplify rules enabled.

---

## Design decisions and trade-offs

**Per-repo queue rather than a global one.** A global queue would serialize
everything; unbounded concurrency would let two reviews of the same repo mutate
one ChromaDB collection simultaneously. Per-repo queues give parallelism exactly
where it's safe. The cost: the guarantee is process-local, which caps horizontal
scaling until the queue moves to Redis.

**LangGraph rather than a plain function chain.** The review flow has a genuine
cycle (`call_llm → retry → call_llm`) and a conditional branch. Expressing that
as a compiled state graph makes the control flow declarative and the state
transitions inspectable, and leaves room for future branches (severity-gated
paths, multi-model comparison) without rewriting the orchestration. For a purely
linear pipeline it would be overkill.

**Per-file graph invocation rather than one whole-PR prompt.** Keeps each prompt
focused on one diff, which measurably improves comment specificity and keeps
context well inside limits on large PRs. The trade-off is one LLM call per
changed file and no cross-file reasoning within a single call — the AST context
partially compensates by surfacing cross-file relationships.

**AST alongside RAG, not instead of it.** They fail in opposite directions.
Embedding similarity finds code that *looks* like the diff but can't tell you
whether a called function actually exists. The AST call graph answers existence
and reachability exactly but can't find a semantically similar implementation
under a different name. Feeding both makes the "this already exists, use it"
and "this function doesn't exist" comments possible.

**Per-repo ChromaDB collections rather than one collection with a metadata
filter.** Isolation is structural rather than query-dependent — a forgotten
`where` clause can't leak repo A's code into repo B's review — and invalidation
is a single `delete_collection` instead of a filtered delete. The cost is
collection-name sanitization and the collision registry that goes with it.

**CodeBERT rather than a general-purpose sentence encoder.** Both the corpus and
the query are source code; a code-pretrained encoder produces meaningfully
better neighbors on identifier-dense text. It's larger and slower, which is why
it's baked into the Docker image.

**Degrade rather than fail.** Every enrichment layer has a defined empty
outcome: RAG returns `[]`, AST returns `""`, linters return `[]`, the LLM
returns `[]` after retries, ChromaDB falls back to in-memory. A review with less
context is still a review; a crashed webhook handler is a dropped PR.

**Retry at two layers.** `call_llm` retries malformed JSON and API errors
in-process; the graph's `should_retry` edge retries a syntactically valid but
empty response. They catch different failures, and the graph-level retry is
visible in the state machine rather than buried in a helper.

---

## Known limitations

- **AST analysis is Python-only.** Other languages get lint + RAG + LLM.
- **Single-process concurrency guarantee.** Horizontal scaling needs a shared
  queue before `numReplicas` can exceed 1.
- **Ephemeral vector index in the default deploy.** Fixed by mounting a volume
  at `/app/chroma_db`.
- **Full re-index on invalidation.** A push to the default branch rebuilds the
  whole collection rather than only the changed files.
- **Clones live in temp directories** keyed by an in-memory map, so a restart
  orphans them until the OS reclaims the temp dir.
- **Inline comment placement uses the LLM-reported diff line number**, which
  GitHub occasionally rejects; those failures are logged per comment and the
  content still appears in the summary.
