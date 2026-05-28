"""RAG layer for the Code Review Agent.

Wraps ChromaDB with a per-repo collection model so multiple repos can be
indexed simultaneously without cross-contamination. Storage is persistent via
``PersistentClient``; the path is configurable through the ``CHROMA_PATH``
environment variable.

Also owns the global AST index (built in tandem with the vector index inside
``index_codebase``) and exposes ``get_ast_context`` for the LLM reviewer.
"""

import hashlib
import logging
import os
import re
from typing import Any, Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from ast_parser import ASTIndex, build_index, format_ast_context_for_llm

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "microsoft/codebert-base")
MODEL = SentenceTransformer(MODEL_NAME)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")


def _build_client() -> chromadb.api.ClientAPI:
    """Create the ChromaDB client.

    Tries ``PersistentClient(CHROMA_PATH)`` first. On ``PermissionError`` or
    other OS errors (read-only path, disk full at init), falls back to the
    in-memory ``Client`` so the agent can still run — just without
    cross-restart persistence. A warning is logged in the fallback case.
    """
    try:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        return chromadb.PersistentClient(path=CHROMA_PATH)
    except PermissionError as e:
        logger.error(
            f"ChromaDB path {CHROMA_PATH!r} not writable ({e}); "
            "falling back to in-memory storage. Index will reset on restart."
        )
        return chromadb.Client()
    except OSError as e:
        logger.error(
            f"OSError opening ChromaDB at {CHROMA_PATH!r} ({e}); "
            "falling back to in-memory storage."
        )
        return chromadb.Client()
    except Exception as e:
        logger.error(
            f"Unexpected error opening ChromaDB at {CHROMA_PATH!r}: {e}; "
            "falling back to in-memory storage.",
            exc_info=True,
        )
        return chromadb.Client()


CLIENT = _build_client()

# Per-repo collection cache: repo_name → ChromaDB Collection.
_collections: Dict[str, Any] = {}

# Sanitized-name registry: sanitized_name → original repo_name.
# Used to detect collisions where two distinct repo_names sanitize to the
# same string (e.g. ``owner/foo-bar`` and ``owner/foo_bar``).
_name_registry: Dict[str, str] = {}

# Backwards compatibility: some code paths (older tests, ``repo_cache``)
# import ``COLLECTION``. Bind it to the default collection.
DEFAULT_REPO_NAME = "default"

# Global AST index — built alongside the ChromaDB index in ``index_codebase``.
# Consumed by ``get_ast_context`` so other modules (e.g. ``llm_reviewer``) can
# fetch structural context for a diff without re-walking the repo.
AST_INDEX: Optional[ASTIndex] = None


# ─── Collection name handling ─────────────────────────────────────────


_VALID_CHAR_RE = re.compile(r"[^a-zA-Z0-9_]")


def _sanitize_repo_name(repo_name: str) -> str:
    """Convert a GitHub-style repo name into a ChromaDB-safe collection name.

    Rules (from spec):
      - Replace ``/``, ``-``, ``.`` with ``_``.
      - Drop any character that is not alphanumeric or underscore.
      - If the result starts with a digit, prepend ``r_``.
      - Truncate to 63 characters max.
      - Empty result → raise ``ValueError``.

    Also enforces ChromaDB's minimum-length-3 requirement by padding short
    names with trailing underscores.
    """
    if not repo_name or not repo_name.strip():
        raise ValueError("Invalid repo name: empty or whitespace")

    # Replace separators with underscore, then strip any other invalid chars.
    name = repo_name.replace("/", "_").replace("-", "_").replace(".", "_")
    name = _VALID_CHAR_RE.sub("", name)

    if not name:
        raise ValueError(f"Invalid repo name: {repo_name!r} sanitizes to empty string")

    if name[0].isdigit():
        name = "r_" + name

    # ChromaDB collection names must be 3-63 chars.
    if len(name) > 63:
        name = name[:63]
    while len(name) < 3:
        name = name + "_"

    return name


def _collection_name_for(repo_name: str) -> str:
    """Return a stable, collision-free collection name for ``repo_name``.

    If two distinct repo names sanitize to the same string, append a short
    hash of the original name to disambiguate.
    """
    base = _sanitize_repo_name(repo_name)

    claimed = _name_registry.get(base)
    if claimed is None or claimed == repo_name:
        _name_registry[base] = repo_name
        return base

    # Collision with a different repo — disambiguate with a hash suffix.
    suffix = "_" + hashlib.sha1(repo_name.encode("utf-8")).hexdigest()[:8]
    candidate = (base[: 63 - len(suffix)] + suffix) if (len(base) + len(suffix)) > 63 else (base + suffix)
    _name_registry[candidate] = repo_name
    logger.info(
        f"Collection name collision for {repo_name!r} on base {base!r}; "
        f"using {candidate!r}"
    )
    return candidate


def get_collection(repo_name: str):
    """Get or create the ChromaDB collection for ``repo_name``.

    The collection name is derived by sanitizing the repo name. Multiple calls
    for the same repo return the cached collection object.

    Raises:
        ValueError: if ``repo_name`` is empty or sanitizes to an unusable name.
    """
    if not repo_name:
        raise ValueError("repo_name must be a non-empty string")

    cached = _collections.get(repo_name)
    if cached is not None:
        return cached

    collection_name = _collection_name_for(repo_name)
    try:
        collection = CLIENT.get_or_create_collection(collection_name)
    except Exception as e:
        logger.error(
            f"Failed to get_or_create_collection {collection_name!r} for {repo_name!r}: {e}",
            exc_info=True,
        )
        raise

    _collections[repo_name] = collection
    return collection


def delete_collection(repo_name: str) -> None:
    """Delete the ChromaDB collection for ``repo_name``.

    Used when invalidating cache (e.g. push to default branch). Does not
    raise on missing collections or ChromaDB errors — just logs and returns.
    """
    if not repo_name:
        logger.warning("delete_collection called with empty repo_name")
        return

    try:
        collection_name = _collection_name_for(repo_name)
    except ValueError as e:
        logger.warning(f"delete_collection: cannot sanitize {repo_name!r}: {e}")
        return

    try:
        CLIENT.delete_collection(collection_name)
        logger.info(f"Deleted ChromaDB collection {collection_name!r} for {repo_name!r}")
    except Exception as e:
        # ChromaDB raises if the collection doesn't exist — that's fine.
        logger.warning(f"Could not delete collection {collection_name!r}: {e}")

    _collections.pop(repo_name, None)
    _name_registry.pop(collection_name, None)


def list_indexed_repos() -> List[str]:
    """Return the sanitized names of every collection currently stored in ChromaDB."""
    try:
        cols = CLIENT.list_collections()
    except Exception as e:
        logger.error(f"list_indexed_repos: ChromaDB list_collections failed: {e}", exc_info=True)
        return []

    names: List[str] = []
    for c in cols:
        # ChromaDB returns objects with ``.name``; in older versions list of strings.
        name = getattr(c, "name", None)
        if name is None and isinstance(c, str):
            name = c
        if name:
            names.append(name)
    return names


# Bind the default collection now so ``COLLECTION`` keeps working for any
# existing callers that imported it. Build lazily so import doesn't fail if
# ChromaDB is unhappy at startup.
try:
    COLLECTION = get_collection(DEFAULT_REPO_NAME)
except Exception as e:  # pragma: no cover — defensive only
    logger.error(f"Could not create default collection: {e}", exc_info=True)
    COLLECTION = None


# ─── Chunking + indexing ──────────────────────────────────────────────


def chunk_file(filepath, chunk_size=50, overlap=10):
    """Split a file into overlapping line-windows for embedding."""
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


def index_codebase(repo_path, repo_name: str = DEFAULT_REPO_NAME):
    """Build (or refresh) the vector + AST indexes for one repo.

    Args:
        repo_path: Local filesystem path to the cloned repo.
        repo_name: Logical repo name (e.g. ``"owner/repo"``). Determines which
            per-repo ChromaDB collection is used. Defaults to ``"default"`` for
            backwards compatibility with older single-collection callers.

    Skips indexing when the collection already contains chunks — trust the
    count over disk state, as per spec. Push-event handlers should call
    ``delete_collection`` first to force a rebuild.
    """
    print(f"\nIndexing codebase at: {repo_path} (repo={repo_name})")

    try:
        collection = get_collection(repo_name)
    except ValueError as e:
        logger.error(f"index_codebase: invalid repo_name {repo_name!r}: {e}")
        return
    except Exception as e:
        logger.error(f"index_codebase: get_collection failed for {repo_name!r}: {e}", exc_info=True)
        return

    try:
        existing = collection.count()
    except Exception as e:
        logger.warning(f"collection.count() failed for {repo_name!r}: {e}")
        existing = 0

    if existing > 0:
        print(f"Codebase already indexed for {repo_name} ({existing} chunks). Skipping vector index.")
    else:
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
        else:
            texts = [c["text"] for c in all_chunks]
            try:
                embeddings = MODEL.encode(texts).tolist()
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}", exc_info=True)
                embeddings = None

            if embeddings is not None:
                ids = [f"chunk_{i}" for i in range(len(all_chunks))]
                metadatas = [
                    {
                        "filename": c["filename"],
                        "start_line": c["start_line"],
                        "end_line": c["end_line"],
                    }
                    for c in all_chunks
                ]
                try:
                    collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=texts,
                        metadatas=metadatas,
                    )
                    print(f"\nTotal chunks indexed: {len(all_chunks)}")
                except OSError as e:
                    logger.error(
                        f"OSError writing chunks to ChromaDB (disk full?): {e}", exc_info=True
                    )
                except Exception as e:
                    logger.error(f"collection.add failed: {e}", exc_info=True)

    # Build the AST index alongside the vector index. Failures here must not
    # break RAG retrieval — we just log and continue without AST context.
    global AST_INDEX
    try:
        AST_INDEX = build_index(repo_path)
        logger.info("AST index built successfully")
    except Exception as e:
        AST_INDEX = None
        logger.error(f"Failed to build AST index for {repo_path}: {e}", exc_info=True)


def get_ast_context(diff_text: str) -> str:
    """Return formatted AST context for a PR diff.

    Returns an empty string if the AST index has not been built yet (e.g.
    ``index_codebase`` was never called) or if formatting fails. Callers should
    treat an empty string as "no AST context available" rather than an error.
    """
    if AST_INDEX is None:
        return ""
    try:
        return format_ast_context_for_llm(AST_INDEX, diff_text or "")
    except Exception as e:
        logger.error(f"Failed to format AST context: {e}", exc_info=True)
        return ""


def retrieve_context(diff_text, repo_name: str = DEFAULT_REPO_NAME, n_results: int = 3):
    """Retrieve the top-N semantically similar code chunks from a repo's index.

    Args:
        diff_text: PR diff or arbitrary query string.
        repo_name: Repo whose collection to query. Defaults to ``"default"``.
        n_results: Max chunks to return. Capped at the collection size.

    Returns ``[]`` for any of these — never raises:
        - ``diff_text`` is empty or ``None``.
        - The collection doesn't exist yet for this repo.
        - The collection is empty.
        - ChromaDB query fails.
    """
    if diff_text is None or diff_text == "":
        return []

    try:
        collection = get_collection(repo_name)
    except ValueError as e:
        logger.warning(f"retrieve_context: invalid repo_name {repo_name!r}: {e}")
        return []
    except Exception as e:
        logger.error(f"retrieve_context: get_collection failed: {e}", exc_info=True)
        return []

    try:
        count = collection.count()
    except Exception as e:
        logger.warning(f"retrieve_context: collection.count failed: {e}")
        return []

    if count == 0:
        logger.warning(
            f"retrieve_context: collection for {repo_name!r} is empty — index not built?"
        )
        return []

    effective_n = max(1, min(n_results, count))

    try:
        query_embedding = MODEL.encode([diff_text]).tolist()
    except Exception as e:
        logger.error(f"retrieve_context: embedding failed: {e}", exc_info=True)
        return []

    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=effective_n,
        )
    except Exception as e:
        logger.error(f"retrieve_context: ChromaDB query failed: {e}", exc_info=True)
        return []

    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]

    context_chunks = []
    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) else {}
        context_chunks.append({
            "text": docs[i],
            "filename": meta.get("filename"),
            "start_line": meta.get("start_line"),
        })

    return context_chunks
