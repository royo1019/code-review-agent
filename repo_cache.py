"""Repo cloning + per-repo index lifecycle.

Keeps an in-memory map of ``repo_name → local_path``. The first time a repo
is reviewed we clone it and trigger ``index_codebase`` (which builds both the
ChromaDB collection and the AST index). Subsequent reviews reuse the cached
clone.
"""

import logging
import os
import shutil
import tempfile

from dotenv import load_dotenv
from git import Repo

from rag import delete_collection, index_codebase

load_dotenv()

logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# in-memory cache: repo_name → local path
_cache: dict[str, str] = {}


def clone_repo(repo_name: str) -> str:
    """Clone a GitHub repo into a temp directory and return its path."""
    print(f"\nCloning {repo_name}...")
    temp_dir = tempfile.mkdtemp()
    clone_url = f"https://{GITHUB_TOKEN}@github.com/{repo_name}.git"
    Repo.clone_from(clone_url, temp_dir)
    print(f"Cloned to {temp_dir}")
    return temp_dir


def get_repo_path(repo_name: str) -> str:
    """Return a local path for the repo, cloning + indexing on first call.

    Uses ``rag.index_codebase`` with the repo-scoped collection so multiple
    repos can coexist in ChromaDB without cross-contamination.
    """
    if repo_name in _cache:
        print(f"Using cached index for {repo_name}")
        return _cache[repo_name]

    repo_path = clone_repo(repo_name)
    index_codebase(repo_path, repo_name=repo_name)

    _cache[repo_name] = repo_path
    print(f"Cached index for {repo_name}")
    return repo_path


def invalidate_cache(repo_name: str) -> None:
    """Drop the cached clone and ChromaDB collection for a repo.

    Called when a push to the default branch invalidates the index. Safe to
    call for a repo that was never cached.
    """
    if repo_name in _cache:
        old_path = _cache.pop(repo_name)
        shutil.rmtree(old_path, ignore_errors=True)
        print(f"Cache invalidated for {repo_name}")

    try:
        delete_collection(repo_name)
    except Exception as e:
        logger.error(
            f"invalidate_cache: failed to delete collection for {repo_name!r}: {e}",
            exc_info=True,
        )

    logger.info(f"Cache and index invalidated for {repo_name}")
