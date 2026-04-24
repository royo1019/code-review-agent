import os
import shutil
import tempfile
from git import Repo
from rag import index_codebase, COLLECTION
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# in-memory cache: repo_name → local path
_cache = {}

def clone_repo(repo_name: str) -> str:
    print(f"\nCloning {repo_name}...")
    temp_dir = tempfile.mkdtemp()
    clone_url = f"https://{GITHUB_TOKEN}@github.com/{repo_name}.git"
    Repo.clone_from(clone_url, temp_dir)
    print(f"Cloned to {temp_dir}")
    return temp_dir

def get_repo_path(repo_name: str) -> str:
    # return cached path if already indexed
    if repo_name in _cache:
        print(f"Using cached index for {repo_name}")
        return _cache[repo_name]

    # clone and index fresh
    repo_path = clone_repo(repo_name)
    index_codebase(repo_path)

    # cache for future PRs
    _cache[repo_name] = repo_path
    print(f"Cached index for {repo_name}")

    return repo_path

def invalidate_cache(repo_name: str):
    if repo_name in _cache:
        old_path = _cache.pop(repo_name)
        shutil.rmtree(old_path, ignore_errors=True)
        print(f"Cache invalidated for {repo_name}")