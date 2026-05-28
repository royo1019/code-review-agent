"""GitHub App authentication.

When ``GITHUB_APP_ID`` and ``GITHUB_APP_PRIVATE_KEY`` are configured, the
agent can authenticate per-installation instead of using a long-lived
personal access token. This module owns:

- JWT minting for the App identity (``generate_jwt``).
- Exchange of the JWT for a per-installation access token
  (``get_installation_token``) with retry + exponential backoff.
- A cached, ready-to-use ``Github`` client per installation
  (``get_github_client_for_installation``) — caches for 50 minutes since
  installation tokens live for 60.

``github_client.get_github_client(installation_id=...)`` consults this
module when App credentials are present; otherwise it falls back to PAT.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Dict, Tuple

import jwt as pyjwt
import requests
from github import Github

logger = logging.getLogger(__name__)

# Cache: installation_id → (Github client, expiry unix timestamp).
# Tokens last 60 min; we cache for 50 to leave headroom for in-flight calls.
_TOKEN_CACHE_TTL_SECONDS = 50 * 60

_client_cache: Dict[int, Tuple[Github, float]] = {}

# Cache: repo full name → installation_id. Populated by the ``installation``
# webhook handler so PR review code can resolve a repo to its installation
# without re-deriving it on every call.
_repo_to_installation: Dict[str, int] = {}


# ─── Private key handling ─────────────────────────────────────────────


def _normalize_private_key(private_key: str) -> str:
    """Coerce a private-key env var into PEM text PyJWT can consume.

    Handles two common deployment quirks:

    - Windows line endings (``\\r\\n``): replaced with ``\\n``.
    - Base64-encoded PEM (some secret stores require b64): detected by the
      absence of ``-----BEGIN`` and round-tripped through ``base64.b64decode``.
    """
    if not private_key:
        raise ValueError("GitHub App private key not configured")

    key = private_key.replace("\r\n", "\n").replace("\r", "\n")

    # Some secret stores require base64-encoded PEMs. Detect by the absence
    # of a BEGIN marker and try to decode; only accept the decoded bytes if
    # they themselves contain a PEM header.
    if "-----BEGIN" not in key:
        try:
            decoded = base64.b64decode(key, validate=True).decode("utf-8")
        except Exception:
            decoded = ""
        if "-----BEGIN" in decoded:
            return decoded.replace("\r\n", "\n").replace("\r", "\n")

    return key


# ─── JWT minting ──────────────────────────────────────────────────────


def generate_jwt(app_id, private_key: str) -> str:
    """Mint a short-lived JWT proving we are the GitHub App.

    Claims follow GitHub's required shape: ``iat`` set 60 s in the past to
    survive clock skew, ``exp`` set 10 min in the future (GitHub's hard
    ceiling), ``iss`` is the numeric App id.

    Raises:
        ValueError: if ``app_id`` is missing or ``private_key`` is missing /
            unparseable as RSA PEM.
    """
    if app_id is None or str(app_id).strip() == "":
        raise ValueError("app_id is required")
    if not private_key:
        raise ValueError("GitHub App private key not configured")

    key = _normalize_private_key(private_key)
    now = int(time.time())
    payload = {
        "iat": now - 60,
        "exp": now + 600,
        "iss": str(app_id),
    }

    try:
        return pyjwt.encode(payload, key, algorithm="RS256")
    except pyjwt.exceptions.InvalidKeyError as e:
        raise ValueError(f"Invalid GitHub App private key: {e}") from e
    except Exception as e:
        # PyJWT can raise non-jwt exceptions on bad key shapes
        raise ValueError(f"Could not sign JWT: {e}") from e


# ─── Installation token exchange ──────────────────────────────────────


def get_installation_token(app_id, private_key: str, installation_id: int) -> str:
    """Exchange a JWT for an installation access token.

    Retries up to three times with exponential backoff (1 s, 2 s, 4 s) on
    network errors and 5xx responses. Surfaces a clear error on 404
    (installation removed / app uninstalled).

    Raises:
        ValueError: invalid ``installation_id``.
        RuntimeError: GitHub returned a non-success status or the network
            kept failing after retries.
    """
    if not isinstance(installation_id, int) or installation_id <= 0:
        raise ValueError("installation_id must be a positive integer")

    jwt_token = generate_jwt(app_id, private_key)
    url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    delays = [1, 2, 4]
    last_error = None
    for attempt, delay in enumerate(delays, start=1):
        try:
            resp = requests.post(url, headers=headers, timeout=30)
        except requests.RequestException as e:
            last_error = e
            logger.warning(
                f"Installation token request failed (attempt {attempt}/{len(delays)}): {e}"
            )
            if attempt < len(delays):
                time.sleep(delay)
            continue

        if resp.status_code == 201:
            data = resp.json()
            token = data.get("token")
            if not token:
                raise RuntimeError(f"GitHub responded 201 but no token in body: {data}")
            return token

        if resp.status_code == 404:
            raise RuntimeError(
                f"Installation {installation_id} not found — app may be uninstalled"
            )

        if resp.status_code == 401:
            raise RuntimeError(
                f"GitHub rejected JWT for installation {installation_id} (401). "
                "Check GITHUB_APP_ID and private key."
            )

        if resp.status_code in (403, 429, 500, 502, 503, 504):
            last_error = RuntimeError(f"GitHub {resp.status_code}: {resp.text[:200]}")
            logger.warning(
                f"Installation token request returned {resp.status_code} "
                f"(attempt {attempt}/{len(delays)})"
            )
            if attempt < len(delays):
                time.sleep(delay)
            continue

        # Non-retriable status — fail immediately
        raise RuntimeError(
            f"GitHub installation token request failed: {resp.status_code} {resp.text[:200]}"
        )

    raise RuntimeError(
        f"Failed to fetch installation token after {len(delays)} attempts: {last_error}"
    )


# ─── Cached client ────────────────────────────────────────────────────


def get_github_client_for_installation(installation_id: int) -> Github:
    """Return a cached ``Github`` client authenticated as ``installation_id``.

    On cache miss, mints a JWT, exchanges it for an installation token, and
    constructs a fresh ``Github`` client. Cached for 50 minutes — well inside
    GitHub's 60-minute token lifetime.

    Raises:
        ValueError: ``installation_id`` is None / non-positive, or App
            credentials are missing from the environment.
    """
    if installation_id is None:
        raise ValueError("installation_id must be a positive integer")
    if not isinstance(installation_id, int) or installation_id <= 0:
        raise ValueError("installation_id must be a positive integer")

    now = time.time()
    cached = _client_cache.get(installation_id)
    if cached is not None:
        client, expiry = cached
        if now < expiry:
            return client
        # expired — fall through to refresh

    app_id = os.getenv("GITHUB_APP_ID")
    private_key = os.getenv("GITHUB_APP_PRIVATE_KEY")
    if not app_id or not private_key:
        raise ValueError(
            "GitHub App credentials missing — set GITHUB_APP_ID and GITHUB_APP_PRIVATE_KEY"
        )

    token = get_installation_token(app_id, private_key, installation_id)
    client = Github(token)
    _client_cache[installation_id] = (client, now + _TOKEN_CACHE_TTL_SECONDS)
    return client


def invalidate_installation_cache(installation_id: int = None) -> None:
    """Drop cached client(s) — used when an installation is revoked.

    Pass an ``installation_id`` to clear that single entry; pass ``None`` to
    clear the entire cache.
    """
    if installation_id is None:
        _client_cache.clear()
        return
    _client_cache.pop(installation_id, None)


# ─── repo → installation cache ────────────────────────────────────────


def register_repo_installation(repo_name: str, installation_id: int) -> None:
    """Remember which installation owns a given repo.

    Called by the ``installation`` and ``pull_request`` webhook handlers so
    that PR review code can resolve a repo to its installation without
    re-deriving the mapping on every call.
    """
    if not repo_name or not installation_id:
        return
    if not isinstance(installation_id, int) or installation_id <= 0:
        return
    _repo_to_installation[repo_name] = installation_id


def get_installation_for_repo(repo_name: str):
    """Look up the cached installation id for ``repo_name``, or ``None``."""
    if not repo_name:
        return None
    return _repo_to_installation.get(repo_name)


def forget_repo_installation(repo_name: str) -> None:
    """Drop the repo → installation mapping (used on uninstall)."""
    if repo_name:
        _repo_to_installation.pop(repo_name, None)
