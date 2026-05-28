"""Unit tests for github_app.py.

Generates a real RSA key via ``cryptography`` (already a dep) so JWT signing
can be tested end-to-end without an external file. Network calls are mocked
via ``monkeypatch`` on ``github_app.requests.post``.
"""

from __future__ import annotations

import base64
import time

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import github_app


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rsa_keypair():
    """Generate one RSA-2048 keypair for the whole test module."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    public_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    return private_pem, public_pem


@pytest.fixture(autouse=True)
def clear_caches():
    """Reset module-level caches between tests."""
    github_app.invalidate_installation_cache(None)
    github_app._repo_to_installation.clear()


# ── _normalize_private_key ───────────────────────────────────────────


def test_normalize_handles_windows_line_endings(rsa_keypair):
    pem, _ = rsa_keypair
    crlf = pem.replace("\n", "\r\n")
    out = github_app._normalize_private_key(crlf)
    assert "\r" not in out
    assert "-----BEGIN" in out


def test_normalize_decodes_base64_pem(rsa_keypair):
    pem, _ = rsa_keypair
    encoded = base64.b64encode(pem.encode("utf-8")).decode("ascii")
    out = github_app._normalize_private_key(encoded)
    assert "-----BEGIN" in out


def test_normalize_passthrough_for_normal_pem(rsa_keypair):
    pem, _ = rsa_keypair
    assert github_app._normalize_private_key(pem) == pem


def test_normalize_empty_raises():
    with pytest.raises(ValueError):
        github_app._normalize_private_key("")


# ── generate_jwt ──────────────────────────────────────────────────────


def test_generate_jwt_round_trip(rsa_keypair):
    private_pem, public_pem = rsa_keypair
    token = github_app.generate_jwt("12345", private_pem)
    decoded = pyjwt.decode(token, public_pem, algorithms=["RS256"])
    assert decoded["iss"] == "12345"
    # iat ≤ now-60 ≤ exp; with current clock both should be in expected window
    now = int(time.time())
    assert decoded["iat"] <= now
    assert decoded["exp"] > now
    assert decoded["exp"] - decoded["iat"] <= 700  # ~11 minutes max


def test_generate_jwt_missing_app_id_raises(rsa_keypair):
    private_pem, _ = rsa_keypair
    with pytest.raises(ValueError, match="app_id"):
        github_app.generate_jwt(None, private_pem)
    with pytest.raises(ValueError, match="app_id"):
        github_app.generate_jwt("", private_pem)


def test_generate_jwt_missing_private_key_raises():
    with pytest.raises(ValueError, match="private key"):
        github_app.generate_jwt("12345", None)
    with pytest.raises(ValueError, match="private key"):
        github_app.generate_jwt("12345", "")


def test_generate_jwt_invalid_private_key_raises():
    with pytest.raises(ValueError):
        github_app.generate_jwt("12345", "-----BEGIN RSA PRIVATE KEY-----\nnot real\n-----END RSA PRIVATE KEY-----")


# ── get_installation_token (mocked HTTP) ──────────────────────────────


class _FakeResponse:
    def __init__(self, status_code, json_body=None, text=""):
        self.status_code = status_code
        self._json = json_body or {}
        self.text = text

    def json(self):
        return self._json


def test_token_success(monkeypatch, rsa_keypair):
    pem, _ = rsa_keypair
    calls = []

    def fake_post(url, headers=None, timeout=None):
        calls.append({"url": url, "headers": headers})
        return _FakeResponse(201, {"token": "ghs_mocked_token_xyz", "expires_at": "..."})

    monkeypatch.setattr(github_app.requests, "post", fake_post)
    token = github_app.get_installation_token("12345", pem, 99)
    assert token == "ghs_mocked_token_xyz"
    assert len(calls) == 1
    assert "installations/99/access_tokens" in calls[0]["url"]
    assert "Bearer" in calls[0]["headers"]["Authorization"]


def test_token_retries_on_5xx_then_succeeds(monkeypatch, rsa_keypair):
    pem, _ = rsa_keypair
    monkeypatch.setattr(github_app.time, "sleep", lambda s: None)  # skip backoff

    responses = [
        _FakeResponse(503, text="busy"),
        _FakeResponse(502, text="bad gateway"),
        _FakeResponse(201, {"token": "tok_after_retries"}),
    ]
    calls = {"n": 0}

    def fake_post(url, headers=None, timeout=None):
        r = responses[calls["n"]]
        calls["n"] += 1
        return r

    monkeypatch.setattr(github_app.requests, "post", fake_post)
    assert github_app.get_installation_token("12345", pem, 99) == "tok_after_retries"
    assert calls["n"] == 3


def test_token_404_raises_clear_error(monkeypatch, rsa_keypair):
    pem, _ = rsa_keypair
    monkeypatch.setattr(github_app.requests, "post",
                        lambda *a, **k: _FakeResponse(404, text="Not Found"))
    with pytest.raises(RuntimeError, match="not found"):
        github_app.get_installation_token("12345", pem, 99)


def test_token_401_raises_clear_error(monkeypatch, rsa_keypair):
    pem, _ = rsa_keypair
    monkeypatch.setattr(github_app.requests, "post",
                        lambda *a, **k: _FakeResponse(401, text="Bad credentials"))
    with pytest.raises(RuntimeError, match="(JWT|401)"):
        github_app.get_installation_token("12345", pem, 99)


def test_token_invalid_installation_id_raises(rsa_keypair):
    pem, _ = rsa_keypair
    for bad in (0, -1, None, "abc"):
        with pytest.raises(ValueError):
            github_app.get_installation_token("12345", pem, bad)


def test_token_all_retries_fail_raises(monkeypatch, rsa_keypair):
    pem, _ = rsa_keypair
    monkeypatch.setattr(github_app.time, "sleep", lambda s: None)
    monkeypatch.setattr(github_app.requests, "post",
                        lambda *a, **k: _FakeResponse(503, text="busy"))
    with pytest.raises(RuntimeError, match="after"):
        github_app.get_installation_token("12345", pem, 99)


def test_token_network_error_retries_then_fails(monkeypatch, rsa_keypair):
    pem, _ = rsa_keypair
    monkeypatch.setattr(github_app.time, "sleep", lambda s: None)

    import requests
    def boom(*a, **k):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(github_app.requests, "post", boom)
    with pytest.raises(RuntimeError):
        github_app.get_installation_token("12345", pem, 99)


# ── get_github_client_for_installation ────────────────────────────────


def test_client_caches_per_installation(monkeypatch, rsa_keypair):
    pem, _ = rsa_keypair
    monkeypatch.setenv("GITHUB_APP_ID", "12345")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", pem)

    token_calls = {"n": 0}
    def fake_get_token(*a, **k):
        token_calls["n"] += 1
        return f"tok_{token_calls['n']}"

    monkeypatch.setattr(github_app, "get_installation_token", fake_get_token)

    c1 = github_app.get_github_client_for_installation(7)
    c2 = github_app.get_github_client_for_installation(7)
    assert c1 is c2  # cached
    assert token_calls["n"] == 1

    c3 = github_app.get_github_client_for_installation(8)
    assert c3 is not c1  # different installation → different client
    assert token_calls["n"] == 2


def test_client_refreshes_after_ttl_expiry(monkeypatch, rsa_keypair):
    pem, _ = rsa_keypair
    monkeypatch.setenv("GITHUB_APP_ID", "12345")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", pem)

    call_count = {"n": 0}
    def fake_get_token(*a, **k):
        call_count["n"] += 1
        return f"tok_{call_count['n']}"

    monkeypatch.setattr(github_app, "get_installation_token", fake_get_token)

    # First call populates cache
    base_t = time.time()
    monkeypatch.setattr(github_app.time, "time", lambda: base_t)
    github_app.get_github_client_for_installation(7)
    assert call_count["n"] == 1

    # Move clock past TTL — cache should refresh
    monkeypatch.setattr(github_app.time, "time", lambda: base_t + 60 * 60)
    github_app.get_github_client_for_installation(7)
    assert call_count["n"] == 2


def test_client_invalid_installation_id_raises(monkeypatch, rsa_keypair):
    pem, _ = rsa_keypair
    monkeypatch.setenv("GITHUB_APP_ID", "12345")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", pem)
    for bad in (None, 0, -1, "abc"):
        with pytest.raises(ValueError):
            github_app.get_github_client_for_installation(bad)


def test_client_missing_env_vars_raises(monkeypatch):
    monkeypatch.delenv("GITHUB_APP_ID", raising=False)
    monkeypatch.delenv("GITHUB_APP_PRIVATE_KEY", raising=False)
    with pytest.raises(ValueError, match="credentials missing"):
        github_app.get_github_client_for_installation(7)


def test_invalidate_cache_clears_all():
    github_app._client_cache[1] = ("c1", time.time() + 100)
    github_app._client_cache[2] = ("c2", time.time() + 100)
    github_app.invalidate_installation_cache(None)
    assert github_app._client_cache == {}


def test_invalidate_cache_single_entry():
    github_app._client_cache[1] = ("c1", time.time() + 100)
    github_app._client_cache[2] = ("c2", time.time() + 100)
    github_app.invalidate_installation_cache(1)
    assert 1 not in github_app._client_cache
    assert 2 in github_app._client_cache


# ── Repo → installation cache ─────────────────────────────────────────


def test_register_and_lookup_repo_installation():
    github_app.register_repo_installation("owner/repo", 42)
    assert github_app.get_installation_for_repo("owner/repo") == 42


def test_lookup_unknown_repo_returns_none():
    assert github_app.get_installation_for_repo("never/registered") is None


def test_forget_repo_installation():
    github_app.register_repo_installation("owner/repo", 42)
    github_app.forget_repo_installation("owner/repo")
    assert github_app.get_installation_for_repo("owner/repo") is None


def test_register_ignores_invalid_inputs():
    github_app.register_repo_installation("", 42)
    github_app.register_repo_installation("owner/repo", 0)
    github_app.register_repo_installation("owner/repo", -1)
    github_app.register_repo_installation("owner/repo", None)
    assert github_app.get_installation_for_repo("owner/repo") is None
