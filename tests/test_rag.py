"""Unit tests for rag.py."""

from __future__ import annotations

import pytest

import rag


# ── Sanitization ──────────────────────────────────────────────────────


def test_sanitize_slashes_to_underscore():
    assert rag._sanitize_repo_name("owner/repo") == "owner_repo"


def test_sanitize_dots_and_hyphens():
    assert rag._sanitize_repo_name("a.b.c") == "a_b_c"
    assert rag._sanitize_repo_name("foo-bar") == "foo_bar"


def test_sanitize_strips_invalid_chars():
    assert rag._sanitize_repo_name("a@b!c%d") == "abcd"


def test_sanitize_digit_prefix():
    assert rag._sanitize_repo_name("1abc") == "r_1abc"


def test_sanitize_length_cap():
    n = rag._sanitize_repo_name("a" * 200)
    assert len(n) == 63


def test_sanitize_short_padded():
    n = rag._sanitize_repo_name("ab")
    assert len(n) >= 3


def test_sanitize_empty_raises():
    with pytest.raises(ValueError):
        rag._sanitize_repo_name("")


def test_sanitize_all_invalid_raises():
    with pytest.raises(ValueError):
        rag._sanitize_repo_name("@@@")


# ── Collision ─────────────────────────────────────────────────────────


def test_collision_different_repos_disambiguated():
    a = rag._collection_name_for("ns/foo-collide-x")
    b = rag._collection_name_for("ns/foo.collide.x")
    assert a != b


def test_collision_idempotent_for_same_repo():
    a = rag._collection_name_for("ns/sameRepoIdem")
    b = rag._collection_name_for("ns/sameRepoIdem")
    assert a == b


# ── get_collection ────────────────────────────────────────────────────


def test_get_collection_empty_repo_raises():
    with pytest.raises(ValueError):
        rag.get_collection("")


def test_get_collection_cached():
    c1 = rag.get_collection("ns/cached_repo_xyz")
    c2 = rag.get_collection("ns/cached_repo_xyz")
    assert c1 is c2


def test_different_repos_different_collections():
    c1 = rag.get_collection("ns/alpha_unique_abc")
    c2 = rag.get_collection("ns/beta_unique_abc")
    assert c1 is not c2


# ── retrieve_context edge cases ───────────────────────────────────────


def test_retrieve_empty_diff_returns_empty():
    assert rag.retrieve_context("", repo_name="ns/empty_diff_repo") == []


def test_retrieve_none_diff_returns_empty():
    assert rag.retrieve_context(None, repo_name="ns/none_diff_repo") == []


def test_retrieve_unknown_repo_returns_empty():
    assert rag.retrieve_context("anything", repo_name="ns/never_indexed_xxx") == []


# ── index_codebase + retrieve_context happy path ──────────────────────


def test_index_and_retrieve(sample_repo):
    rag.index_codebase(sample_repo, repo_name="ns/sample_repo_index")
    chunks = rag.retrieve_context("get_user", repo_name="ns/sample_repo_index", n_results=3)
    assert len(chunks) >= 1
    assert any("get_user" in (c.get("text") or "") for c in chunks)


def test_index_idempotent_skips_reindex(sample_repo):
    rag.index_codebase(sample_repo, repo_name="ns/idempotent_repo")
    coll = rag.get_collection("ns/idempotent_repo")
    count_before = coll.count()
    # second invocation should be a no-op for the vector store
    rag.index_codebase(sample_repo, repo_name="ns/idempotent_repo")
    assert coll.count() == count_before


def test_retrieve_n_results_clamped(sample_repo):
    rag.index_codebase(sample_repo, repo_name="ns/clamp_repo")
    coll = rag.get_collection("ns/clamp_repo")
    # ask for way more chunks than exist
    chunks = rag.retrieve_context("get_user", repo_name="ns/clamp_repo", n_results=coll.count() + 100)
    assert len(chunks) <= coll.count()


# ── delete_collection + list_indexed_repos ────────────────────────────


def test_delete_collection_removes_repo(sample_repo):
    rag.index_codebase(sample_repo, repo_name="ns/delete_me_repo")
    assert "ns_delete_me_repo" in rag.list_indexed_repos()
    rag.delete_collection("ns/delete_me_repo")
    assert "ns_delete_me_repo" not in rag.list_indexed_repos()


def test_delete_missing_collection_does_not_raise():
    # must not raise
    rag.delete_collection("ns/never_existed_zzz")


def test_delete_empty_repo_name_does_not_raise():
    rag.delete_collection("")


# ── AST_INDEX side effect of index_codebase ───────────────────────────


def test_index_codebase_populates_ast_index(sample_repo):
    rag.index_codebase(sample_repo, repo_name="ns/ast_repo_check")
    assert rag.AST_INDEX is not None
    assert "get_user" in rag.AST_INDEX.functions


def test_get_ast_context_uses_index(sample_repo):
    rag.index_codebase(sample_repo, repo_name="ns/ast_ctx_repo")
    out = rag.get_ast_context("+x = get_user(name)\n")
    assert "get_user" in out


def test_get_ast_context_empty_when_no_index(monkeypatch):
    monkeypatch.setattr(rag, "AST_INDEX", None)
    assert rag.get_ast_context("anything") == ""
