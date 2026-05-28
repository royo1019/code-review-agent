"""Async queue tests for queue_manager.py."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

import queue_manager as qm


@pytest.fixture(autouse=True)
def fast_timers(monkeypatch):
    """Shorten timeouts so tests run in seconds, not minutes."""
    monkeypatch.setattr(qm, "WORKER_IDLE_TIMEOUT_SECONDS", 2)
    monkeypatch.setattr(qm, "JOB_TIMEOUT_SECONDS", 2)
    monkeypatch.setattr(qm, "INTER_JOB_DELAY_SECONDS", 0)


async def _drain(mgr, target_stat_key, expected, timeout=10):
    """Spin on get_status() until ``stats[key] >= expected`` or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = await mgr.get_status()
        if status["stats"].get(target_stat_key, 0) >= expected:
            return status
        await asyncio.sleep(0.05)
    return await mgr.get_status()


# ── Input validation ──────────────────────────────────────────────────


async def test_enqueue_empty_repo_returns_false():
    mgr = qm.QueueManager(processor=lambda r, n: None)
    assert await mgr.enqueue_pr("", 1) is False
    await mgr.shutdown()


async def test_enqueue_zero_pr_returns_false():
    mgr = qm.QueueManager(processor=lambda r, n: None)
    assert await mgr.enqueue_pr("o/r", 0) is False
    await mgr.shutdown()


async def test_enqueue_negative_pr_returns_false():
    mgr = qm.QueueManager(processor=lambda r, n: None)
    assert await mgr.enqueue_pr("o/r", -5) is False
    await mgr.shutdown()


async def test_enqueue_non_int_pr_returns_false():
    mgr = qm.QueueManager(processor=lambda r, n: None)
    assert await mgr.enqueue_pr("o/r", "abc") is False
    await mgr.shutdown()


# ── Processing ────────────────────────────────────────────────────────


async def test_jobs_processed_in_order():
    seen = []
    mgr = qm.QueueManager(processor=lambda r, n: seen.append(n))
    for i in range(3):
        await mgr.enqueue_pr("o/r", i + 1)
    await _drain(mgr, "total_processed", 3)
    await mgr.shutdown()
    assert seen == [1, 2, 3]


async def test_same_repo_processes_sequentially():
    timeline = []

    def proc(repo, pr):
        timeline.append((pr, "start", time.time()))
        time.sleep(0.1)
        timeline.append((pr, "end", time.time()))

    mgr = qm.QueueManager(processor=proc)
    await mgr.enqueue_pr("o/r", 1)
    await mgr.enqueue_pr("o/r", 2)
    await _drain(mgr, "total_processed", 2)
    await mgr.shutdown()

    end_1 = next(t for pr, kind, t in timeline if pr == 1 and kind == "end")
    start_2 = next(t for pr, kind, t in timeline if pr == 2 and kind == "start")
    assert end_1 <= start_2 + 0.01  # tiny grace for clock jitter


async def test_different_repos_process_in_parallel():
    starts = {}

    def proc(repo, pr):
        starts[repo] = time.time()
        time.sleep(0.2)

    mgr = qm.QueueManager(processor=proc)
    await mgr.enqueue_pr("o/A", 1)
    await mgr.enqueue_pr("o/B", 1)
    await _drain(mgr, "total_processed", 2)
    await mgr.shutdown()

    assert abs(starts["o/A"] - starts["o/B"]) < 0.1


# ── Retry semantics ───────────────────────────────────────────────────


async def test_retry_on_exception():
    calls = {"n": 0}

    def flaky(repo, pr):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")

    mgr = qm.QueueManager(processor=flaky)
    await mgr.enqueue_pr("o/r", 1)
    final = await _drain(mgr, "total_processed", 1)
    await mgr.shutdown()
    assert calls["n"] == 3
    assert final["stats"]["total_retried"] == 2
    assert final["stats"]["total_failed"] == 0


async def test_max_attempts_marks_failed():
    def always_fail(repo, pr):
        raise RuntimeError("permanent")

    mgr = qm.QueueManager(processor=always_fail)
    await mgr.enqueue_pr("o/r", 1)
    final = await _drain(mgr, "total_failed", 1)
    await mgr.shutdown()
    assert final["stats"]["total_failed"] == 1


async def test_timeout_marks_failed_without_retry():
    def slow(repo, pr):
        time.sleep(qm.JOB_TIMEOUT_SECONDS + 1)

    mgr = qm.QueueManager(processor=slow)
    await mgr.enqueue_pr("o/r", 1)
    final = await _drain(mgr, "total_failed", 1, timeout=qm.JOB_TIMEOUT_SECONDS + 5)
    await mgr.shutdown()
    assert final["stats"]["total_failed"] == 1
    assert final["stats"]["total_retried"] == 0


# ── Status + lifecycle ────────────────────────────────────────────────


async def test_get_status_shape():
    mgr = qm.QueueManager(processor=lambda r, n: time.sleep(0.5))
    await mgr.enqueue_pr("o/r", 1)
    await asyncio.sleep(0.1)
    s = await mgr.get_status()
    assert {"queues", "stats", "total_active_repos"} <= set(s.keys())
    q = s["queues"].get("o/r", {})
    assert {"depth", "currently_processing", "worker_alive"} <= set(q.keys())
    # cancel and shut down
    mgr._shutting_down = True
    for w in list(mgr._workers.values()):
        w.cancel()
    await asyncio.gather(*mgr._workers.values(), return_exceptions=True)


async def test_shutdown_drains_pending_jobs():
    done = []

    def proc(repo, pr):
        time.sleep(0.05)
        done.append(pr)

    mgr = qm.QueueManager(processor=proc)
    for i in range(5):
        await mgr.enqueue_pr("o/r", i + 1)
    await asyncio.sleep(0.05)  # let worker start
    await mgr.shutdown()
    assert len(done) == 5


async def test_duplicate_enqueue_allowed():
    seen = []
    mgr = qm.QueueManager(processor=lambda r, n: seen.append(n))
    r1 = await mgr.enqueue_pr("o/r", 7)
    r2 = await mgr.enqueue_pr("o/r", 7)
    await _drain(mgr, "total_processed", 2)
    await mgr.shutdown()
    assert r1 and r2
    assert seen == [7, 7]


async def test_enqueue_rejected_after_shutdown():
    mgr = qm.QueueManager(processor=lambda r, n: None)
    await mgr.shutdown()
    assert await mgr.enqueue_pr("o/r", 1) is False


# ── Capacity ──────────────────────────────────────────────────────────


async def test_queue_capacity_rejects(monkeypatch):
    monkeypatch.setattr(qm, "QUEUE_MAX_SIZE", 3)

    def slow(repo, pr):
        time.sleep(5)  # block worker so queue fills up

    mgr = qm.QueueManager(processor=slow)
    results = [await mgr.enqueue_pr("o/r", i + 1) for i in range(6)]

    # cancel the slow workers so the test exits fast
    mgr._shutting_down = True
    for w in list(mgr._workers.values()):
        w.cancel()
    await asyncio.gather(*mgr._workers.values(), return_exceptions=True)

    assert sum(1 for r in results if r) >= 3
    assert sum(1 for r in results if not r) >= 2
