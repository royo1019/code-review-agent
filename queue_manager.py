"""Async queue for concurrent PR reviews.

Maintains one ``asyncio.Queue`` per repo with a dedicated worker coroutine.
This gives us:

- Parallelism across repos — repo A and repo B are reviewed at the same time.
- Sequential processing within a repo — PR #1 finishes before PR #2 starts on
  the same repo, which keeps the per-repo RAG index from being mutated from
  two workers at once.

The queue manager is owned by ``server.py`` via the module-level ``queue_manager``
singleton at the bottom of this file.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Tunables — module-level so tests can monkey-patch them down for fast runs.
QUEUE_MAX_SIZE = 50
WORKER_IDLE_TIMEOUT_SECONDS = 300  # 5 min idle → worker exits
JOB_TIMEOUT_SECONDS = 300          # 5 min per job → fail
INTER_JOB_DELAY_SECONDS = 1


@dataclass
class PRJob:
    """A single queued PR review request."""

    repo_name: str
    pr_number: int
    queued_at: datetime = field(default_factory=datetime.now)
    attempts: int = 0
    max_attempts: int = 3


def _default_processor(repo_name: str, pr_number: int) -> None:
    """Default job processor — imports ``server.process_pr`` lazily.

    The lazy import breaks the circular dependency: ``server.py`` imports the
    ``queue_manager`` singleton, while the worker needs to call ``process_pr``
    inside ``server.py``. By deferring the import to call time, both modules
    load cleanly.
    """
    from server import process_pr
    process_pr(repo_name, pr_number)


class QueueManager:
    """Per-repo async queue with worker coroutines."""

    def __init__(self, processor=None):
        """Initialize the manager. Workers are created on-demand by ``enqueue_pr``.

        Args:
            processor: Optional override for the per-job callable. Useful for
                tests so the queue can drive a stub instead of ``process_pr``.
                Must be a synchronous callable taking ``(repo_name, pr_number)``;
                it is run inside ``asyncio.to_thread`` so it cannot block the
                event loop.
        """
        self._queues: Dict[str, asyncio.Queue[PRJob]] = {}
        self._workers: Dict[str, asyncio.Task] = {}
        self._processing: Dict[str, Optional[PRJob]] = {}
        self._stats: Dict[str, int] = {
            "total_queued": 0,
            "total_processed": 0,
            "total_failed": 0,
            "total_retried": 0,
        }
        self._lock = asyncio.Lock()
        self._processor = processor or _default_processor
        self._shutting_down = False

    # ── Public API ────────────────────────────────────────────────────

    async def enqueue_pr(self, repo_name: str, pr_number: int) -> bool:
        """Enqueue a PR review job for ``repo_name``.

        Returns:
            ``True`` if the job was queued.
            ``False`` if the input is invalid, the per-repo queue is full,
            or the event loop is unavailable.

        Duplicate enqueues (same PR # for same repo) are allowed — GitHub may
        send duplicate webhooks and the second one is cheap to drop later if
        nothing has changed.
        """
        if not repo_name or not isinstance(repo_name, str):
            logger.error(f"enqueue_pr: invalid repo_name {repo_name!r}")
            return False
        if not isinstance(pr_number, int) or pr_number <= 0:
            logger.error(f"enqueue_pr: invalid pr_number {pr_number!r}")
            return False
        if self._shutting_down:
            logger.warning(
                f"enqueue_pr: refusing PR #{pr_number} for {repo_name} — shutting down"
            )
            return False

        job = PRJob(repo_name=repo_name, pr_number=pr_number)

        async with self._lock:
            queue = self._queues.get(repo_name)
            if queue is None:
                try:
                    queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
                except Exception as e:
                    logger.error(
                        f"enqueue_pr: could not create queue for {repo_name}: {e}",
                        exc_info=True,
                    )
                    return False
                self._queues[repo_name] = queue
                self._processing[repo_name] = None

            if queue.full():
                logger.warning(
                    f"Queue full for {repo_name} (size={queue.qsize()}); "
                    f"rejecting PR #{pr_number}"
                )
                return False

            try:
                queue.put_nowait(job)
            except asyncio.QueueFull:
                logger.warning(
                    f"Queue full for {repo_name} (race); rejecting PR #{pr_number}"
                )
                return False

            self._stats["total_queued"] += 1

            worker = self._workers.get(repo_name)
            if worker is None or worker.done():
                try:
                    self._workers[repo_name] = asyncio.create_task(
                        self._worker(repo_name),
                        name=f"queue-worker:{repo_name}",
                    )
                except RuntimeError as e:
                    # No running event loop — caller must be in async context.
                    logger.error(
                        f"enqueue_pr: cannot start worker for {repo_name}: {e}"
                    )
                    return False

        logger.info(
            f"Queued PR #{pr_number} for {repo_name} (queue depth: {queue.qsize()})"
        )
        return True

    async def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of queue depths, currently processing jobs, and stats."""
        async with self._lock:
            queues_status: Dict[str, Dict[str, Any]] = {}
            for repo_name, queue in self._queues.items():
                worker = self._workers.get(repo_name)
                processing = self._processing.get(repo_name)
                queues_status[repo_name] = {
                    "depth": queue.qsize(),
                    "currently_processing": processing.pr_number if processing else None,
                    "worker_alive": bool(worker and not worker.done()),
                }
            active = sum(1 for w in self._workers.values() if not w.done())
            return {
                "queues": queues_status,
                "stats": dict(self._stats),
                "total_active_repos": active,
            }

    async def shutdown(self) -> None:
        """Gracefully shut down the queue manager.

        Stops accepting new jobs, then waits up to 30 seconds for in-flight
        workers to drain naturally. Any worker still running after the
        deadline is cancelled.
        """
        self._shutting_down = True

        async with self._lock:
            workers = list(self._workers.values())

        if not workers:
            logger.info("QueueManager shutdown complete (no active workers)")
            return

        try:
            await asyncio.wait_for(
                asyncio.gather(*workers, return_exceptions=True),
                timeout=30,
            )
        except asyncio.TimeoutError:
            logger.warning("QueueManager shutdown: drain timed out — cancelling workers")
            for w in workers:
                if not w.done():
                    w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        logger.info("QueueManager shutdown complete")

    # ── Internal ──────────────────────────────────────────────────────

    async def _worker(self, repo_name: str) -> None:
        """Per-repo worker loop. Processes jobs sequentially until idle timeout."""
        logger.debug(f"Worker started for {repo_name}")
        try:
            while True:
                queue = self._queues.get(repo_name)
                if queue is None:
                    logger.debug(f"Worker for {repo_name}: queue gone, exiting")
                    return

                try:
                    job = await asyncio.wait_for(
                        queue.get(),
                        timeout=WORKER_IDLE_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.debug(
                        f"Worker for {repo_name}: idle for "
                        f"{WORKER_IDLE_TIMEOUT_SECONDS}s, exiting"
                    )
                    return

                self._processing[repo_name] = job
                logger.info(
                    f"Processing PR #{job.pr_number} for {repo_name} "
                    f"(attempt {job.attempts + 1}/{job.max_attempts})"
                )

                outcome = await self._run_job(job)
                self._processing[repo_name] = None
                queue.task_done()

                if outcome == "ok":
                    self._stats["total_processed"] += 1
                elif outcome == "retry":
                    self._stats["total_retried"] += 1
                elif outcome == "failed":
                    self._stats["total_failed"] += 1

                # Small breather between jobs so we don't spin.
                try:
                    await asyncio.sleep(INTER_JOB_DELAY_SECONDS)
                except asyncio.CancelledError:
                    raise
        except asyncio.CancelledError:
            logger.info(f"Worker for {repo_name} cancelled")
            raise
        finally:
            async with self._lock:
                # Only clean up our own slot — a new worker may have already
                # been spawned for this repo by a concurrent enqueue.
                if self._workers.get(repo_name) is asyncio.current_task():
                    self._workers.pop(repo_name, None)
                if not (self._queues.get(repo_name) and not self._queues[repo_name].empty()):
                    self._queues.pop(repo_name, None)
                self._processing.pop(repo_name, None)
            logger.debug(f"Worker for {repo_name} stopped")

    async def _run_job(self, job: PRJob) -> str:
        """Run a single job through the processor with timeout + retry semantics.

        Returns one of ``"ok"``, ``"retry"``, ``"failed"`` so the caller can
        update stats. On exceptions the job is re-queued unless we've already
        exhausted ``max_attempts``. Timeouts are terminal — no retry.
        """
        job.attempts += 1
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._processor, job.repo_name, job.pr_number),
                timeout=JOB_TIMEOUT_SECONDS,
            )
            logger.info(
                f"Completed PR #{job.pr_number} for {job.repo_name}"
            )
            return "ok"
        except asyncio.TimeoutError:
            logger.error(
                f"PR #{job.pr_number} for {job.repo_name} timed out after "
                f"{JOB_TIMEOUT_SECONDS}s — marking failed (no retry)"
            )
            return "failed"
        except asyncio.CancelledError:
            # Worker is being cancelled (shutdown). Propagate.
            raise
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(
                f"PR #{job.pr_number} for {job.repo_name} failed on attempt "
                f"{job.attempts}: {e}\n{tb}"
            )
            if job.attempts < job.max_attempts:
                logger.info(
                    f"Re-queuing PR #{job.pr_number} for {job.repo_name} "
                    f"(attempt {job.attempts + 1}/{job.max_attempts})"
                )
                queue = self._queues.get(job.repo_name)
                if queue is not None:
                    try:
                        queue.put_nowait(job)
                        return "retry"
                    except asyncio.QueueFull:
                        logger.error(
                            f"Cannot retry PR #{job.pr_number} for {job.repo_name}: queue full"
                        )
                        return "failed"
                return "failed"
            logger.error(
                f"PR #{job.pr_number} for {job.repo_name}: exhausted "
                f"{job.max_attempts} attempts — marking failed"
            )
            return "failed"


# Module-level singleton.
queue_manager = QueueManager()
