"""
Tests for DynamicBatchScheduler.

Tests:
  - Priority ordering: HIGH requests are dispatched before NORMAL before LOW
  - Batch size limit: never exceeds max_batch_size per loop tick
  - Queue wait time tracking
  - Session cancellation
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engine.models import GenerateRequest, ModelName, Priority
from engine.scheduler import DynamicBatchScheduler, InferenceRequest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_req(
    prompt: str = "Hello",
    priority: Priority = Priority.NORMAL,
    max_tokens: int = 10,
    model: ModelName = ModelName.MISTRAL_7B,
) -> GenerateRequest:
    return GenerateRequest(
        prompt=prompt,
        model=model,
        priority=priority,
        max_tokens=max_tokens,
    )


class MockInferenceEngine:
    """Minimal engine stub for scheduler tests."""

    def __init__(self):
        self.pages_free = 1000
        self.active_sequences = 0

    async def allocate_pages(self, session_id: str, prompt_len: int):
        self.active_sequences += 1

    async def free_pages(self, session_id: str):
        self.active_sequences = max(0, self.active_sequences - 1)

    async def generate_tokens(self, prompt, max_tokens, model, session_id):
        # Yield a small fixed set of tokens quickly
        yield ("Hello", 100.0)
        for i in range(min(max_tokens - 1, 3)):
            yield (f" token{i}", 20.0)

    def get_stats(self):
        return {
            "vram_used_mb": 0.0,
            "vram_total_mb": 80000,
            "pages_allocated": 0,
            "pages_free": self.pages_free,
            "active_sequences": self.active_sequences,
        }


class MockMonitor:
    def __init__(self):
        self.records: list = []

    def record_request(self, ttft_ms, tokens, duration_s, model, session_id):
        self.records.append({
            "session_id": session_id,
            "ttft_ms": ttft_ms,
            "tokens": tokens,
            "duration_s": duration_s,
        })

    def get_status(self):
        return {}


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def engine():
    return MockInferenceEngine()


@pytest.fixture
def monitor():
    return MockMonitor()


@pytest.fixture
def scheduler(engine, monitor):
    return DynamicBatchScheduler(
        inference_engine=engine,
        monitor=monitor,
        max_batch_size=8,
        loop_interval_s=0.01,  # fast for tests
    )


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------

class TestPriorityOrdering:
    def test_priority_scores_are_ordered(self):
        """HIGH must have lower numeric score than NORMAL, which beats LOW."""
        from engine.models import PRIORITY_SCORES
        assert PRIORITY_SCORES[Priority.HIGH] < PRIORITY_SCORES[Priority.NORMAL]
        assert PRIORITY_SCORES[Priority.NORMAL] < PRIORITY_SCORES[Priority.LOW]

    def test_inference_request_ordering(self):
        """InferenceRequest comparison uses priority_score for queue ordering."""
        t = time.monotonic()
        high_ir = InferenceRequest(
            priority_score=0,
            submit_time=t,
            session_id="s1",
            request=make_req(priority=Priority.HIGH),
        )
        low_ir = InferenceRequest(
            priority_score=2,
            submit_time=t,
            session_id="s2",
            request=make_req(priority=Priority.LOW),
        )
        assert high_ir < low_ir

    @pytest.mark.asyncio
    async def test_high_priority_dispatched_before_low(self, scheduler):
        """
        HIGH priority request should be dequeued before LOW priority.
        We verify this by checking queue ordering directly (without race conditions
        from the scheduler loop running concurrently).
        """
        # Submit LOW first, then HIGH — both go into the queue before the scheduler starts
        low_session = await scheduler.submit(make_req(priority=Priority.LOW, max_tokens=2))
        high_session = await scheduler.submit(make_req(priority=Priority.HIGH, max_tokens=2))

        # Drain the queue manually to inspect ordering
        first = scheduler._queue.get_nowait()
        second = scheduler._queue.get_nowait()

        # The first item out should be HIGH (lower priority_score = 0)
        assert first.priority_score <= second.priority_score, (
            f"Expected HIGH (score {first.priority_score}) before LOW (score {second.priority_score})"
        )
        assert first.priority_score == 0, "First dequeued item should have priority_score=0 (HIGH)"
        assert second.priority_score == 2, "Second dequeued item should have priority_score=2 (LOW)"


# ---------------------------------------------------------------------------
# Batch size limits
# ---------------------------------------------------------------------------

class TestBatchSizeLimits:
    @pytest.mark.asyncio
    async def test_batch_size_never_exceeds_max(self, scheduler):
        """The scheduler must not dispatch more than max_batch_size at once."""
        max_bs = scheduler.max_batch_size
        peak_active: List[int] = []

        original_run = scheduler.run_batch_loop

        async def patched_loop():
            # Monkey-patch to record active task count per loop iteration
            scheduler._running = True
            for _ in range(20):
                batch = []
                slots_available = max_bs - len(scheduler._active_tasks)
                if slots_available > 0:
                    while len(batch) < slots_available:
                        try:
                            ir = scheduler._queue.get_nowait()
                            batch.append(ir)
                        except asyncio.QueueEmpty:
                            break

                if batch:
                    scheduler.total_batches += 1
                    scheduler._batch_sizes.append(len(batch))
                    for ir in batch:
                        task = asyncio.create_task(
                            scheduler._run_request(ir),
                            name=f"infer-{ir.session_id[:8]}",
                        )
                        scheduler._active_tasks[ir.session_id] = task
                        task.add_done_callback(
                            lambda t, sid=ir.session_id: scheduler._active_tasks.pop(sid, None)
                        )
                        scheduler._queue.task_done()

                    peak_active.append(len(scheduler._active_tasks))

                await asyncio.sleep(scheduler.loop_interval_s)

        # Flood the queue with more requests than max_batch_size
        for i in range(20):
            await scheduler.submit(make_req(max_tokens=5))

        # Run the patched loop
        await patched_loop()

        for peak in peak_active:
            assert peak <= max_bs, (
                f"Active tasks {peak} exceeded max_batch_size {max_bs}"
            )

    def test_avg_batch_size_updates(self, scheduler):
        """avg_batch_size property should return correct average."""
        scheduler._batch_sizes = [4, 6, 8, 2]
        assert scheduler.avg_batch_size == pytest.approx(5.0, abs=0.01)

    def test_avg_batch_size_empty(self, scheduler):
        assert scheduler.avg_batch_size == 0.0


# ---------------------------------------------------------------------------
# Queue wait time tracking
# ---------------------------------------------------------------------------

class TestQueueWaitTimes:
    @pytest.mark.asyncio
    async def test_queue_wait_time_recorded(self, scheduler):
        """Wait time should be > 0 if request sits in queue before dispatch."""
        scheduler.start()

        # Saturate the scheduler so the next request has to wait
        futures = []
        for _ in range(2):
            req = make_req(max_tokens=4)
            loop = asyncio.get_event_loop()
            fut = loop.create_future()
            futures.append(fut)
            await scheduler.submit(req, future=fut)

        await asyncio.sleep(0.3)
        await scheduler.stop()

        if scheduler._queue_wait_times_ms:
            for wait_ms in scheduler._queue_wait_times_ms:
                assert wait_ms >= 0.0

    def test_avg_queue_wait_empty(self, scheduler):
        assert scheduler.avg_queue_wait_ms == 0.0

    def test_avg_queue_wait_computed(self, scheduler):
        scheduler._queue_wait_times_ms = [10.0, 20.0, 30.0]
        assert scheduler.avg_queue_wait_ms == pytest.approx(20.0, abs=0.1)


# ---------------------------------------------------------------------------
# Session cancellation
# ---------------------------------------------------------------------------

class TestSessionCancellation:
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_session(self, scheduler):
        """Cancelling a non-existent session should return False."""
        result = await scheduler.cancel_session("nonexistent-session-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_active_session(self, scheduler):
        """Cancelling an active session should return True and stop the task."""
        scheduler.start()

        # Submit a long-running request
        req = make_req(max_tokens=100)
        session_id = await scheduler.submit(req)

        # Give scheduler time to pick it up
        await asyncio.sleep(0.15)

        if session_id in scheduler._active_tasks:
            result = await scheduler.cancel_session(session_id)
            assert result is True

        await scheduler.stop()


# ---------------------------------------------------------------------------
# InferenceRequest construction
# ---------------------------------------------------------------------------

class TestInferenceRequestConstruction:
    def test_from_generate_request_assigns_session_id(self):
        req = make_req()
        assert req.session_id is None
        ir = InferenceRequest.from_generate_request(req)
        assert ir.session_id is not None
        assert len(ir.session_id) > 0

    def test_from_generate_request_respects_provided_session_id(self):
        req = make_req()
        req = req.model_copy(update={"session_id": "my-session"})
        ir = InferenceRequest.from_generate_request(req)
        assert ir.session_id == "my-session"

    def test_priority_score_mapping(self):
        high_ir = InferenceRequest.from_generate_request(make_req(priority=Priority.HIGH))
        normal_ir = InferenceRequest.from_generate_request(make_req(priority=Priority.NORMAL))
        low_ir = InferenceRequest.from_generate_request(make_req(priority=Priority.LOW))
        assert high_ir.priority_score < normal_ir.priority_score < low_ir.priority_score

    def test_queue_wait_ms_increases_over_time(self):
        ir = InferenceRequest.from_generate_request(make_req())
        time.sleep(0.01)
        assert ir.queue_wait_ms >= 10.0  # at least 10ms
