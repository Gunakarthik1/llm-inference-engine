"""
Dynamic batch scheduler with priority queue.

Architecture:
  - Three logical priority lanes: HIGH=0, NORMAL=1, LOW=2
  - asyncio.PriorityQueue holds (priority_score, submit_time, InferenceRequest)
  - The batch loop runs every ~50 ms, drains up to max_batch_size items,
    dispatches them concurrently to the InferenceEngine.
  - Continuous batching: each request runs independently as an asyncio task;
    the loop doesn't wait for all tasks to complete before pulling more work.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

from engine.models import GenerateRequest, Priority, PRIORITY_SCORES


# ---------------------------------------------------------------------------
# Internal request wrapper
# ---------------------------------------------------------------------------

@dataclass(order=True)
class InferenceRequest:
    priority_score: int
    submit_time: float
    session_id: str = field(compare=False)
    request: GenerateRequest = field(compare=False)
    result_callback: Optional[Callable] = field(default=None, compare=False)
    # Future that callers can await to get the full result (REST path)
    future: Optional[asyncio.Future] = field(default=None, compare=False)

    @classmethod
    def from_generate_request(
        cls,
        req: GenerateRequest,
        callback: Optional[Callable] = None,
        future: Optional[asyncio.Future] = None,
    ) -> "InferenceRequest":
        session_id = req.session_id or str(uuid.uuid4())
        req = req.model_copy(update={"session_id": session_id})
        return cls(
            priority_score=PRIORITY_SCORES[req.priority],
            submit_time=time.monotonic(),
            session_id=session_id,
            request=req,
            result_callback=callback,
            future=future,
        )

    @property
    def queue_wait_ms(self) -> float:
        return (time.monotonic() - self.submit_time) * 1000.0


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class DynamicBatchScheduler:
    def __init__(
        self,
        inference_engine: Any,
        monitor: Any,
        max_batch_size: int = 8,
        loop_interval_s: float = 0.05,
    ) -> None:
        self._engine = inference_engine
        self._monitor = monitor
        self.max_batch_size = max_batch_size
        self.loop_interval_s = loop_interval_s

        # Priority queue: smallest score = highest priority
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()

        # Active async tasks (session_id → task)
        self._active_tasks: Dict[str, asyncio.Task] = {}

        # Stats
        self.total_batches: int = 0
        self._batch_sizes: List[int] = []
        self._queue_wait_times_ms: List[float] = []

        self._running: bool = False
        self._loop_task: Optional[asyncio.Task] = None

        # Per-priority counters
        self._priority_counts: Dict[str, int] = {"high": 0, "normal": 0, "low": 0}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(
        self,
        request: GenerateRequest,
        callback: Optional[Callable] = None,
        future: Optional[asyncio.Future] = None,
    ) -> str:
        """Enqueue a request. Returns the session_id."""
        ir = InferenceRequest.from_generate_request(request, callback, future)
        await self._queue.put(ir)
        self._priority_counts[request.priority.value] += 1
        return ir.session_id

    async def cancel_session(self, session_id: str) -> bool:
        """Cancel an in-flight session if it exists."""
        task = self._active_tasks.get(session_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    def start(self) -> None:
        self._running = True
        self._loop_task = asyncio.create_task(self.run_batch_loop(), name="scheduler-loop")

    async def stop(self) -> None:
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Batch loop
    # ------------------------------------------------------------------

    async def run_batch_loop(self) -> None:
        """
        Continuously:
          1. Drain up to max_batch_size requests from the queue
          2. Launch each as an independent asyncio task (continuous batching)
          3. Sleep for loop_interval_s
        """
        while self._running:
            batch: List[InferenceRequest] = []

            # Drain up to max_batch_size without blocking
            slots_available = self.max_batch_size - len(self._active_tasks)
            if slots_available > 0:
                while len(batch) < slots_available:
                    try:
                        ir = self._queue.get_nowait()
                        batch.append(ir)
                    except asyncio.QueueEmpty:
                        break

            if batch:
                self.total_batches += 1
                self._batch_sizes.append(len(batch))
                if len(self._batch_sizes) > 200:
                    self._batch_sizes = self._batch_sizes[-200:]

                for ir in batch:
                    wait_ms = ir.queue_wait_ms
                    self._queue_wait_times_ms.append(wait_ms)
                    if len(self._queue_wait_times_ms) > 200:
                        self._queue_wait_times_ms = self._queue_wait_times_ms[-200:]

                    task = asyncio.create_task(
                        self._run_request(ir),
                        name=f"infer-{ir.session_id[:8]}",
                    )
                    self._active_tasks[ir.session_id] = task

                    # Clean up task dict when done
                    task.add_done_callback(
                        lambda t, sid=ir.session_id: self._active_tasks.pop(sid, None)
                    )

                    self._queue.task_done()

            await asyncio.sleep(self.loop_interval_s)

    # ------------------------------------------------------------------
    # Per-request runner
    # ------------------------------------------------------------------

    async def _run_request(self, ir: InferenceRequest) -> None:
        req = ir.request
        session_id = ir.session_id
        tokens: List[str] = []
        ttft_ms: float = 0.0
        start_time = time.monotonic()

        try:
            # Allocate KV-cache pages
            await self._engine.allocate_pages(session_id, len(req.prompt.split()))

            token_count = 0
            async for token_text, latency_ms in self._engine.generate_tokens(
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                model=req.model.value,
                session_id=session_id,
            ):
                if token_count == 0:
                    ttft_ms = latency_ms

                tokens.append(token_text)
                token_count += 1

                # Stream to WebSocket / callback
                if ir.result_callback is not None:
                    elapsed_ms = (time.monotonic() - start_time) * 1000.0
                    try:
                        await ir.result_callback(
                            token_text,
                            ttft_ms if token_count == 1 else None,
                            False,
                            session_id,
                            token_count,
                            elapsed_ms,
                        )
                    except Exception:
                        # Client disconnected
                        break

            total_ms = (time.monotonic() - start_time) * 1000.0
            elapsed_ms = total_ms

            # Send final marker
            if ir.result_callback is not None:
                try:
                    await ir.result_callback(
                        "",
                        None,
                        True,
                        session_id,
                        token_count,
                        elapsed_ms,
                    )
                except Exception:
                    pass

            # Resolve REST future
            if ir.future is not None and not ir.future.done():
                full_text = "".join(tokens)
                tps = token_count / max(total_ms / 1000.0, 0.001)
                ir.future.set_result({
                    "session_id": session_id,
                    "text": full_text,
                    "model": req.model.value,
                    "tokens_generated": token_count,
                    "ttft_ms": ttft_ms,
                    "total_time_ms": total_ms,
                    "tokens_per_sec": tps,
                })

            # Record metrics
            if token_count > 0:
                tps = token_count / max(total_ms / 1000.0, 0.001)
                self._monitor.record_request(
                    ttft_ms=ttft_ms,
                    tokens=token_count,
                    duration_s=total_ms / 1000.0,
                    model=req.model.value,
                    session_id=session_id,
                )

        except asyncio.CancelledError:
            if ir.future and not ir.future.done():
                ir.future.cancel()
        except Exception as exc:
            if ir.future and not ir.future.done():
                ir.future.set_exception(exc)
        finally:
            await self._engine.free_pages(session_id)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def active_sessions(self) -> int:
        return len(self._active_tasks)

    @property
    def avg_batch_size(self) -> float:
        if not self._batch_sizes:
            return 0.0
        return sum(self._batch_sizes) / len(self._batch_sizes)

    @property
    def avg_queue_wait_ms(self) -> float:
        if not self._queue_wait_times_ms:
            return 0.0
        return sum(self._queue_wait_times_ms) / len(self._queue_wait_times_ms)

    def get_priority_counts(self) -> Dict[str, int]:
        # Approximate from queue snapshot — not precise but good enough for status
        high = normal = low = 0
        # Walk active tasks by checking priority in names (we don't store it directly)
        # Return historical submit counts instead
        return dict(self._priority_counts)

    def get_status(self) -> dict:
        return {
            "queue_depth": self.queue_depth,
            "active_sessions": self.active_sessions,
            "total_batches": self.total_batches,
            "avg_batch_size": round(self.avg_batch_size, 2),
            "avg_queue_wait_ms": round(self.avg_queue_wait_ms, 2),
            "priority_counts": self.get_priority_counts(),
        }
