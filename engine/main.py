"""
FastAPI application — REST + WebSocket inference gateway.

Endpoints:
  POST   /api/generate              — non-streaming full completion
  WS     /ws/generate               — streaming token-by-token
  GET    /api/queue/status          — scheduler queue info
  GET    /api/metrics/summary       — throughput/latency stats
  GET    /metrics                   — Prometheus exposition
  GET    /api/health                — liveness probe
  DELETE /api/sessions/{session_id} — cancel active session
  GET    /api/sessions/log          — last 10 completed sessions
  GET    /api/throughput/history    — 60-second chart data
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response

from engine.inference import InferenceEngine
from engine.models import (
    GenerateRequest,
    GenerateResponse,
    MetricsSummary,
    QueueStatus,
)
from engine.monitor import SystemMonitor
from engine.scheduler import DynamicBatchScheduler


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

engine: InferenceEngine
scheduler: DynamicBatchScheduler
monitor: SystemMonitor
_metrics_task: asyncio.Task


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, scheduler, monitor, _metrics_task

    monitor = SystemMonitor()
    engine = InferenceEngine()
    scheduler = DynamicBatchScheduler(
        inference_engine=engine,
        monitor=monitor,
        max_batch_size=8,
        loop_interval_s=0.05,
    )

    monitor.set_engine(engine)
    monitor.set_scheduler(scheduler)
    scheduler.start()

    # Periodic Prometheus gauge update
    async def _update_metrics_loop():
        while True:
            monitor.update_prometheus_gauges()
            await asyncio.sleep(5)

    _metrics_task = asyncio.create_task(_update_metrics_loop(), name="metrics-updater")

    yield

    # Shutdown
    _metrics_task.cancel()
    await scheduler.stop()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM Inference Engine",
    version="1.0.0",
    description="High-Throughput Distributed LLM Inference with PagedAttention",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "uptime_s": round(monitor.uptime_seconds, 1),
        "active_sessions": scheduler.active_sessions,
        "queue_depth": scheduler.queue_depth,
    }


# ---------------------------------------------------------------------------
# WebSocket streaming endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/generate")
async def ws_generate(websocket: WebSocket):
    await websocket.accept()

    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
    except asyncio.TimeoutError:
        await websocket.close(code=1008)
        return

    try:
        data = json.loads(raw)
        req = GenerateRequest(**data)
    except Exception as exc:
        await websocket.send_text(json.dumps({"error": str(exc)}))
        await websocket.close(code=1003)
        return

    # Build streaming callback
    async def token_callback(
        token: str,
        ttft_ms: float | None,
        is_final: bool,
        session_id: str,
        tokens_generated: int,
        elapsed_ms: float,
    ):
        payload = {
            "token": token,
            "ttft_ms": ttft_ms,
            "is_final": is_final,
            "session_id": session_id,
            "tokens_generated": tokens_generated,
            "elapsed_ms": elapsed_ms,
        }
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception:
            raise  # scheduler catches and aborts

    session_id = await scheduler.submit(req, callback=token_callback)

    # Keep connection alive until generation completes or client disconnects
    try:
        while True:
            # Wait for client close or timeout; generation drives sends via callback
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                # Ping-pong keepalive
                if msg == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                await scheduler.cancel_session(session_id)
                return

            # Check if session has completed
            if session_id not in scheduler._active_tasks:
                break

    except WebSocketDisconnect:
        await scheduler.cancel_session(session_id)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# REST: full completion
# ---------------------------------------------------------------------------

@app.post("/api/generate", response_model=GenerateResponse)
async def rest_generate(req: GenerateRequest):
    loop = asyncio.get_event_loop()
    future: asyncio.Future = loop.create_future()

    await scheduler.submit(req, future=future)

    try:
        result = await asyncio.wait_for(future, timeout=120.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Generation timed out")
    except asyncio.CancelledError:
        raise HTTPException(status_code=499, detail="Session cancelled")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return GenerateResponse(**result)


# ---------------------------------------------------------------------------
# Queue status
# ---------------------------------------------------------------------------

@app.get("/api/queue/status", response_model=QueueStatus)
async def queue_status():
    status = scheduler.get_status()
    pc = status["priority_counts"]
    return QueueStatus(
        queue_depth=status["queue_depth"],
        active_sessions=status["active_sessions"],
        high_priority=pc.get("high", 0),
        normal_priority=pc.get("normal", 0),
        low_priority=pc.get("low", 0),
        total_batches_processed=status["total_batches"],
        avg_batch_size=status["avg_batch_size"],
        avg_queue_wait_ms=status["avg_queue_wait_ms"],
    )


# ---------------------------------------------------------------------------
# Metrics summary
# ---------------------------------------------------------------------------

@app.get("/api/metrics/summary", response_model=MetricsSummary)
async def metrics_summary():
    s = monitor.get_summary()
    return MetricsSummary(**s)


# ---------------------------------------------------------------------------
# Prometheus exposition
# ---------------------------------------------------------------------------

@app.get("/metrics")
async def prometheus_metrics():
    output = monitor.get_prometheus_output()
    return Response(
        content=output,
        media_type=monitor.CONTENT_TYPE,
    )


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@app.delete("/api/sessions/{session_id}")
async def cancel_session(session_id: str):
    cancelled = await scheduler.cancel_session(session_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail="Session not found or already complete")
    return {"session_id": session_id, "status": "cancelled"}


@app.get("/api/sessions/log")
async def session_log():
    return {"sessions": monitor.get_session_log()}


# ---------------------------------------------------------------------------
# Throughput history (for live chart)
# ---------------------------------------------------------------------------

@app.get("/api/throughput/history")
async def throughput_history():
    return {"history": monitor.throughput_history}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "engine.main:app",
        host="0.0.0.0",
        port=8001,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )
