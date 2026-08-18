# High-Throughput Distributed LLM Inference Engine

A systems-engineering focused LLM serving layer that simulates production-grade inference infrastructure: **vLLM-style PagedAttention KV-cache management**, **dynamic request batching**, **WebSocket token streaming**, **priority queue scheduling**, and **Prometheus metrics**.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    FastAPI (port 8001)                   │
│  POST /api/generate    WS /ws/generate    GET /metrics   │
└───────────────────────────┬──────────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │   DynamicBatchScheduler    │
              │  priority queue (H/N/L)    │
              │  max_batch_size = 8        │
              │  loop every 50ms           │
              └─────────────┬──────────────┘
                            │ dispatch batch
              ┌─────────────▼──────────────┐
              │      InferenceEngine       │
              │  PagedAttention KV-cache   │
              │  PAGE_SIZE = 16 tokens     │
              │  VRAM budget = 80 GB H100  │
              └─────────────┬──────────────┘
                            │ async token stream
              ┌─────────────▼──────────────┐
              │       SystemMonitor        │
              │  rolling percentiles       │
              │  Prometheus gauges         │
              └────────────────────────────┘
```

## Components

### `engine/inference.py` — PagedAttention Simulation
- Divides the KV cache into 16-token pages (blocks), mirroring vLLM's architecture
- `allocate_pages(session_id, prompt_len)` — reserves pages from the free pool
- `extend_pages(session_id, n)` — lazily extends during decode
- `free_pages(session_id)` — returns pages to pool on completion
- Raises `RuntimeError` if VRAM budget exceeded
- Token generation: TTFT 80-200ms (model-dependent), inter-token 15-40ms

### `engine/scheduler.py` — Dynamic Batch Scheduler
- Three priority lanes: HIGH (score 0) > NORMAL (score 1) > LOW (score 2)
- `asyncio.PriorityQueue` ensures correct ordering
- Batch loop runs every 50ms, drains up to 8 requests
- **Continuous batching**: each request runs as an independent asyncio task — new requests slot in as old ones finish (no head-of-line blocking)
- Tracks: total batches, avg batch size, queue wait times

### `engine/monitor.py` — System Monitor + Prometheus
- Rolling window of last 100 requests for TTFT percentiles (p50/p95/p99)
- 60-second throughput history for the live chart
- Prometheus gauges: `llm_vram_used_mb`, `llm_tokens_per_sec`, `llm_ttft_p50_ms`, `llm_ttft_p95_ms`, `llm_ttft_p99_ms`, `llm_active_sessions`, `llm_queue_depth`, `llm_batch_size_avg`, `llm_kv_pages_free`

### `engine/main.py` — FastAPI Gateway
- WebSocket `/ws/generate` — streams tokens as JSON chunks in real time
- `POST /api/generate` — non-streaming, awaits future resolved by scheduler
- Background lifespan: starts scheduler loop, periodic Prometheus gauge update
- Session cancellation: `DELETE /api/sessions/{session_id}`

### `frontend/index.html` — Monitoring Dashboard
- Vanilla JS, no frameworks, single HTML file
- Left 60%: WebSocket chat terminal with live token streaming
- Right 40%: "tape readout" metrics panel
- Canvas line chart (hand-drawn aesthetic, slight stroke noise)
- Live TTFT and tok/s indicators
- Model selector (Mistral-7B / Llama-3-8B), priority tabs, max-tokens input

---

## Quickstart — Local (Python)

```bash
cd llm-inference-engine

# Create virtualenv
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r engine/requirements.txt

# Start the engine
python -m uvicorn engine.main:app --host 0.0.0.0 --port 8001 --reload

# Open the dashboard
open frontend/index.html         # macOS
# or just open the file in your browser
```

### API examples

```bash
# Non-streaming completion
curl -X POST http://localhost:8001/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain attention mechanisms","model":"Mistral-7B","priority":"high","max_tokens":128}'

# Queue status
curl http://localhost:8001/api/queue/status

# Metrics summary
curl http://localhost:8001/api/metrics/summary

# Prometheus exposition
curl http://localhost:8001/metrics

# Health check
curl http://localhost:8001/api/health

# Cancel a session
curl -X DELETE http://localhost:8001/api/sessions/<session_id>
```

---

## Quickstart — Docker Compose

```bash
cd llm-inference-engine
docker compose up --build

# Services:
#   Engine:     http://localhost:8001
#   Prometheus: http://localhost:9090
#   Grafana:    http://localhost:3000  (admin / admin)
```

Prometheus is pre-configured to scrape the engine's `/metrics` endpoint every 5 seconds.

---

## Running Tests

```bash
# From project root with virtualenv active
pip install -r engine/requirements.txt
pytest tests/ -v
```

Test coverage:
- `tests/test_inference.py` — PagedAttention page math, VRAM budget enforcement, allocation/deallocation, token generation count and latency ordering, stats
- `tests/test_scheduler.py` — Priority ordering, batch size limits, queue wait time tracking, session cancellation, InferenceRequest construction

---

## Prometheus Metrics Reference

| Metric | Type | Description |
|---|---|---|
| `llm_vram_used_mb` | Gauge | KV-cache VRAM consumed (MB) |
| `llm_tokens_per_sec` | Gauge | Rolling 10s token throughput |
| `llm_ttft_p50_ms` | Gauge | TTFT p50 over last 100 requests |
| `llm_ttft_p95_ms` | Gauge | TTFT p95 over last 100 requests |
| `llm_ttft_p99_ms` | Gauge | TTFT p99 over last 100 requests |
| `llm_active_sessions` | Gauge | Concurrently running sessions |
| `llm_queue_depth` | Gauge | Pending requests in scheduler queue |
| `llm_batch_size_avg` | Gauge | Average batch size (recent batches) |
| `llm_kv_pages_free` | Gauge | Available KV-cache pages |
| `llm_total_tokens_generated` | Counter | Cumulative tokens since start |

---

## Key Design Decisions

**PagedAttention**: KV-cache pages are allocated from a fixed pool (70% of H100 VRAM). Each page holds 16 tokens. Fragmentation is bounded — unused slots within a page are the only waste. Pages are returned to the pool immediately on session completion.

**Continuous Batching**: The scheduler does not wait for an entire batch to finish before scheduling the next. Each request runs as an independent `asyncio.Task`. As tasks complete, slots open for new requests. This eliminates head-of-line blocking from slow (long-output) requests.

**Priority Queue**: `asyncio.PriorityQueue` with numeric priority scores (HIGH=0, NORMAL=1, LOW=2) ensures that high-priority requests are dispatched first, even if they arrive after lower-priority ones.

**WebSocket Streaming**: Each WebSocket connection gets a per-request callback. The scheduler calls the callback for every generated token. The connection stays open until generation completes or the client disconnects.
