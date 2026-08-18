"""
System monitor + Prometheus metrics exporter.

Tracks rolling statistics for the inference engine and exports them
in Prometheus text exposition format.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

try:
    from prometheus_client import (
        CollectorRegistry,
        Gauge,
        Counter,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False


ROLLING_WINDOW = 100   # last N requests for percentile calculations
THROUGHPUT_WINDOW = 60  # seconds for rolling tok/s chart


# ---------------------------------------------------------------------------
# Per-request record
# ---------------------------------------------------------------------------

@dataclass
class RequestRecord:
    session_id: str
    model: str
    ttft_ms: float
    tokens: int
    duration_s: float
    tokens_per_sec: float
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class SystemMonitor:
    def __init__(self) -> None:
        self._start_time = time.time()
        self._records: Deque[RequestRecord] = deque(maxlen=ROLLING_WINDOW)

        # Rolling throughput: (timestamp, tokens) pairs
        self._throughput_events: Deque[tuple] = deque(maxlen=2000)

        # External references set after construction
        self._engine: Optional[Any] = None
        self._scheduler: Optional[Any] = None

        # Session log (last 10 completed)
        self._session_log: Deque[Dict] = deque(maxlen=10)

        # Totals
        self._total_requests: int = 0
        self._total_tokens: int = 0

        # Simulated PCIe bandwidth (GB/s)
        self._pcie_bw_gb_s: float = 0.0

        if _PROM_AVAILABLE:
            self._registry = CollectorRegistry()
            self._g_vram_used = Gauge(
                "llm_vram_used_mb", "VRAM used by KV cache (MB)",
                registry=self._registry
            )
            self._g_tokens_per_sec = Gauge(
                "llm_tokens_per_sec", "Token generation throughput (tokens/s)",
                registry=self._registry
            )
            self._g_ttft_p50 = Gauge(
                "llm_ttft_p50_ms", "TTFT p50 latency (ms)",
                registry=self._registry
            )
            self._g_ttft_p95 = Gauge(
                "llm_ttft_p95_ms", "TTFT p95 latency (ms)",
                registry=self._registry
            )
            self._g_ttft_p99 = Gauge(
                "llm_ttft_p99_ms", "TTFT p99 latency (ms)",
                registry=self._registry
            )
            self._g_active_sessions = Gauge(
                "llm_active_sessions", "Currently active inference sessions",
                registry=self._registry
            )
            self._g_queue_depth = Gauge(
                "llm_queue_depth", "Pending requests in scheduler queue",
                registry=self._registry
            )
            self._g_batch_size_avg = Gauge(
                "llm_batch_size_avg", "Average batch size over recent batches",
                registry=self._registry
            )
            self._g_pages_free = Gauge(
                "llm_kv_pages_free", "Free KV cache pages",
                registry=self._registry
            )
            self._c_total_tokens = Counter(
                "llm_total_tokens_generated", "Total tokens generated since start",
                registry=self._registry
            )
        else:
            self._registry = None

    def set_engine(self, engine: Any) -> None:
        self._engine = engine

    def set_scheduler(self, scheduler: Any) -> None:
        self._scheduler = scheduler

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_request(
        self,
        ttft_ms: float,
        tokens: int,
        duration_s: float,
        model: str,
        session_id: str,
    ) -> None:
        tps = tokens / max(duration_s, 0.001)
        rec = RequestRecord(
            session_id=session_id,
            model=model,
            ttft_ms=ttft_ms,
            tokens=tokens,
            duration_s=duration_s,
            tokens_per_sec=tps,
        )
        self._records.append(rec)
        self._total_requests += 1
        self._total_tokens += tokens
        self._throughput_events.append((time.time(), tokens))

        # Add to session log
        self._session_log.append({
            "session_id": session_id,
            "model": model,
            "tokens": tokens,
            "ttft_ms": round(ttft_ms, 1),
            "duration_ms": round(duration_s * 1000, 1),
            "tokens_per_sec": round(tps, 1),
            "timestamp": rec.timestamp,
        })

        # Update Prometheus counters
        if _PROM_AVAILABLE and self._registry:
            self._c_total_tokens.inc(tokens)

    # ------------------------------------------------------------------
    # Computed stats
    # ------------------------------------------------------------------

    def _percentile(self, data: List[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100.0
        lo = math.floor(k)
        hi = math.ceil(k)
        if lo == hi:
            return sorted_data[lo]
        return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)

    @property
    def ttft_samples(self) -> List[float]:
        return [r.ttft_ms for r in self._records]

    @property
    def ttft_p50(self) -> float:
        return self._percentile(self.ttft_samples, 50)

    @property
    def ttft_p95(self) -> float:
        return self._percentile(self.ttft_samples, 95)

    @property
    def ttft_p99(self) -> float:
        return self._percentile(self.ttft_samples, 99)

    @property
    def tokens_per_sec(self) -> float:
        """Rolling tok/s over last 10 seconds."""
        now = time.time()
        cutoff = now - 10.0
        recent = [(ts, tok) for ts, tok in self._throughput_events if ts >= cutoff]
        if not recent:
            return 0.0
        total_toks = sum(tok for _, tok in recent)
        window = max(now - recent[0][0], 0.1)
        return total_toks / window

    @property
    def throughput_history(self) -> List[Dict]:
        """Per-second token counts for last 60 seconds (for chart)."""
        now = time.time()
        buckets: Dict[int, int] = {}
        for ts, tok in self._throughput_events:
            age = int(now - ts)
            if age <= 60:
                buckets[age] = buckets.get(age, 0) + tok
        result = []
        for sec in range(60, -1, -1):
            result.append({"age_s": sec, "tokens": buckets.get(sec, 0)})
        return result

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        engine_stats = self._engine.get_stats() if self._engine else {}
        sched_status = self._scheduler.get_status() if self._scheduler else {}

        return {
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "ttft_p50_ms": round(self.ttft_p50, 1),
            "ttft_p95_ms": round(self.ttft_p95, 1),
            "ttft_p99_ms": round(self.ttft_p99, 1),
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "vram_used_mb": engine_stats.get("vram_used_mb", 0.0),
            "vram_total_mb": engine_stats.get("vram_total_mb", 80000),
            "pages_allocated": engine_stats.get("pages_allocated", 0),
            "pages_free": engine_stats.get("pages_free", 0),
            "active_sessions": sched_status.get("active_sessions", 0),
            "queue_depth": sched_status.get("queue_depth", 0),
            "avg_batch_size": sched_status.get("avg_batch_size", 0.0),
            "uptime_seconds": round(self.uptime_seconds, 1),
        }

    def get_session_log(self) -> List[Dict]:
        return list(reversed(list(self._session_log)))

    # ------------------------------------------------------------------
    # Prometheus export
    # ------------------------------------------------------------------

    def update_prometheus_gauges(self) -> None:
        if not _PROM_AVAILABLE or not self._registry:
            return
        summary = self.get_summary()
        self._g_vram_used.set(summary["vram_used_mb"])
        self._g_tokens_per_sec.set(summary["tokens_per_sec"])
        self._g_ttft_p50.set(summary["ttft_p50_ms"])
        self._g_ttft_p95.set(summary["ttft_p95_ms"])
        self._g_ttft_p99.set(summary["ttft_p99_ms"])
        self._g_active_sessions.set(summary["active_sessions"])
        self._g_queue_depth.set(summary["queue_depth"])
        self._g_batch_size_avg.set(summary["avg_batch_size"])
        if self._engine:
            self._g_pages_free.set(self._engine.pages_free)

    def get_prometheus_output(self) -> str:
        if not _PROM_AVAILABLE or not self._registry:
            return self._fallback_prometheus()
        self.update_prometheus_gauges()
        return generate_latest(self._registry).decode("utf-8")

    def _fallback_prometheus(self) -> str:
        """Minimal Prometheus text format if library unavailable."""
        s = self.get_summary()
        lines = [
            f"# HELP llm_vram_used_mb VRAM used by KV cache (MB)",
            f"# TYPE llm_vram_used_mb gauge",
            f"llm_vram_used_mb {s['vram_used_mb']}",
            f"# HELP llm_tokens_per_sec Token generation throughput",
            f"# TYPE llm_tokens_per_sec gauge",
            f"llm_tokens_per_sec {s['tokens_per_sec']}",
            f"# HELP llm_ttft_p50_ms TTFT p50 latency ms",
            f"# TYPE llm_ttft_p50_ms gauge",
            f"llm_ttft_p50_ms {s['ttft_p50_ms']}",
            f"# HELP llm_active_sessions Active inference sessions",
            f"# TYPE llm_active_sessions gauge",
            f"llm_active_sessions {s['active_sessions']}",
            f"# HELP llm_queue_depth Pending requests in queue",
            f"# TYPE llm_queue_depth gauge",
            f"llm_queue_depth {s['queue_depth']}",
        ]
        return "\n".join(lines) + "\n"

    CONTENT_TYPE = CONTENT_TYPE_LATEST if _PROM_AVAILABLE else "text/plain; version=0.0.4"
