from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import time


class ModelName(str, Enum):
    MISTRAL_7B = "Mistral-7B"
    LLAMA_3_8B = "Llama-3-8B"


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


PRIORITY_SCORES = {
    Priority.HIGH: 0,
    Priority.NORMAL: 1,
    Priority.LOW: 2,
}


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8192)
    model: ModelName = ModelName.MISTRAL_7B
    priority: Priority = Priority.NORMAL
    max_tokens: int = Field(default=256, ge=1, le=2048)
    session_id: Optional[str] = None


class GenerateResponse(BaseModel):
    session_id: str
    text: str
    model: str
    tokens_generated: int
    ttft_ms: float
    total_time_ms: float
    tokens_per_sec: float


class TokenChunk(BaseModel):
    token: str
    ttft_ms: Optional[float] = None
    is_final: bool = False
    session_id: str
    tokens_generated: int = 0
    elapsed_ms: float = 0.0


class QueueStatus(BaseModel):
    queue_depth: int
    active_sessions: int
    high_priority: int
    normal_priority: int
    low_priority: int
    total_batches_processed: int
    avg_batch_size: float
    avg_queue_wait_ms: float


class MetricsSummary(BaseModel):
    tokens_per_sec: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    total_requests: int
    total_tokens: int
    vram_used_mb: float
    vram_total_mb: float
    pages_allocated: int
    pages_free: int
    active_sessions: int
    uptime_seconds: float


class SessionInfo(BaseModel):
    session_id: str
    model: str
    prompt_preview: str
    tokens_generated: int
    ttft_ms: float
    total_time_ms: float
    tokens_per_sec: float
    status: str
    started_at: float = Field(default_factory=time.time)
