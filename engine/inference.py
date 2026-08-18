"""
Inference engine with PagedAttention-style KV-cache simulation.

PagedAttention divides the KV cache into fixed-size pages (blocks).
Each page holds KEY and VALUE tensors for PAGE_SIZE tokens.
Sessions are allocated pages on arrival; pages are freed on completion.
This prevents memory fragmentation and allows fine-grained reclamation.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, List, Optional, Tuple

PAGE_SIZE = 16          # tokens per KV-cache page
VRAM_TOTAL_MB = 80_000  # H100 80 GB
BYTES_PER_TOKEN_KV = 4096 * 2  # fp16 KV for a 7B-class model (simplified)
BYTES_PER_PAGE = PAGE_SIZE * BYTES_PER_TOKEN_KV
MB = 1024 * 1024


# ---------------------------------------------------------------------------
# Token vocabulary — realistic-looking text continuations
# ---------------------------------------------------------------------------
_VOCAB_CHUNKS: List[str] = [
    " the", " a", " is", " are", " was", " were", " be", " been",
    " have", " has", " had", " do", " does", " did", " will", " would",
    " can", " could", " should", " may", " might", " shall",
    " I", " you", " he", " she", " it", " we", " they",
    " this", " that", " these", " those", " which", " who", " what",
    " when", " where", " why", " how", " all", " each", " every",
    " more", " most", " other", " some", " such", " no", " not",
    " only", " same", " so", " than", " too", " very",
    " system", " model", " data", " function", " class", " object",
    " method", " return", " value", " type", " error", " result",
    " input", " output", " token", " layer", " attention", " weight",
    " gradient", " loss", " batch", " epoch", " training", " inference",
    " transformer", " embedding", " vector", " matrix", " tensor",
    " memory", " cache", " block", " page", " allocation", " sequence",
    " generation", " prediction", " probability", " distribution",
    " language", " neural", " network", " parameter", " dimension",
    " context", " prompt", " response", " query", " key",
    " according", " however", " therefore", " furthermore", " additionally",
    " specifically", " generally", " particularly", " significantly",
    " implementation", " configuration", " optimization", " performance",
    " efficient", " effective", " robust", " scalable", " distributed",
    ".\n", ".\n\n", ", ", ". ", "; ", " — ", ":\n", ".\n",
]

_SENTENCE_STARTERS: List[str] = [
    "The ", "A ", "This ", "In ", "When ", "For ", "By ",
    "To ", "With ", "As ", "At ", "On ", "From ", "Through ",
]


def _make_token(rng: random.Random) -> str:
    """Return a single realistic-looking token."""
    return rng.choice(_VOCAB_CHUNKS)


# ---------------------------------------------------------------------------
# Page + KV-cache structures
# ---------------------------------------------------------------------------

@dataclass
class KVPage:
    page_id: int
    size_mb: float = field(init=False)

    def __post_init__(self) -> None:
        self.size_mb = BYTES_PER_PAGE / MB


@dataclass
class SessionKVState:
    session_id: str
    pages: List[KVPage] = field(default_factory=list)
    tokens_prefilled: int = 0
    tokens_decoded: int = 0

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def vram_mb(self) -> float:
        return sum(p.size_mb for p in self.pages)


# ---------------------------------------------------------------------------
# Main inference engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    def __init__(self) -> None:
        # Physical page pool
        self._total_pages = self._compute_total_pages()
        self._free_page_ids: List[int] = list(range(self._total_pages))
        self._kv_cache: Dict[str, SessionKVState] = {}

        # Stats
        self.pages_allocated: int = 0
        self.active_sequences: int = 0
        self._lock = asyncio.Lock()

        # Per-model speed knobs
        self._model_config = {
            "Mistral-7B":  {"ttft_lo": 80,  "ttft_hi": 160, "inter_lo": 15, "inter_hi": 30},
            "Llama-3-8B":  {"ttft_lo": 120, "ttft_hi": 200, "inter_lo": 22, "inter_hi": 40},
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vram_total_mb(self) -> float:
        return VRAM_TOTAL_MB

    @property
    def vram_used_mb(self) -> float:
        return sum(s.vram_mb for s in self._kv_cache.values())

    @property
    def pages_free(self) -> int:
        return len(self._free_page_ids)

    @property
    def pages_in_use(self) -> int:
        return self._total_pages - self.pages_free

    # ------------------------------------------------------------------
    # Page management
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_total_pages() -> int:
        """How many pages fit in available VRAM (reserve 30% for weights)."""
        kv_budget_mb = VRAM_TOTAL_MB * 0.70
        return int((kv_budget_mb * MB) / BYTES_PER_PAGE)

    def _pages_needed(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / PAGE_SIZE)

    async def allocate_pages(self, session_id: str, prompt_len: int) -> SessionKVState:
        """
        Allocate KV-cache pages for a new session.
        Raises RuntimeError if VRAM budget is exceeded.
        """
        pages_needed = self._pages_needed(prompt_len)

        async with self._lock:
            if len(self._free_page_ids) < pages_needed:
                raise RuntimeError(
                    f"Out of VRAM: need {pages_needed} pages but only "
                    f"{len(self._free_page_ids)} free "
                    f"(prompt_len={prompt_len}, page_size={PAGE_SIZE})"
                )

            allocated: List[KVPage] = []
            for _ in range(pages_needed):
                pid = self._free_page_ids.pop()
                allocated.append(KVPage(page_id=pid))

            state = SessionKVState(
                session_id=session_id,
                pages=allocated,
                tokens_prefilled=prompt_len,
            )
            self._kv_cache[session_id] = state
            self.pages_allocated += pages_needed
            self.active_sequences += 1

        return state

    async def extend_pages(self, session_id: str, new_tokens: int) -> None:
        """
        During decoding, extend page allocation if current pages are full.
        This mirrors vLLM's lazy page extension.
        """
        async with self._lock:
            state = self._kv_cache.get(session_id)
            if state is None:
                return

            total_tokens = state.tokens_prefilled + state.tokens_decoded + new_tokens
            pages_needed = self._pages_needed(total_tokens)
            extra = pages_needed - state.total_pages

            for _ in range(extra):
                if not self._free_page_ids:
                    break  # best-effort; in production would evict
                pid = self._free_page_ids.pop()
                state.pages.append(KVPage(page_id=pid))
                self.pages_allocated += 1

            state.tokens_decoded += new_tokens

    async def free_pages(self, session_id: str) -> None:
        """Return all pages for a session to the free pool."""
        async with self._lock:
            state = self._kv_cache.pop(session_id, None)
            if state is None:
                return
            for page in state.pages:
                self._free_page_ids.append(page.page_id)
            self.active_sequences = max(0, self.active_sequences - 1)

    # ------------------------------------------------------------------
    # Token generation
    # ------------------------------------------------------------------

    async def generate_tokens(
        self,
        prompt: str,
        max_tokens: int,
        model: str,
        session_id: str,
    ) -> AsyncGenerator[Tuple[str, float], None]:
        """
        Async generator that yields (token_text, latency_ms) pairs.

        First token simulates TTFT (prefill phase).
        Subsequent tokens simulate the decode phase with lower latency.
        """
        cfg = self._model_config.get(model, self._model_config["Mistral-7B"])
        rng = random.Random(hash(prompt + session_id) & 0xFFFFFFFF)

        # ---- Prefill / TTFT ----
        ttft_ms = rng.uniform(cfg["ttft_lo"], cfg["ttft_hi"])
        await asyncio.sleep(ttft_ms / 1000.0)

        first_token = rng.choice(_SENTENCE_STARTERS) if rng.random() > 0.3 else _make_token(rng)
        yield (first_token, ttft_ms)

        # ---- Decode phase ----
        for idx in range(1, max_tokens):
            inter_ms = rng.uniform(cfg["inter_lo"], cfg["inter_hi"])
            await asyncio.sleep(inter_ms / 1000.0)

            # Extend KV pages every PAGE_SIZE decoded tokens
            if idx % PAGE_SIZE == 0:
                await self.extend_pages(session_id, PAGE_SIZE)

            token = _make_token(rng)

            # Occasionally end with punctuation for realism
            if idx == max_tokens - 1:
                token = token.rstrip() + ".\n"

            yield (token, inter_ms)

    # ------------------------------------------------------------------
    # Stats snapshot
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "vram_used_mb": round(self.vram_used_mb, 2),
            "vram_total_mb": self.vram_total_mb,
            "pages_allocated": self.pages_in_use,
            "pages_free": self.pages_free,
            "active_sequences": self.active_sequences,
            "total_pages": self._total_pages,
        }
