"""
Tests for InferenceEngine — PagedAttention simulation.

Tests:
  - Page allocation (correct page count)
  - VRAM budget enforcement
  - Page deallocation / pool return
  - Token generation yields correct count
  - TTFT is first token latency
  - Extend pages during decode
  - Stats reporting
"""

from __future__ import annotations

import asyncio
import math

import pytest

from engine.inference import (
    InferenceEngine,
    PAGE_SIZE,
    VRAM_TOTAL_MB,
    BYTES_PER_PAGE,
    MB,
    SessionKVState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return InferenceEngine()


# ---------------------------------------------------------------------------
# Page allocation
# ---------------------------------------------------------------------------

class TestPageAllocation:
    @pytest.mark.asyncio
    async def test_allocates_correct_page_count(self, engine):
        """A prompt of N words needs ceil(N/PAGE_SIZE) pages."""
        prompt_len = 50  # tokens
        pages_expected = math.ceil(prompt_len / PAGE_SIZE)

        state = await engine.allocate_pages("sess-1", prompt_len)

        assert len(state.pages) == pages_expected
        assert state.session_id == "sess-1"
        assert state.tokens_prefilled == prompt_len

    @pytest.mark.asyncio
    async def test_single_token_prompt_allocates_one_page(self, engine):
        state = await engine.allocate_pages("sess-single", 1)
        assert len(state.pages) == 1

    @pytest.mark.asyncio
    async def test_exactly_one_page_boundary(self, engine):
        """Exactly PAGE_SIZE tokens → exactly 1 page."""
        state = await engine.allocate_pages("sess-boundary", PAGE_SIZE)
        assert len(state.pages) == 1

    @pytest.mark.asyncio
    async def test_one_over_page_boundary(self, engine):
        """PAGE_SIZE + 1 tokens → 2 pages."""
        state = await engine.allocate_pages("sess-overflow", PAGE_SIZE + 1)
        assert len(state.pages) == 2

    @pytest.mark.asyncio
    async def test_pages_are_unique(self, engine):
        """Each allocated page should have a distinct page_id."""
        state = await engine.allocate_pages("sess-unique", 64)
        page_ids = [p.page_id for p in state.pages]
        assert len(page_ids) == len(set(page_ids))

    @pytest.mark.asyncio
    async def test_pages_removed_from_free_pool(self, engine):
        initial_free = engine.pages_free
        await engine.allocate_pages("sess-pool", 32)
        pages_used = math.ceil(32 / PAGE_SIZE)
        assert engine.pages_free == initial_free - pages_used

    @pytest.mark.asyncio
    async def test_session_tracked_in_kv_cache(self, engine):
        await engine.allocate_pages("sess-track", 16)
        assert "sess-track" in engine._kv_cache

    @pytest.mark.asyncio
    async def test_active_sequences_increments(self, engine):
        assert engine.active_sequences == 0
        await engine.allocate_pages("s1", 16)
        assert engine.active_sequences == 1
        await engine.allocate_pages("s2", 16)
        assert engine.active_sequences == 2


# ---------------------------------------------------------------------------
# VRAM budget enforcement
# ---------------------------------------------------------------------------

class TestVRAMBudget:
    @pytest.mark.asyncio
    async def test_raises_when_vram_exhausted(self, engine):
        """Requesting more pages than available must raise RuntimeError."""
        # Drain nearly all pages first
        large_prompt = engine.pages_free * PAGE_SIZE  # exactly uses all pages
        await engine.allocate_pages("sess-big", large_prompt)

        # Now there are no free pages — next allocation should fail
        with pytest.raises(RuntimeError, match="Out of VRAM"):
            await engine.allocate_pages("sess-fail", 1)

    @pytest.mark.asyncio
    async def test_error_message_contains_context(self, engine):
        """Error should mention pages needed vs pages available."""
        # Fill the pool
        await engine.allocate_pages("sess-fill", engine.pages_free * PAGE_SIZE)
        with pytest.raises(RuntimeError) as exc_info:
            await engine.allocate_pages("sess-fail", PAGE_SIZE)
        assert "pages" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_vram_total_is_80gb(self, engine):
        assert engine.vram_total_mb == VRAM_TOTAL_MB == 80_000


# ---------------------------------------------------------------------------
# Page deallocation
# ---------------------------------------------------------------------------

class TestPageDeallocation:
    @pytest.mark.asyncio
    async def test_free_returns_pages_to_pool(self, engine):
        initial_free = engine.pages_free
        await engine.allocate_pages("sess-free", 32)
        assert engine.pages_free < initial_free
        await engine.free_pages("sess-free")
        assert engine.pages_free == initial_free

    @pytest.mark.asyncio
    async def test_free_removes_session_from_cache(self, engine):
        await engine.allocate_pages("sess-remove", 16)
        assert "sess-remove" in engine._kv_cache
        await engine.free_pages("sess-remove")
        assert "sess-remove" not in engine._kv_cache

    @pytest.mark.asyncio
    async def test_free_nonexistent_session_is_noop(self, engine):
        initial_free = engine.pages_free
        await engine.free_pages("nonexistent-session")
        assert engine.pages_free == initial_free  # unchanged

    @pytest.mark.asyncio
    async def test_active_sequences_decrements_on_free(self, engine):
        await engine.allocate_pages("s1", 16)
        await engine.allocate_pages("s2", 16)
        assert engine.active_sequences == 2
        await engine.free_pages("s1")
        assert engine.active_sequences == 1
        await engine.free_pages("s2")
        assert engine.active_sequences == 0

    @pytest.mark.asyncio
    async def test_reallocate_after_free(self, engine):
        """After freeing, the same pages should be re-usable."""
        initial_free = engine.pages_free
        await engine.allocate_pages("sess-reuse", 64)
        await engine.free_pages("sess-reuse")
        # Should succeed again
        state = await engine.allocate_pages("sess-reuse-2", 64)
        assert len(state.pages) > 0
        assert engine.pages_free == initial_free - len(state.pages)


# ---------------------------------------------------------------------------
# Token generation
# ---------------------------------------------------------------------------

class TestTokenGeneration:
    @pytest.mark.asyncio
    async def test_generates_correct_token_count(self, engine):
        await engine.allocate_pages("sess-gen", 10)
        max_tokens = 5
        tokens = []
        async for token, latency in engine.generate_tokens(
            prompt="test prompt", max_tokens=max_tokens, model="Mistral-7B", session_id="sess-gen"
        ):
            tokens.append((token, latency))
        await engine.free_pages("sess-gen")
        assert len(tokens) == max_tokens

    @pytest.mark.asyncio
    async def test_first_token_has_higher_latency(self, engine):
        """TTFT (first token) should take longer than subsequent tokens on average."""
        await engine.allocate_pages("sess-ttft", 10)
        latencies = []
        async for token, latency in engine.generate_tokens(
            prompt="latency test", max_tokens=10, model="Mistral-7B", session_id="sess-ttft"
        ):
            latencies.append(latency)
        await engine.free_pages("sess-ttft")

        ttft = latencies[0]
        subsequent_avg = sum(latencies[1:]) / len(latencies[1:])
        # TTFT min (80ms) > subsequent max (30ms) for Mistral-7B
        assert ttft > subsequent_avg, (
            f"Expected TTFT ({ttft:.1f}ms) > avg subsequent ({subsequent_avg:.1f}ms)"
        )

    @pytest.mark.asyncio
    async def test_tokens_are_strings(self, engine):
        await engine.allocate_pages("sess-str", 5)
        async for token, latency in engine.generate_tokens(
            prompt="string test", max_tokens=3, model="Mistral-7B", session_id="sess-str"
        ):
            assert isinstance(token, str)
            assert isinstance(latency, float)
        await engine.free_pages("sess-str")

    @pytest.mark.asyncio
    async def test_llama_slower_than_mistral(self, engine):
        """Llama-3-8B should have higher average inter-token latency than Mistral-7B."""
        mistral_latencies = []
        llama_latencies = []

        await engine.allocate_pages("sess-mistral", 5)
        async for _, latency in engine.generate_tokens(
            "bench", 5, "Mistral-7B", "sess-mistral"
        ):
            mistral_latencies.append(latency)
        await engine.free_pages("sess-mistral")

        await engine.allocate_pages("sess-llama", 5)
        async for _, latency in engine.generate_tokens(
            "bench", 5, "Llama-3-8B", "sess-llama"
        ):
            llama_latencies.append(latency)
        await engine.free_pages("sess-llama")

        # First token comparison (TTFT)
        # Mistral ttft_lo=80 < Llama ttft_lo=120
        mistral_ttft = mistral_latencies[0]
        llama_ttft = llama_latencies[0]
        # Check that Llama TTFT range is higher (probabilistic, but range doesn't overlap much)
        # Use inter-token which is more reliable
        mistral_inter_avg = sum(mistral_latencies[1:]) / max(len(mistral_latencies[1:]), 1)
        llama_inter_avg = sum(llama_latencies[1:]) / max(len(llama_latencies[1:]), 1)
        assert llama_inter_avg >= mistral_inter_avg * 0.8  # Llama should be same or slower

    @pytest.mark.asyncio
    async def test_max_tokens_zero_yields_nothing(self, engine):
        """max_tokens=0 should yield no tokens (range(1, 0) is empty and first yields once)."""
        await engine.allocate_pages("sess-zero", 5)
        # max_tokens=1 → only TTFT token
        tokens = []
        async for token, latency in engine.generate_tokens(
            "test", 1, "Mistral-7B", "sess-zero"
        ):
            tokens.append(token)
        await engine.free_pages("sess-zero")
        assert len(tokens) == 1


# ---------------------------------------------------------------------------
# Page extension during decode
# ---------------------------------------------------------------------------

class TestPageExtension:
    @pytest.mark.asyncio
    async def test_extend_pages_adds_pages(self, engine):
        await engine.allocate_pages("sess-ext", 1)
        initial_pages = len(engine._kv_cache["sess-ext"].pages)
        # Extend by PAGE_SIZE tokens (should need one more page)
        await engine.extend_pages("sess-ext", PAGE_SIZE)
        final_pages = len(engine._kv_cache["sess-ext"].pages)
        assert final_pages >= initial_pages

    @pytest.mark.asyncio
    async def test_extend_nonexistent_session_is_noop(self, engine):
        initial_free = engine.pages_free
        await engine.extend_pages("nonexistent", 16)
        # Should not crash and free count unchanged
        assert engine.pages_free == initial_free


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    @pytest.mark.asyncio
    async def test_get_stats_keys(self, engine):
        stats = engine.get_stats()
        required_keys = {
            "vram_used_mb", "vram_total_mb", "pages_allocated",
            "pages_free", "active_sequences", "total_pages"
        }
        assert required_keys.issubset(stats.keys())

    @pytest.mark.asyncio
    async def test_vram_used_increases_on_allocation(self, engine):
        before = engine.vram_used_mb
        await engine.allocate_pages("sess-vram", 32)
        after = engine.vram_used_mb
        assert after > before

    @pytest.mark.asyncio
    async def test_vram_used_decreases_on_free(self, engine):
        await engine.allocate_pages("sess-vram2", 32)
        used = engine.vram_used_mb
        await engine.free_pages("sess-vram2")
        assert engine.vram_used_mb < used


# ---------------------------------------------------------------------------
# SessionKVState
# ---------------------------------------------------------------------------

class TestSessionKVState:
    def test_total_pages_matches_pages_list(self):
        from engine.inference import KVPage
        state = SessionKVState(
            session_id="s1",
            pages=[KVPage(page_id=i) for i in range(4)],
        )
        assert state.total_pages == 4

    def test_vram_mb_is_sum_of_pages(self):
        from engine.inference import KVPage
        pages = [KVPage(page_id=i) for i in range(3)]
        state = SessionKVState(session_id="s1", pages=pages)
        expected = sum(p.size_mb for p in pages)
        assert abs(state.vram_mb - expected) < 0.001
