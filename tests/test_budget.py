"""S6 -- the control variable.

Rule 2. What is held constant across packs is the token budget, not top_k.
The two behaviours worth a test are the ones that look like rounding errors and
are not: a chunk that does not fit is skipped rather than cut, and skipping
continues down the ranking instead of stopping.
"""
from __future__ import annotations

import pytest

from kbeval.budget import estimate_tokens, fit
from kbeval.types import Chunk


def chunk(cid: str, chars: int) -> Chunk:
    return Chunk(cid, "d.md", 0, chars, "가" * chars)


def test_tokens_are_estimated_from_chars():
    assert estimate_tokens("가" * 100, 2.0) == 51        # +1, never zero


def test_empty_text_still_costs_something():
    assert estimate_tokens("", 2.0) == 1


def test_chunks_are_never_truncated():
    """A half chunk is neither this pack's chunk nor a smaller one, and its
    retrieval score stops describing what the model saw."""
    c = chunk("c0", 1000)
    packed = fit([c], budget_tokens=10, max_chunks=8, chars_per_token=2.0)
    assert packed.chunks == []
    assert packed.skipped == ["c0"]


def test_an_oversized_chunk_does_not_blank_out_the_rest():
    """Stopping at the first overflow would penalise packs with uneven chunk
    sizes twice -- once for the fat chunk, once for everything under it."""
    packed = fit([chunk("big", 10_000), chunk("small", 40)],
                 budget_tokens=100, max_chunks=8, chars_per_token=2.0)
    assert [c.chunk_id for c in packed.chunks] == ["small"]
    assert packed.skipped == ["big"]


def test_rank_order_is_preserved():
    cs = [chunk(f"c{i}", 40) for i in range(4)]
    packed = fit(cs, budget_tokens=1000, max_chunks=8, chars_per_token=2.0)
    assert [c.chunk_id for c in packed.chunks] == ["c0", "c1", "c2", "c3"]


def test_max_chunks_is_a_ceiling_not_the_control():
    packed = fit([chunk(f"c{i}", 10) for i in range(10)],
                 budget_tokens=10_000, max_chunks=3, chars_per_token=2.0)
    assert len(packed.chunks) == 3
    assert len(packed.skipped) == 7


def test_reported_tokens_match_what_was_kept():
    cs = [chunk("a", 100), chunk("b", 100)]
    packed = fit(cs, budget_tokens=1000, max_chunks=8, chars_per_token=2.0)
    assert packed.tokens == sum(estimate_tokens(c.text, 2.0) for c in packed.chunks)


def test_nothing_in_nothing_out():
    packed = fit([], budget_tokens=100, max_chunks=8, chars_per_token=2.0)
    assert packed.chunks == [] and packed.tokens == 0 and packed.skipped == []


@pytest.mark.parametrize("budget", [50, 200, 1000, 3000])
def test_the_budget_is_never_exceeded(budget):
    cs = [chunk(f"c{i}", 120) for i in range(20)]
    assert fit(cs, budget, 99, 2.0).tokens <= budget
