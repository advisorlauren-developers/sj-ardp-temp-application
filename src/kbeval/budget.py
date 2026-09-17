"""The context budget -- the control variable of the whole experiment.

Rule 2 in one place: what is held constant across packs is the TOKEN BUDGET,
not top_k. Packs produce chunks of very different sizes, so a fixed top_k
would hand each pack a different prompt length, and a 30B model's long-context
degradation would arrive in the results dressed up as data quality.

Two consequences that look like details and are not:

* A chunk that does not fit is SKIPPED, never truncated. Truncating damages the
  thing being measured -- a half chunk is neither the pack's chunk nor a
  smaller one, and its retrieval score no longer describes what the model saw.
* Skipping CONTINUES down the ranking instead of stopping. A single oversized
  chunk must not blank out everything beneath it; that would penalise packs
  with uneven chunk sizes twice.

Token counting is deliberately crude and model-independent (chars_per_token).
A real tokenizer would differ per model family, and then "same budget" would
silently mean "different budget per model" -- the exact confound this exists
to prevent.
"""
from __future__ import annotations

from dataclasses import dataclass

from .types import Chunk


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return int(len(text) / max(chars_per_token, 1e-9)) + 1


@dataclass(frozen=True)
class Packed:
    chunks: list[Chunk]
    tokens: int
    skipped: list[str]      # chunk_ids that did not fit; kept for the trace


def fit(chunks: list[Chunk], budget_tokens: int, max_chunks: int,
        chars_per_token: float) -> Packed:
    """Fill the budget in rank order. Order is preserved; nothing is cut."""
    kept: list[Chunk] = []
    skipped: list[str] = []
    used = 0
    for c in chunks:
        if len(kept) >= max_chunks:
            skipped.append(c.chunk_id)
            continue
        cost = estimate_tokens(c.text, chars_per_token)
        if used + cost > budget_tokens:
            skipped.append(c.chunk_id)
            continue
        kept.append(c)
        used += cost
    return Packed(chunks=kept, tokens=used, skipped=skipped)
