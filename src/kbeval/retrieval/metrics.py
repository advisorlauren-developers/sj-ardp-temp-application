"""Retrieval metrics -- the only numbers in this project that cost nothing.

They are worth reading before any generation result: if hit@5 is already low
for a pack, no amount of prompt work downstream will recover the answer, and a
low score there is a data finding, not a model finding.

Unanswerable questions are excluded here by construction (they have no gold).
They are not a retrieval failure -- they are an abstention test, scored at S8.
"""
from __future__ import annotations

from collections.abc import Sequence

DEFAULT_KS = (1, 3, 5, 10)


def hit_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & set(gold) else 0.0


def reciprocal_rank(ranked: Sequence[str], gold: Sequence[str]) -> float:
    g = set(gold)
    return next((1.0 / i for i, cid in enumerate(ranked, start=1) if cid in g), 0.0)


def summarize(rows: list[tuple[Sequence[str], Sequence[str]]],
              ks: Sequence[int] = DEFAULT_KS) -> dict[str, float]:
    """rows = [(ranked_chunk_ids, gold_chunk_ids)] for gold-bearing questions."""
    n = len(rows)
    if not n:
        return {f"hit@{k}": 0.0 for k in ks} | {"mrr": 0.0, "n": 0}
    out = {f"hit@{k}": round(sum(hit_at_k(r, g, k) for r, g in rows) / n, 3) for k in ks}
    out["mrr"] = round(sum(reciprocal_rank(r, g) for r, g in rows) / n, 3)
    out["n"] = n
    return out
