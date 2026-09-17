"""Retrieval scoring -- computed from the trace, costing nothing.

Two different questions live here and must not be collapsed:

    hit@k         did search FIND the answer?
    gold_injected did the answer actually REACH the model?

They come apart exactly when the token budget bites: a pack can rank gold first
and still lose it to a fatter chunk above it. Reporting only hit@k would hide
that, and reporting only gold_injected would blame search for a budget effect.
"""
from __future__ import annotations

from typing import Any

from ...config import Config
from ...registry import SCORERS
from ...retrieval import hit_at_k, reciprocal_rank
from ...types import Question


@SCORERS.register("retrieval")
class RetrievalScorer:
    name = "retrieval"

    def __init__(self, cfg: Config) -> None:
        self.ks = cfg.scoring.ks

    def score(self, row: dict[str, Any], q: Question) -> dict[str, Any]:
        gold = row.get("gold_chunk_ids") or []
        if not gold or not row.get("retrieved"):
            return {}                      # bare / oracle_ref / unanswerable: nothing to score

        ranked = [r["chunk_id"] for r in sorted(row["retrieved"], key=lambda r: r["rank"])]
        out: dict[str, Any] = {f"hit@{k}": hit_at_k(ranked, gold, k) for k in self.ks}
        out["mrr"] = round(reciprocal_rank(ranked, gold), 4)
        out["gold_injected"] = float(bool(set(row.get("injected_chunk_ids", [])) & set(gold)))
        out["gold_lost_to_budget"] = float(
            bool(set(row.get("skipped_chunk_ids", [])) & set(gold)) and not out["gold_injected"])
        return out
