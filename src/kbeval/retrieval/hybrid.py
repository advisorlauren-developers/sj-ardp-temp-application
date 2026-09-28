"""Hybrid retrieval: BM25 and dense, fused by reciprocal rank (RRF).

    score(c) = sum over rankers of 1 / (rrf_k + rank(c))

Why RRF and not a weighted score sum: BM25 scores are unbounded and move with
corpus size, cosine is in [-1, 1]. Any weight between them is a tuning knob
that would have to be re-tuned per pack -- and a per-pack knob is exactly what
the retrieval block is not allowed to have. Ranks are comparable as-is.

A chunk BM25 did not match at all (score 0) gets no BM25 term, rather than a
last-place rank: "no shared word" is information, not a tie.

Both halves use the same settings as when run alone, so hybrid is never a
third, differently-tuned retriever.
"""
from __future__ import annotations

from ..config import RetrievalCfg
from ..registry import RETRIEVERS
from ..types import Chunk, Retrieved
from .bm25 import BM25Retriever
from .dense import DenseRetriever


@RETRIEVERS.register("hybrid")
class HybridRetriever:
    def __init__(self, cfg: RetrievalCfg) -> None:
        self.bm25 = BM25Retriever(cfg)
        self.dense = DenseRetriever(cfg)
        self.rrf_k = cfg.hybrid.rrf_k
        self.n = 0

    def index(self, chunks: list[Chunk]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)
        self.n = len(chunks)

    def search(self, query: str, k: int) -> list[Retrieved]:
        fused: dict[str, float] = {}
        for r in self.bm25.search(query, self.n):
            fused[r.chunk_id] = fused.get(r.chunk_id, 0.0) + 1.0 / (self.rrf_k + r.rank)
        for rank, (_, cid) in enumerate(self.dense.rank_all(query), start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        scored = sorted(fused.items(), key=lambda x: (-x[1], x[0]))
        return [Retrieved(chunk_id=cid, score=round(s, 6), rank=r)
                for r, (cid, s) in enumerate(scored[:k], start=1)]
