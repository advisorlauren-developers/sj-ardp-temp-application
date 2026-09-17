"""Okapi BM25, pure Python. The PoC default.

No dependency, no embedding model, no index-build cost worth measuring. It also
keeps the pack axis interpretable: BM25 scores are lexical, so when a pack wins
it is because the transform changed the words in the chunk -- which is a claim
about the data, testable by reading the chunk.
"""
from __future__ import annotations

import math
from collections import Counter

from ..config import RetrievalCfg
from ..registry import RETRIEVERS
from ..types import Chunk, Retrieved
from .base import tokenize


@RETRIEVERS.register("bm25")
class BM25Retriever:
    def __init__(self, cfg: RetrievalCfg) -> None:
        self.k1 = cfg.bm25.k1
        self.b = cfg.bm25.b
        self.bigrams = cfg.tokenizer.bigrams
        self.ids: list[str] = []
        self.tf: list[Counter[str]] = []
        self.dl: list[int] = []
        self.df: Counter[str] = Counter()
        self.avgdl = 0.0

    def index(self, chunks: list[Chunk]) -> None:
        self.ids = [c.chunk_id for c in chunks]
        toks = [tokenize(c.text, self.bigrams) for c in chunks]
        self.tf = [Counter(t) for t in toks]
        self.dl = [len(t) for t in toks]
        self.avgdl = sum(self.dl) / max(len(self.dl), 1)
        self.df = Counter()
        for t in toks:
            self.df.update(set(t))

    def _idf(self, term: str) -> float:
        """The 1 + (...) form, not the classic one.

        With six chunks in a pack, a common term has df close to N and the
        textbook idf goes NEGATIVE -- a frequent word would then push a chunk
        DOWN the ranking. On a corpus this small that is not a rounding detail,
        it reverses results.
        """
        n = len(self.ids)
        df = self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[Retrieved]:
        if not self.ids:
            raise RuntimeError("search() before index()")
        terms = set(tokenize(query, self.bigrams))
        scored: list[tuple[float, str]] = []
        for i, cid in enumerate(self.ids):
            s = 0.0
            for t in terms:
                f = self.tf[i].get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / max(self.avgdl, 1e-9))
                s += self._idf(t) * f * (self.k1 + 1) / denom
            if s > 0:
                scored.append((s, cid))

        # chunk_id breaks ties, so a rerun can never reorder a result set.
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [Retrieved(chunk_id=cid, score=round(s, 6), rank=r)
                for r, (s, cid) in enumerate(scored[:k], start=1)]
