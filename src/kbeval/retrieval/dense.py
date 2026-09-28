"""Dense retrieval: cosine similarity over embedded chunks.

Why it exists: BM25 is lexical, and the corpus is an English manual asked about
in Korean. BM25 can only match the English tokens a Korean question happens to
carry (SS-RSRQ, 5G, MAC); "라이선스" never meets "license". A multilingual
embedder bridges that.

What it costs the experiment: a dense score is not readable by inspecting the
chunk, the way a BM25 score is. So when a pack wins under dense retrieval, the
claim is "the embedder placed this pack's text closer to the question", which
is a claim about the data AND the embedder. That is why the embedder is fixed
across packs (retrieval: is top-level) and why hybrid, not dense, is the
recommended default -- the lexical half stays inspectable.

The embedder is built from config here; this file never names a provider.
"""
from __future__ import annotations

import math

from ..config import RetrievalCfg
from ..embed import build_embedder
from ..registry import RETRIEVERS
from ..types import Chunk, Retrieved


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


@RETRIEVERS.register("dense")
class DenseRetriever:
    def __init__(self, cfg: RetrievalCfg) -> None:
        self.embedder = build_embedder(cfg.embedder)
        self.ids: list[str] = []
        self.vecs: list[list[float]] = []

    def index(self, chunks: list[Chunk]) -> None:
        self.ids = [c.chunk_id for c in chunks]
        self.vecs = [_unit(v) for v in self.embedder.embed([c.text for c in chunks], "document")]

    def rank_all(self, query: str) -> list[tuple[float, str]]:
        if not self.ids:
            raise RuntimeError("search() before index()")
        q = _unit(self.embedder.embed([query], "query")[0])
        scored = [(sum(a * b for a, b in zip(q, v)), cid) for cid, v in zip(self.ids, self.vecs)]
        # chunk_id breaks ties, so a rerun can never reorder a result set.
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored

    def search(self, query: str, k: int) -> list[Retrieved]:
        return [Retrieved(chunk_id=cid, score=round(s, 6), rank=r)
                for r, (s, cid) in enumerate(self.rank_all(query)[:k], start=1)]
