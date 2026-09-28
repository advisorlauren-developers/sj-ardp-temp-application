"""Embedder Protocol + the one factory, and the cache that makes it an index.

Why a package of its own rather than a file under retrieval/: the retrieval
package is not allowed to see the provider SDK (tests/test_architecture.py),
and a real embedder is a provider call. So the provider lives here, and
retrieval/dense.py only ever sees `Embedder.embed()`.

**Queries and documents are embedded differently.** Asymmetric models (Cohere,
most modern ones) take an input type, and using the document mode for a query
quietly costs recall. The Protocol makes the caller say which one it wants.

**The cache is part of the measurement, not an optimisation.** A pack's
vectors are its dense index. Re-embedding the same text on every run would
make the index a fresh API response each time, and a provider-side model
refresh would show up as a pack effect. Cached vectors pin the index the same
way data/extracted pins the PDF text. The key is (model, dim, kind, text), so a
transform that changes a chunk's text gets a new vector and nothing else does.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from ..config import EmbedderCfg
from ..registry import EMBEDDERS

Kind = Literal["query", "document"]

CACHE_ROOT = Path(os.environ.get("KBEVAL_EMBED_CACHE", "data/embed_cache"))


@runtime_checkable
class Embedder(Protocol):
    model_id: str

    def embed(self, texts: list[str], kind: Kind) -> list[list[float]]:
        ...


class CachedEmbedder:
    """Append-only (model, kind, text) -> vector store in front of an Embedder.

    One JSONL per model+dim, one line per vector, flushed per batch -- the same
    discipline as traces.jsonl, for the same reason: a line on disk is a vector
    that will never be bought again.
    """

    def __init__(self, inner: Embedder, cfg: EmbedderCfg, persist: bool) -> None:
        self.inner = inner
        self.model_id = inner.model_id
        self.batch = max(cfg.batch, 1)
        self._mem: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self._path: Path | None = None
        if persist:
            slug = f"{cfg.provider}-{cfg.id}-{cfg.dim}".replace("/", "_").replace(":", "_")
            self._path = CACHE_ROOT / f"{slug}.jsonl"
            self._load()

    @staticmethod
    def _key(kind: str, text: str) -> str:
        return hashlib.sha256(f"{kind}\x00{text}".encode()).hexdigest()

    def _load(self) -> None:
        assert self._path is not None
        if not self._path.exists():
            return
        for ln in self._path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                continue          # a torn last line costs one vector, not the cache
            self._mem[row["k"]] = row["v"]

    def _store(self, rows: list[tuple[str, list[float]]]) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            for k, v in rows:
                f.write(json.dumps({"k": k, "v": v}) + "\n")
            f.flush()

    def embed(self, texts: list[str], kind: Kind) -> list[list[float]]:
        keys = [self._key(kind, t) for t in texts]
        with self._lock:
            todo = sorted({(k, t) for k, t in zip(keys, texts) if k not in self._mem})
        for i in range(0, len(todo), self.batch):
            part = todo[i:i + self.batch]
            vecs = self.inner.embed([t for _, t in part], kind)
            if len(vecs) != len(part):
                raise RuntimeError(f"embedder returned {len(vecs)} vectors for {len(part)} texts")
            rows = [(k, v) for (k, _), v in zip(part, vecs)]
            with self._lock:
                self._mem.update(rows)
                self._store(rows)
        return [self._mem[k] for k in keys]


def build_embedder(cfg: EmbedderCfg) -> Embedder:
    """The only entry point. The mock is not persisted: it is free to recompute,
    and tests must not write under data/."""
    inner = EMBEDDERS.get(cfg.provider)(cfg)
    return CachedEmbedder(inner, cfg, persist=cfg.provider != "mock")
