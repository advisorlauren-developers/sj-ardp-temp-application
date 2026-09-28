"""Dense + hybrid retrieval, judged with the mock embedder (no AWS).

What matters is not that the mock ranks well -- it is character trigrams -- but
that the machinery around it keeps the experiment's promises: deterministic
rankings, the embedder fixed by config rather than by pack, an index that is
pinned on disk, and bm25 coordinates that did not move when the embedder
arrived.
"""
from __future__ import annotations

import json

import pytest

from kbeval.config import EmbedderCfg, RetrievalCfg, load_config
from kbeval.embed import build_embedder
from kbeval.embed.base import CachedEmbedder
from kbeval.embed.mock import MockEmbedder
from kbeval.retrieval import build_retriever
from kbeval.types import Chunk


def chunks() -> list[Chunk]:
    texts = ["Request a license by sending an IMSI, IMEI, or MAC address.",
             "DL MAC TP\nUL MAC TP\nDL RLC TP",
             "The 5G NR CA screen shows PCI, NR-ARFCN, SS-RSRP and SS-RSRQ."]
    out, pos = [], 0
    for i, t in enumerate(texts):
        out.append(Chunk(f"d#c{i}", "d", pos, pos + len(t), t, {}))
        pos += len(t) + 1
    return out


@pytest.mark.parametrize("name", ["dense", "hybrid"])
def test_registered_and_deterministic(name):
    cfg = RetrievalCfg(retriever=name)
    a, b = build_retriever(cfg), build_retriever(cfg)
    a.index(chunks())
    b.index(chunks())
    q = "SS-RSRQ on the 5G NR CA screen"
    assert [r.chunk_id for r in a.search(q, 3)] == [r.chunk_id for r in b.search(q, 3)]
    assert a.search(q, 3)[0].chunk_id == "d#c2"
    assert [r.rank for r in a.search(q, 3)] == [1, 2, 3]


def test_search_before_index_is_an_error():
    with pytest.raises(RuntimeError):
        build_retriever(RetrievalCfg(retriever="dense")).search("x", 1)


def test_hybrid_is_rrf_of_its_two_halves():
    cfg = RetrievalCfg(retriever="hybrid")
    h = build_retriever(cfg)
    h.index(chunks())
    q = "MAC address license"
    bm = {r.chunk_id: r.rank for r in h.bm25.search(q, 3)}
    dn = {cid: i for i, (_, cid) in enumerate(h.dense.rank_all(q), start=1)}
    k = cfg.hybrid.rrf_k
    for r in h.search(q, 3):
        want = 1 / (k + dn[r.chunk_id]) + (1 / (k + bm[r.chunk_id]) if r.chunk_id in bm else 0)
        assert r.score == pytest.approx(want, abs=1e-6)


def test_bm25_coordinates_ignore_embedder_settings(cfg):
    """Adding the embedder must not orphan every existing bm25 run."""
    c2 = cfg.model_copy(deep=True)
    c2.retrieval.embedder.id = "something-else"
    c2.retrieval.hybrid.rrf_k = 7
    assert c2.hash == cfg.hash


def test_embedder_is_part_of_a_dense_coordinate(cfg):
    a = cfg.model_copy(deep=True)
    a.retrieval.retriever = "dense"
    b = a.model_copy(deep=True)
    b.retrieval.embedder.dim = 256
    assert a.hash != cfg.hash and a.hash != b.hash


def test_unknown_embedder_key_is_an_error(raw_cfg):
    raw_cfg["retrieval"]["embedder"] = {"provider": "mock", "idd": "typo"}
    with pytest.raises(Exception):
        from kbeval.config import Config
        Config.model_validate(raw_cfg)


class Counting(MockEmbedder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.calls = 0

    def embed(self, texts, kind):
        self.calls += len(texts)
        return super().embed(texts, kind)


def test_cache_pins_the_index_and_is_reused(tmp_path, monkeypatch):
    monkeypatch.setattr("kbeval.embed.base.CACHE_ROOT", tmp_path)
    cfg = EmbedderCfg(provider="mock", id="m", dim=16)
    inner = Counting(cfg)
    e = CachedEmbedder(inner, cfg, persist=True)
    v1 = e.embed(["a", "b", "a"], "document")
    assert inner.calls == 2                       # duplicates embedded once
    e.embed(["a"], "query")
    assert inner.calls == 3                       # kind is part of the key

    again = Counting(cfg)
    v2 = CachedEmbedder(again, cfg, persist=True).embed(["a", "b", "a"], "document")
    assert again.calls == 0 and v1 == v2          # a new process reads the pin

    f = next(tmp_path.glob("*.jsonl"))
    f.write_text(f.read_text() + '{"k": "torn', encoding="utf-8")
    CachedEmbedder(Counting(cfg), cfg, persist=True)   # torn tail is not fatal
    assert all("k" in json.loads(ln) for ln in f.read_text().splitlines()[:-1])


def test_mock_embedder_is_not_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr("kbeval.embed.base.CACHE_ROOT", tmp_path)
    build_embedder(EmbedderCfg()).embed(["x"], "query")
    assert not list(tmp_path.iterdir())


def test_hybrid_overlay_config_loads():
    c = load_config(["configs/base.yaml", "configs/retrieval-hybrid.yaml"])
    assert c.retrieval.retriever == "hybrid"
    assert c.retrieval.embedder.id.startswith("us.")    # INFERENCE_PROFILE only
