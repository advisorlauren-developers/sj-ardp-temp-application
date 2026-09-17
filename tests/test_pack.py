"""S4 -- gold mapping and the coverage gate.

The payoff of the offset invariant lands here: one labelling pass on the source
resolves onto three packs that cut and rewrite the corpus completely
differently. The gate that protects it is coverage -- a transform which deletes
the only chunk holding an answer lowers that pack's ceiling, and in the final
report that is indistinguishable from the model getting worse.
"""
from __future__ import annotations

import json

import pytest

from kbeval.packs import build_pack, load_pack, load_questions
from kbeval.packs.builder import Pack, corpus_stats
from kbeval.types import Chunk, GoldSpan, Question

PACKS = ["raw", "mid", "full"]


@pytest.mark.parametrize("name", PACKS)
def test_gold_survives_every_pack(name, cfg, questions):
    pk = build_pack(cfg, name)
    cov = pk.verify_gold_coverage(questions, cfg.eval.gold_overlap_ratio)
    assert cov["coverage"] == 1.0, cov["missing_gold"]
    assert cov["n_answerable"] == 5


@pytest.mark.parametrize("name", PACKS)
def test_every_answerable_question_lands_on_a_chunk(name, cfg, questions):
    pk = build_pack(cfg, name)
    for q in questions:
        if q.answerable:
            assert pk.gold_chunk_ids(q, cfg.eval.gold_overlap_ratio)


@pytest.mark.parametrize("name", PACKS)
def test_unanswerable_questions_have_no_gold(name, cfg, questions):
    pk = build_pack(cfg, name)
    for q in questions:
        if not q.answerable:
            assert pk.gold_chunk_ids(q, cfg.eval.gold_overlap_ratio) == []


@pytest.mark.parametrize("name", PACKS)
def test_manifest_carries_the_covariates(name, cfg):
    """n_chunks and avg_chunk_chars are not decoration: a pack that grows the
    index makes retrieval harder for reasons unrelated to data quality."""
    m = build_pack(cfg, name).manifest
    assert {"n_chunks", "avg_chunk_chars", "max_chunk_chars", "total_chars",
            "chunker", "transforms", "source"} <= set(m)
    assert m["n_chunks"] > 0


def test_packs_actually_differ(cfg):
    """If two packs produce identical corpora the axis is not being exercised."""
    shapes = {n: build_pack(cfg, n).manifest["n_chunks"] for n in PACKS}
    assert shapes["raw"] != shapes["mid"]
    raw_text = " ".join(c.text for c in build_pack(cfg, "raw").chunks)
    full_text = " ".join(c.text for c in build_pack(cfg, "full").chunks)
    assert raw_text != full_text


def test_inline_meta_does_not_dilute_gold_status(cfg, questions):
    """gold_chunk_ids must judge by SOURCE coordinates. Measuring overlap
    against len(chunk.text) would let a prepended header push a chunk under the
    threshold -- gold would silently vanish from exactly the packs that add
    metadata, i.e. the good ones."""
    mid = build_pack(cfg, "mid")          # heading + inline_meta
    ratio = cfg.eval.gold_overlap_ratio
    for q in questions:
        if not q.answerable:
            continue
        ids = mid.gold_chunk_ids(q, ratio)
        assert ids
        for cid in ids:
            c = mid.get(cid)
            assert len(c.text) > c.char_end - c.char_start   # header really was added


def test_a_pack_that_loses_gold_reports_it():
    q = Question(qid="q1", intent="qa", query="?", answerable=True,
                 gold=GoldSpan("d.md", 0, 100))
    empty = Pack("broken", [Chunk("d.md#0", "d.md", 500, 600, "elsewhere")], {})
    cov = empty.verify_gold_coverage([q], 0.3)
    assert cov["coverage"] == 0.0 and cov["missing_gold"] == ["q1"]


def test_a_chunk_fully_inside_the_span_is_gold_by_construction():
    """Fine-grained packs: a 40-char chunk can never cover 30% of a 300-char
    span, but it is entirely made of gold."""
    q = Question(qid="q1", intent="qa", query="?", answerable=True,
                 gold=GoldSpan("d.md", 0, 300))
    pk = Pack("fine", [Chunk("d.md#0", "d.md", 100, 140, "inside")], {})
    assert pk.gold_chunk_ids(q, 0.3) == ["d.md#0"]


def test_gold_does_not_leak_across_documents():
    q = Question(qid="q1", intent="qa", query="?", answerable=True,
                 gold=GoldSpan("a.md", 0, 100))
    pk = Pack("x", [Chunk("b.md#0", "b.md", 0, 100, "same offsets, other doc")], {})
    assert pk.gold_chunk_ids(q, 0.3) == []


@pytest.mark.parametrize("name", PACKS)
def test_pack_round_trips_through_disk(name, cfg):
    built = build_pack(cfg, name)
    loaded = load_pack(name)
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in built.chunks]
    assert [(c.char_start, c.char_end) for c in loaded.chunks] == \
           [(c.char_start, c.char_end) for c in built.chunks]
    assert loaded.manifest == built.manifest


def test_saved_chunks_are_utf8_readable(cfg):
    pk = build_pack(cfg, "mid")
    line = json.loads((pk.dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "\\u" not in (pk.dir / "chunks.jsonl").read_text(encoding="utf-8")
    assert {"chunk_id", "doc_id", "char_start", "char_end", "text", "meta"} == set(line)


def test_corpus_stats_survives_an_empty_corpus():
    assert corpus_stats([])["n_chunks"] == 0


def test_questions_are_filtered_by_intent(cfg):
    assert load_questions(cfg.eval.questions, ["nonexistent"]) == []
    assert len(load_questions(cfg.eval.questions, None)) == 7


def test_questions_parse_their_optional_fields(cfg):
    qs = {q.qid: q for q in load_questions(cfg.eval.questions, ["qa"])}
    assert qs["qa_001"].gold.doc_id == "olt_auth.md"
    assert qs["qa_001"].oracle_ref and qs["qa_001"].gold_answer
    assert qs["qa_101"].gold is None
