"""S5 -- retrieval, settled before a single token is spent.

Two things are being protected here. The obvious one is recall on Korean: with
whitespace tokens alone an inflected query misses its own answer. The subtle
one is that the retriever must NOT normalise terms -- doing so would apply to
every pack at once and quietly delete normalize_terms from the experiment.
"""
from __future__ import annotations

from itertools import pairwise

import pytest

from kbeval.config import RetrievalCfg, TokenizerCfg
from kbeval.packs import build_pack
from kbeval.registry import RETRIEVERS
from kbeval.retrieval import (
    build_retriever,
    hit_at_k,
    reciprocal_rank,
    summarize,
    tokenize,
)
from kbeval.types import Chunk


@pytest.fixture
def toy_chunks() -> list[Chunk]:
    texts = [
        "E-411은 ONT 시리얼 번호가 OLT 가입자 DB에 등록되지 않았을 때 발생한다.",
        "가입자 속도 프로파일은 RADIUS 속성으로 전달되며 CoA로 변경 가능하다.",
        "광 수신 레벨이 -27dBm 이하일 때 E-412가 빈번히 발생한다.",
    ]
    return [Chunk(f"t#{i}", "t.md", i * 100, i * 100 + len(t), t) for i, t in enumerate(texts)]


@pytest.fixture
def indexed(cfg, toy_chunks):
    r = build_retriever(cfg.retrieval)
    r.index(toy_chunks)
    return r


# --- tokenizer -----------------------------------------------------------

def test_hangul_is_split_off_from_latin():
    assert "autofind" in tokenize("show ont autofind로 확인한다")


def test_error_codes_stay_one_token():
    """If "E-411" fragmented into "e" + "411", every pack would match "E411"
    by accident and normalize_terms would have nothing left to fix."""
    assert "e-411" in tokenize("E-411 인증 실패")


def test_the_retriever_does_not_normalise_terms():
    """The factor must stay measurable at the pack layer, not be dissolved
    here. These two SHOULD disagree."""
    assert not set(tokenize("E411")) & set(tokenize("E-411"))


def test_bigrams_give_partial_credit_across_particles():
    """프로파일은 / 프로파일을 / 프로파일 are one word to a reader and three to a
    whitespace tokenizer."""
    assert set(tokenize("프로파일은")) & set(tokenize("프로파일"))


def test_without_bigrams_inflection_breaks_the_match():
    assert not set(tokenize("프로파일은", bigrams=False)) & set(tokenize("프로파일", bigrams=False))


def test_two_syllable_words_are_not_counted_twice():
    assert tokenize("인증") == ["인증"]


def test_tokenizer_is_case_insensitive():
    assert tokenize("OLT") == tokenize("olt")


def test_punctuation_and_empties_produce_nothing():
    assert tokenize("   ...  ") == []


# --- BM25 ----------------------------------------------------------------

def test_search_before_index_is_an_error(cfg):
    with pytest.raises(RuntimeError, match="index"):
        build_retriever(cfg.retrieval).search("아무거나", 5)


def test_ranks_are_dense_and_ordered(indexed):
    hits = indexed.search("가입자 인증", 10)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(a.score >= b.score for a, b in pairwise(hits))


def test_k_limits_the_result(indexed):
    assert len(indexed.search("가입자", 1)) == 1


def test_the_answer_chunk_ranks_first(indexed):
    assert indexed.search("E-412 광 수신 레벨", 3)[0].chunk_id == "t#2"
    assert indexed.search("속도 프로파일 CoA", 3)[0].chunk_id == "t#1"


def test_unrelated_queries_return_nothing(indexed):
    assert indexed.search("김치찌개 레시피", 10) == []


def test_results_are_reproducible(indexed):
    assert indexed.search("가입자", 10) == indexed.search("가입자", 10)


def test_ties_break_on_chunk_id(cfg):
    same = [Chunk(cid, "d.md", 0, 10, "동일한 본문이다.") for cid in ("d#9", "d#1", "d#5")]
    r = build_retriever(cfg.retrieval)
    r.index(same)
    assert [h.chunk_id for h in r.search("동일한 본문", 3)] == ["d#1", "d#5", "d#9"]


def test_idf_never_goes_negative(cfg, toy_chunks):
    """On a six-chunk pack the textbook idf turns negative for common terms and
    a frequent word would push its own chunk DOWN the ranking."""
    r = build_retriever(cfg.retrieval)
    r.index(toy_chunks)
    assert all(r._idf(t) >= 0 for t in ("발생한다", "가입자", "없는단어"))


def test_bigrams_are_what_make_inflected_queries_work(cfg, toy_chunks):
    off = RetrievalCfg(tokenizer=TokenizerCfg(bigrams=False))
    plain, bigram = build_retriever(off), build_retriever(cfg.retrieval)
    plain.index(toy_chunks)
    bigram.index(toy_chunks)
    q = "프로파일에서 가입자별 속도를"          # every word inflected away from the text
    assert not plain.search(q, 5)
    assert bigram.search(q, 5)[0].chunk_id == "t#1"


# --- the axis boundary ---------------------------------------------------

def test_the_algorithm_is_fixed_across_packs(cfg):
    """Only the index may be pack-dependent. If PackCfg could name a retriever,
    a pack could win by bringing a better search engine."""
    assert not {f for f in cfg.packs["mid"].model_fields_set if "retriev" in f}
    assert cfg.retrieval.retriever == "bm25"


def test_only_the_index_depends_on_the_pack(cfg):
    idx = {}
    for name in ("raw", "mid"):
        r = build_retriever(cfg.retrieval)
        r.index(build_pack(cfg, name).chunks)
        idx[name] = set(r.ids)
    assert idx["raw"] != idx["mid"]


def test_registry_holds_the_shipped_retriever():
    assert "bm25" in RETRIEVERS.names()


# --- metrics -------------------------------------------------------------

def test_hit_at_k_respects_the_cutoff():
    ranked = ["a", "b", "c"]
    assert hit_at_k(ranked, ["c"], 3) == 1.0
    assert hit_at_k(ranked, ["c"], 2) == 0.0


def test_reciprocal_rank_uses_the_first_hit():
    assert reciprocal_rank(["a", "b", "c"], ["b", "c"]) == 0.5
    assert reciprocal_rank(["a"], ["z"]) == 0.0


def test_summarize_averages_over_questions():
    rows = [(["a", "b"], ["a"]), (["a", "b"], ["b"])]
    m = summarize(rows, ks=(1, 2))
    assert m["hit@1"] == 0.5 and m["hit@2"] == 1.0 and m["mrr"] == 0.75 and m["n"] == 2


def test_summarize_on_no_questions_is_zero_not_a_crash():
    assert summarize([], ks=(1,))["n"] == 0


def test_unanswerable_questions_carry_no_gold_to_score(cfg, questions):
    """They are an abstention test, not a retrieval one -- excluded by having
    no gold at all rather than by a special case in the metric."""
    pk = build_pack(cfg, "mid")
    for q in questions:
        if not q.answerable:
            assert pk.gold_chunk_ids(q, cfg.eval.gold_overlap_ratio) == []
