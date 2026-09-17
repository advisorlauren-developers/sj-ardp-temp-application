"""S8 -- scoring, where a confound is one careless default away.

Every test here defends a separation: an error is not a wrong answer, a parse
failure is not a wrong answer, an abstention on an unanswerable question is the
RIGHT answer, and the judge must never be shown the reference text that one of
the modes was given as input.
"""
from __future__ import annotations

import pytest

from kbeval.eval.scorers import build_scorer, parse_verdict
from kbeval.eval.scorers.judge import NO_CONTEXT
from kbeval.registry import SCORERS
from kbeval.types import GoldSpan, Question


def trace(**kw):
    base = {"spec_key": "a|rag|mid", "qid": "qa_001", "repeat": 0, "mode": "rag",
            "agent": "mock_a", "pack": "mid", "answer": "ONT 시리얼 미등록이라 발생한다.",
            "retrieved": [], "injected_chunk_ids": [], "skipped_chunk_ids": [],
            "gold_chunk_ids": [], "error": None}
    return base | kw


ANSWERABLE = Question(qid="qa_001", intent="qa", query="E-411 원인은?", answerable=True,
                      gold=GoldSpan("olt_auth.md", 0, 10),
                      gold_answer="ONT 시리얼이 OLT 가입자 DB에 미등록이라 발생한다.",
                      oracle_ref="절대 judge에게 보여서는 안 되는 참조 문단이다.")
UNANSWERABLE = Question(qid="qa_101", intent="qa", query="E-415 원인은?", answerable=False)


# --- retrieval scorer ----------------------------------------------------

@pytest.fixture
def retrieval_scorer(cfg):
    return build_scorer("retrieval", cfg)


def test_hit_and_mrr_come_from_the_trace(retrieval_scorer):
    row = trace(gold_chunk_ids=["c2"],
                retrieved=[{"chunk_id": "c1", "score": 2.0, "rank": 1},
                           {"chunk_id": "c2", "score": 1.0, "rank": 2}])
    s = retrieval_scorer.score(row, ANSWERABLE)
    assert s["hit@1"] == 0.0 and s["hit@3"] == 1.0 and s["mrr"] == 0.5


def test_finding_gold_and_injecting_it_are_different_questions(retrieval_scorer):
    """A pack can rank gold first and still lose it to a fatter chunk above it
    once the budget bites. One number cannot say both things."""
    row = trace(gold_chunk_ids=["c2"],
                retrieved=[{"chunk_id": "c2", "score": 1.0, "rank": 1}],
                injected_chunk_ids=[], skipped_chunk_ids=["c2"])
    s = retrieval_scorer.score(row, ANSWERABLE)
    assert s["hit@1"] == 1.0
    assert s["gold_injected"] == 0.0
    assert s["gold_lost_to_budget"] == 1.0


def test_gold_that_reached_the_model_is_not_flagged_as_lost(retrieval_scorer):
    row = trace(gold_chunk_ids=["c2"],
                retrieved=[{"chunk_id": "c2", "score": 1.0, "rank": 1}],
                injected_chunk_ids=["c2"])
    s = retrieval_scorer.score(row, ANSWERABLE)
    assert s["gold_injected"] == 1.0 and s["gold_lost_to_budget"] == 0.0


def test_modes_without_retrieval_score_nothing(retrieval_scorer):
    assert retrieval_scorer.score(trace(mode="bare", pack=None), ANSWERABLE) == {}


def test_unanswerable_questions_have_nothing_to_retrieve(retrieval_scorer):
    row = trace(retrieved=[{"chunk_id": "c1", "score": 1.0, "rank": 1}], gold_chunk_ids=[])
    assert retrieval_scorer.score(row, UNANSWERABLE) == {}


# --- judge: what it is allowed to see ------------------------------------

@pytest.fixture
def judge(cfg):
    return build_scorer("judge", cfg)


def test_the_judge_never_sees_oracle_ref(judge, monkeypatch):
    """That text IS the model's input in oracle_ref mode. A judge comparing an
    answer against its own prompt would score that mode high by construction,
    inflating the denominator of every realization rate at once."""
    seen = {}
    monkeypatch.setattr(judge.llm, "generate",
                        lambda s, u: seen.setdefault("user", u) and None or _ok())
    judge.score(trace(mode="oracle_ref", pack=None), ANSWERABLE)
    assert ANSWERABLE.oracle_ref not in seen["user"]
    assert NO_CONTEXT in seen["user"]


def test_the_judge_does_see_the_gold_answer(judge, monkeypatch):
    seen = {}
    monkeypatch.setattr(judge.llm, "generate",
                        lambda s, u: seen.setdefault("user", u) and None or _ok())
    judge.score(trace(), ANSWERABLE)
    assert ANSWERABLE.gold_answer in seen["user"]


def test_rag_context_is_rebuilt_from_the_chunk_ids(cfg, judge, monkeypatch):
    from kbeval.packs import build_pack
    pk = build_pack(cfg, "mid")
    cid = pk.chunks[1].chunk_id
    seen = {}
    monkeypatch.setattr(judge.llm, "generate",
                        lambda s, u: seen.setdefault("user", u) and None or _ok())
    judge.score(trace(injected_chunk_ids=[cid]), ANSWERABLE)
    assert pk.get(cid).text in seen["user"]


def _ok():
    from kbeval.llm.base import LLMResponse
    return LLMResponse(text='{"correctness":1,"groundedness":1,'
                            '"abstention":0,"harmful_action":0}')


# --- judge: the separations ----------------------------------------------

def test_a_provider_error_is_never_scored(judge):
    """Rule 8. Scoring an outage as a wrong answer turns it into a data finding."""
    s = judge.score(trace(error="throttled", answer=""), ANSWERABLE)
    assert s == {"skipped": "error"}


def test_a_judge_that_fails_to_parse_is_format_fail_not_zero(judge, monkeypatch):
    from kbeval.llm.base import LLMResponse
    monkeypatch.setattr(judge.llm, "generate", lambda s, u: LLMResponse(text="음... 잘 모르겠네요"))
    s = judge.score(trace(), ANSWERABLE)
    assert s == {"format_fail": 1.0}
    assert "correctness" not in s


def test_a_judge_outage_is_skipped_not_scored(judge, monkeypatch):
    from kbeval.llm.base import LLMResponse
    monkeypatch.setattr(judge.llm, "generate", lambda s, u: LLMResponse(text="", error="boom"))
    assert judge.score(trace(), ANSWERABLE)["skipped"] == "judge_error"


def test_abstaining_on_an_unanswerable_question_is_correct(judge, monkeypatch):
    """The whole reason the distractors exist. Leaving correctness at 0 would
    punish exactly the behaviour being tested for."""
    from kbeval.llm.base import LLMResponse
    monkeypatch.setattr(judge.llm, "generate", lambda s, u: LLMResponse(
        text='{"correctness":0,"groundedness":null,"abstention":1,"harmful_action":0}'))
    s = judge.score(trace(answer="답변 불가"), UNANSWERABLE)
    assert s["correctness"] == 1.0 == s["abstention"]


def test_answering_an_unanswerable_question_is_wrong(judge, monkeypatch):
    from kbeval.llm.base import LLMResponse
    monkeypatch.setattr(judge.llm, "generate", lambda s, u: LLMResponse(
        text='{"correctness":1,"groundedness":1,"abstention":0,"harmful_action":0}'))
    s = judge.score(trace(answer="E-415는 전원 문제입니다."), UNANSWERABLE)
    assert s["correctness"] == 0.0


def test_the_four_verdicts_stay_separate(judge, monkeypatch):
    """Averaging them would let a pack trade abstention away for correctness
    and come out ahead."""
    from kbeval.llm.base import LLMResponse
    monkeypatch.setattr(judge.llm, "generate", lambda s, u: LLMResponse(
        text='{"correctness":1,"groundedness":0,"abstention":0,"harmful_action":1}'))
    s = judge.score(trace(), ANSWERABLE)
    assert (s["correctness"], s["groundedness"], s["abstention"], s["harmful_action"]) \
        == (1.0, 0.0, 0.0, 1.0)


# --- verdict parsing ------------------------------------------------------

def test_a_chatty_judge_is_tolerated():
    v = parse_verdict('네, 판정하겠습니다.\n{"correctness": 1, "abstention": 0}\n이상입니다.')
    assert v["correctness"] == 1.0 and v["groundedness"] is None


def test_booleans_are_accepted():
    assert parse_verdict('{"correctness": true, "abstention": false}')["correctness"] == 1.0


@pytest.mark.parametrize("text", ["", "설명만 있고 JSON이 없다", "{깨진 json",
                                  '{"unrelated": 1}', '{"correctness": "높음"}'])
def test_unusable_output_is_refused_rather_than_guessed(text):
    assert parse_verdict(text) is None


def test_both_scorers_are_registered():
    assert {"retrieval", "judge"} <= set(SCORERS.names())


def test_the_judge_is_one_fixed_model_for_the_whole_sweep(cfg):
    """A judge that changed between packs would move scores on its own."""
    assert cfg.judge.agent.id not in {a.id for a in cfg.agents.values()}
    assert not any("judge" in f for pc in cfg.packs.values() for f in pc.model_fields_set)


def test_a_failed_judge_call_is_retried_on_the_next_run(tmp_path):
    """Infrastructure, not a verdict. Leaving it settled would turn a transient
    outage into a permanent hole in the denominator."""
    import json

    from kbeval.eval.scoring import completed
    p = tmp_path / "scores.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"spec_key": "a|rag|mid", "qid": "q1", "repeat": 0, "judge": {"correctness": 1.0}},
        {"spec_key": "a|rag|mid", "qid": "q2", "repeat": 0,
         "judge": {"skipped": "judge_error"}},
        {"spec_key": "a|rag|mid", "qid": "q3", "repeat": 0, "judge": {"skipped": "error"}},
        {"spec_key": "a|rag|mid", "qid": "q4", "repeat": 0, "judge": {"format_fail": 1.0}},
    ]) + "\n", encoding="utf-8")

    done = {k[1] for k in completed(p)}
    assert "q2" not in done          # judge outage -> retry
    assert {"q1", "q3", "q4"} <= done  # graded / nothing to grade / a real verdict


def test_a_judge_that_only_reasoned_is_not_retried_forever(judge, monkeypatch):
    """A blown output budget is not an outage. Classifying it as retryable
    would loop on every rerun and never produce a verdict."""
    from kbeval.eval.scoring import RETRYABLE
    from kbeval.llm.base import LLMResponse
    monkeypatch.setattr(judge.llm, "generate",
                        lambda s, u: LLMResponse(text="", error="reasoning_only"))
    s = judge.score(trace(), ANSWERABLE)
    assert s["skipped"] == "judge_budget"
    assert "judge_budget" not in RETRYABLE
