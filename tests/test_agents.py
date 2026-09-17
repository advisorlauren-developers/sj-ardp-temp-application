"""S6 -- the three modes, and the equality that realization_rate depends on.

realization_rate = (rag - bare) / (oracle_ref - bare). Every term is a
difference between these agents, so any asymmetry between them that is not the
context itself corrupts the ratio -- silently, and in a way that looks like a
real result. Rule 6 is therefore the first thing asserted here.
"""
from __future__ import annotations

import pytest

from kbeval.agents import AgentDeps, build_agent
from kbeval.config import load_config
from kbeval.llm import build_llm
from kbeval.packs import build_pack
from kbeval.registry import AGENTS
from kbeval.retrieval import build_retriever
from kbeval.types import Question
from kbeval.wiring import build

MODES = ["bare", "rag", "oracle_ref"]


@pytest.fixture
def q(questions):
    return next(x for x in questions if x.qid == "qa_001")


@pytest.fixture
def agents(cfg):
    return {m: build(cfg, "mock_a", m, "mid" if m == "rag" else None) for m in MODES}


# --- rule 6 --------------------------------------------------------------

def test_every_mode_shares_one_system_prompt(agents):
    assert len({a.system for a in agents.values()}) == 1


def test_every_mode_shares_one_user_template(agents):
    assert len({a.template for a in agents.values()}) == 1


def test_rag_and_oracle_ref_differ_only_in_the_context_block(agents, q):
    rag, _ = agents["rag"].render(q)
    oracle, _ = agents["oracle_ref"].render(q)
    ctx_rag = agents["rag"].material(q).text
    ctx_oracle = agents["oracle_ref"].material(q).text
    assert rag.replace(ctx_rag, "<CTX>") == oracle.replace(ctx_oracle, "<CTX>")


def test_prompts_come_from_files_not_literals(agents, cfg):
    """A prompt pasted into a .py is invisible to the config hash, so two
    different experiments would share a coordinate."""
    assert agents["rag"].system == cfg.prompts.read("system_qa")
    assert agents["rag"].template == cfg.prompts.read("context")


def test_editing_a_prompt_moves_the_run_id(cfg, base_yaml, tmp_path):
    before = cfg.hash
    edited = tmp_path / "system.md"
    edited.write_text(cfg.prompts.read("system_qa") + "\n6. 한 줄 더.\n", encoding="utf-8")
    after = load_config([base_yaml], [f"prompts.system_qa={edited}"]).hash
    assert after != before


# --- what each mode injects ----------------------------------------------

def test_bare_injects_nothing(agents, q):
    m = agents["bare"].material(q)
    assert m.chunk_ids == [] and m.tokens == 0
    assert m.text == "(제공된 자료 없음)"


def test_rag_injects_pack_chunks(agents, q):
    m = agents["rag"].material(q)
    assert m.chunk_ids and m.tokens > 0
    assert all(cid.endswith(tuple("0123456789")) for cid in m.chunk_ids)


def test_rag_renders_chunk_ids_so_groundedness_is_checkable(agents, q):
    m = agents["rag"].material(q)
    assert all(cid in m.text for cid in m.chunk_ids)


def test_oracle_ref_injects_the_curated_passage(agents, q):
    assert agents["oracle_ref"].material(q).text == q.oracle_ref


def test_oracle_ref_is_identical_for_every_pack(cfg, q):
    """Rule 5. A pack-specific ceiling would let a bad pack lower its own
    denominator and score a flattering realization rate."""
    texts = {build(cfg, "mock_a", "oracle_ref", None).material(q).text for _ in cfg.packs}
    assert len(texts) == 1


def test_oracle_ref_degenerates_to_bare_when_there_is_no_reference(agents, questions):
    """An unanswerable question has no ceiling to reach; it is scored on
    abstention instead."""
    unanswerable = next(x for x in questions if not x.answerable)
    assert agents["oracle_ref"].material(unanswerable).text == "(제공된 자료 없음)"


# --- budget and accounting ------------------------------------------------

def test_the_budget_is_what_limits_injection(cfg, base_yaml, q):
    tight = load_config([base_yaml], ["fixed.context_token_budget=120"])
    m = build(tight, "mock_a", "rag", "mid").material(q)
    assert m.tokens <= 120
    assert m.skipped, "nothing was skipped, so the control never bound"


def test_skipped_chunks_are_recorded_not_dropped_silently(cfg, base_yaml, q):
    tight = load_config([base_yaml], ["fixed.context_token_budget=120"])
    m = build(tight, "mock_a", "rag", "mid").material(q)
    assert set(m.chunk_ids) | set(m.skipped) == {h.chunk_id for h in m.retrieved}


def test_injected_chunks_arrive_whole(cfg, base_yaml, q):
    tight = load_config([base_yaml], ["fixed.context_token_budget=120"])
    a = build(tight, "mock_a", "rag", "mid")
    m = a.material(q)
    for cid in m.chunk_ids:
        assert a.deps.pack.get(cid).text in m.text


def test_rag_falls_back_to_no_material_rather_than_half_a_chunk(cfg, base_yaml, q):
    starved = load_config([base_yaml], ["fixed.context_token_budget=1"])
    m = build(starved, "mock_a", "rag", "mid").material(q)
    assert m.chunk_ids == [] and m.text == "(제공된 자료 없음)"


def test_the_result_carries_its_covariates(agents, q):
    r = agents["rag"].answer(q)
    assert r.injected_tokens > 0
    assert len(r.injected_chunk_ids) == len(set(r.injected_chunk_ids))
    assert r.retrieved and r.latency_ms >= 0


def test_an_error_is_never_reported_as_an_answer(agents, q, monkeypatch):
    monkeypatch.setattr(agents["bare"].deps.llm, "generate",
                        lambda s, u: __import__("kbeval.llm", fromlist=["x"]).validate("", 1.0, {}))
    r = agents["bare"].answer(q)
    assert r.error == "empty_response" and r.text == ""


# --- wiring boundaries ----------------------------------------------------

def test_pack_free_modes_refuse_a_pack(cfg):
    for mode in ("bare", "oracle_ref"):
        with pytest.raises(ValueError, match="pack-independent"):
            build(cfg, "mock_a", mode, "mid")


def test_rag_refuses_to_run_without_a_pack(cfg):
    with pytest.raises(ValueError, match="needs a pack"):
        build(cfg, "mock_a", "rag", None)


def test_rag_refuses_an_unindexed_retriever(cfg):
    deps = AgentDeps(llm=build_llm(cfg.agents["mock_a"], cfg.llm),
                     prompts=cfg.prompts, fixed=cfg.fixed,
                     pack=build_pack(cfg, "mid"), retriever=None)
    with pytest.raises(ValueError):
        build_agent("rag", deps)


def test_the_retriever_is_indexed_on_the_pack_it_was_given(cfg, q):
    a = build(cfg, "mock_a", "rag", "raw")
    assert {h.chunk_id for h in a.material(q).retrieved} <= {c.chunk_id for c in a.deps.pack.chunks}


def test_all_three_modes_are_registered():
    assert set(AGENTS.names()) == set(MODES)


def test_a_question_with_no_gold_still_answers(cfg, questions):
    """Unanswerable questions run through every mode -- that is the whole point
    of keeping them in the set."""
    unanswerable = next(x for x in questions if not x.answerable)
    for mode in MODES:
        r = build(cfg, "mock_a", mode, "mid" if mode == "rag" else None).answer(unanswerable)
        assert r.error is None


def test_two_models_see_exactly_the_same_prompt(cfg, q):
    """Any prompt difference between models would confound the model axis with
    the pack axis."""
    a = build(cfg, "mock_a", "rag", "mid").render(q)[0]
    b = build(cfg, "mock_b", "rag", "mid").render(q)[0]
    assert a == b


def test_the_retriever_is_the_same_algorithm_for_every_pack(cfg):
    kinds = {type(build_retriever(cfg.retrieval)).__name__ for _ in cfg.packs}
    assert len(kinds) == 1


def test_oracle_ref_fits_the_budget_by_construction(cfg, questions):
    """If a curated passage did not fit, the ceiling would be measured on a
    truncated reference and headroom would be understated."""
    from kbeval.budget import estimate_tokens
    for x in questions:
        if x.oracle_ref:
            assert estimate_tokens(x.oracle_ref, cfg.fixed.chars_per_token) \
                   <= cfg.fixed.context_token_budget


def test_unknown_mode_names_itself(cfg):
    q_ = Question(qid="x", intent="qa", query="?", answerable=True)
    assert q_.qid == "x"
    with pytest.raises(KeyError, match="oracle_pack"):
        build(cfg, "mock_a", "oracle_pack", None)
