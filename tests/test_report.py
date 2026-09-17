"""S8 -- the report, judged on what it refuses to say.

A reporting layer fails quietly: it prints a number, the number looks fine, and
nothing downstream disagrees. So most of these tests are about the cases where
the honest output is a dash and a reason.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kbeval.config import load_config
from kbeval.eval import build_report, render, score_run
from kbeval.eval.report import RealizationRow
from kbeval.runner import run_sweep


@pytest.fixture
def scored(base_yaml, tmp_path):
    cfg = load_config([base_yaml], [f"run.out_dir={tmp_path / 'runs'}"])
    root = run_sweep(cfg).root
    score_run(cfg, root)
    return cfg, root


@pytest.fixture
def rep(scored):
    cfg, root = scored
    return build_report(cfg, root)


def row(rep, agent, pack) -> RealizationRow:
    return next(r for r in rep.realization if (r.agent, r.pack) == (agent, pack))


# --- the arithmetic -------------------------------------------------------

def test_realization_is_lift_over_headroom(rep):
    for r in rep.realization:
        if r.realization is None:
            continue
        assert r.headroom == pytest.approx(r.oracle - r.bare)
        assert r.lift == pytest.approx(r.rag - r.bare)
        assert r.realization == pytest.approx(r.lift / r.headroom)


def test_every_model_pack_combination_is_reported(rep, scored):
    cfg, _ = scored
    assert len(rep.realization) == len(cfg.sweep.agent) * len(cfg.sweep.pack)


def test_unanswerable_questions_stay_out_of_realization(rep, questions):
    answerable = sum(1 for q in questions if q.answerable)
    assert all(r.n == answerable for r in rep.realization)


def test_abstention_is_reported_on_the_unanswerable_ones(rep, questions):
    unanswerable = sum(1 for q in questions if not q.answerable)
    assert rep.abstention
    assert all(a["n"] <= unanswerable for a in rep.abstention)


# --- what it refuses to compute ------------------------------------------

def test_a_tiny_headroom_suppresses_realization(base_yaml, tmp_path):
    """A denominator near zero turns noise into a spectacular-looking ratio."""
    cfg = load_config([base_yaml], [f"run.out_dir={tmp_path / 'runs'}",
                                    "scoring.min_headroom=2.0"])
    root = run_sweep(cfg).root
    score_run(cfg, root)
    rep = build_report(cfg, root)
    assert all(r.realization is None for r in rep.realization)
    assert all("headroom<" in r.note for r in rep.realization)


def test_a_negative_realization_is_kept_and_labelled():
    """Context that confuses a model is common at 30B and is itself the
    finding. Dropping it would make every pack look non-harmful."""
    from kbeval.eval.report import Report, _fmt
    r = RealizationRow(agent="m", pack="p", bare=0.6, rag=0.2, oracle=1.0,
                       headroom=0.4, lift=-0.4, realization=-1.0, n=5,
                       note="negative: context made it worse (kept on purpose)")
    rep = Report(run_id="x", config_hash="y", n_traces=1, malformed=0,
                 errors=0, format_fails=0, realization=[r])
    text = render(rep)
    assert "-1.000" in text and "kept on purpose" in text
    assert _fmt(None).strip() == "-"


def test_tau_is_undefined_rather_than_invented_when_packs_tie(rep):
    for t in rep.tau:
        if t["tau"] is None:
            assert "undefined" in t["note"]


def test_an_unscored_run_says_so(base_yaml, tmp_path):
    cfg = load_config([base_yaml], [f"run.out_dir={tmp_path / 'runs'}"])
    rep = build_report(cfg, run_sweep(cfg).root)
    assert rep.realization == []
    assert any("kbeval score" in n for n in rep.notes)


# --- covariates and health ------------------------------------------------

def test_covariates_sit_next_to_every_retrieval_number(rep):
    """A pack with more, smaller chunks is easier to hit and cheaper to
    inject. Without these the hit@k column is unreadable."""
    for e in rep.retrieval:
        assert {"n_chunks", "avg_chunk_chars", "injected_tokens", "injected_chunks"} <= set(e)
        assert e["n_chunks"] > 0


def test_health_counts_errors_and_parse_failures(rep):
    assert rep.errors == 0 and rep.format_fails == 0 and rep.malformed == 0


def test_provider_errors_do_not_become_wrong_answers(base_yaml, tmp_path, monkeypatch):
    from kbeval.llm import mock as mock_mod
    from kbeval.llm.base import LLMResponse
    cfg = load_config([base_yaml], [f"run.out_dir={tmp_path / 'runs'}"])
    monkeypatch.setattr(mock_mod.MockLLM, "generate",
                        lambda self, s, u: LLMResponse(text="", error="throttled"))
    root = run_sweep(cfg).root
    score_run(cfg, root)
    rep = build_report(cfg, root)

    assert rep.errors == rep.n_traces
    assert all(r.bare is None and r.rag is None for r in rep.realization)
    assert all(r.realization is None for r in rep.realization)


def test_the_rendered_report_names_every_section(rep):
    text = render(rep)
    for section in ("realization", "retrieval", "abstention", "kendall tau"):
        assert f"## {section}" in text or f"## {section} " in text


def test_scoring_is_resumable(scored):
    cfg, root = scored
    again = score_run(cfg, root)
    assert again.scored == 0 and again.resumed == again.rows


def test_rescoring_does_not_need_the_candidates_again(scored, monkeypatch):
    """The reason scores live in their own file: a judge prompt can be revised
    and the run re-scored without paying for one candidate call."""
    cfg, root = scored
    from kbeval.llm import mock as mock_mod
    monkeypatch.setattr(mock_mod.MockLLM, "generate",
                        lambda self, s, u: (_ for _ in ()).throw(AssertionError("candidate called")))
    score_run(cfg, root)          # resumes; must not touch the candidate model


def test_traces_are_never_rewritten(scored):
    cfg, root = scored
    before = (root / "traces.jsonl").read_bytes()
    score_run(cfg, root, resume=False)
    assert (root / "traces.jsonl").read_bytes() == before
    assert (root / "scores.jsonl").exists()


def test_the_run_dir_holds_everything_a_reader_needs(scored):
    _, root = scored
    assert {p.name for p in Path(root).iterdir()} == {
        "traces.jsonl", "scores.jsonl", "config.resolved.yaml",
        "scoring.resolved.yaml", "gold_coverage.json"}


def test_the_scores_record_which_judge_produced_them(scored):
    """The judge is a control variable. Scores whose grader is unrecorded
    cannot be compared with anything later."""
    from kbeval.config import load_config
    cfg, root = scored
    assert load_config([root / "scoring.resolved.yaml"]).judge.agent.id == cfg.judge.agent.id


def test_resuming_under_a_different_judge_is_refused(scored, base_yaml, tmp_path):
    """Half the file graded by one model and half by another, with no column
    saying which -- a pack difference would then be a grader difference."""
    from kbeval.config import load_config
    cfg, root = scored
    other = load_config([base_yaml], [f"run.out_dir={cfg.run.out_dir}",
                                      'judge={"agent":{"provider":"mock_judge","id":"other-judge"}}'])
    with pytest.raises(RuntimeError, match="not"):
        score_run(other, root)


def test_starting_over_replaces_the_scores_instead_of_appending(scored):
    """Appending would leave two verdicts per row and let whichever landed last
    silently win."""
    cfg, root = scored
    before = len((root / "scores.jsonl").read_text(encoding="utf-8").splitlines())
    score_run(cfg, root, resume=False)
    after = len((root / "scores.jsonl").read_text(encoding="utf-8").splitlines())
    assert after == before


def test_a_retried_row_is_counted_once(scored):
    """A retried judge call appends a second row for the same key. Counting
    both would double it into every mean, hit@k included."""
    from kbeval.eval import load_scores
    from kbeval.eval.scoring import scores_path
    _, root = scored
    path = scores_path(root)

    before = load_scores(root)
    first = path.read_text(encoding="utf-8").splitlines()[0]
    with path.open("a", encoding="utf-8") as fh:
        fh.write(first + "\n")                      # simulate a retry landing

    after = load_scores(root)
    assert len(after) == len(before)


def test_the_later_row_wins(scored):
    import json

    from kbeval.eval import load_scores
    from kbeval.eval.scoring import scores_path
    _, root = scored
    path = scores_path(root)

    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    row["judge"] = {"correctness": 0.5, "format_fail": 0.0}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    got = next(r for r in load_scores(root)
               if (r["spec_key"], r["qid"], r["repeat"])
               == (row["spec_key"], row["qid"], row["repeat"]))
    assert got["judge"]["correctness"] == 0.5


def test_realization_above_one_is_labelled(base_yaml, tmp_path):
    """The pack beat the curated reference. Usually that means the reference is
    one paragraph while retrieval hands over the whole section -- a result
    about the ceiling, not an arithmetic error."""
    from kbeval.eval.report import Report, render
    r = RealizationRow(agent="m", pack="p", bare=0.0, rag=1.0, oracle=0.8,
                       headroom=0.8, lift=1.0, realization=1.25, n=5,
                       note="above 1: rag beat oracle_ref — check the reference is complete")
    text = render(Report(run_id="x", config_hash="y", n_traces=1, malformed=0,
                         errors=0, format_fails=0, realization=[r]))
    assert "1.250" in text and "above 1" in text


def test_an_axis_that_did_not_move_says_so(rep):
    """Every pack scoring identically is a property of the QUESTION SET, not a
    verdict on preprocessing. Printing the table without saying that invites
    the reader to conclude preprocessing does not matter."""
    identical = all(len({r.rag for r in rep.realization if r.agent == a}) == 1
                    for a in {r.agent for r in rep.realization})
    if identical:
        assert any("did not discriminate" in n for n in rep.notes)


def test_long_model_names_do_not_collide_with_the_next_column(base_yaml, tmp_path):
    from kbeval.eval.report import Report, render
    rows = [RealizationRow(agent="nvidia-nemotron-nano-3-30b", pack="raw", bare=0.0,
                           rag=1.0, oracle=1.0, headroom=1.0, lift=1.0,
                           realization=1.0, n=5)]
    line = next(x for x in render(Report(run_id="x", config_hash="y", n_traces=1,
                                         malformed=0, errors=0, format_fails=0,
                                         realization=rows)).splitlines()
                if x.startswith("nvidia-"))
    assert " raw" in line or "raw " in line
    assert "nvidia-nemotron-nano-3-30braw" not in line
