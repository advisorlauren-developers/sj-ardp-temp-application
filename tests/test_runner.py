"""S7 -- the runner, judged by what survives a failure.

At this scale a mid-run failure is certain, so the properties under test are
not about throughput: a finished unit must never be redone, a torn write must
never corrupt its neighbour, and a pack that lost its gold must stop the sweep
before any of it is spent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kbeval.config import load_config
from kbeval.runner import (
    RunPaths,
    completed_keys,
    plan_tasks,
    read_traces,
    run_sweep,
)
from kbeval.sweep import expand_sweep


@pytest.fixture
def cfg_out(base_yaml, tmp_path):
    """Every run writes inside tmp_path, so tests can never append to a real run."""
    return load_config([base_yaml], [f"run.out_dir={tmp_path / 'runs'}"])


@pytest.fixture
def summary(cfg_out):
    return run_sweep(cfg_out)


# --- shape ---------------------------------------------------------------

def test_one_row_per_spec_question_repeat(summary, cfg_out, questions):
    assert summary.planned == len(expand_sweep(cfg_out)) * len(questions) * cfg_out.fixed.n_repeat
    assert summary.written == summary.planned


def test_the_sweep_is_not_a_cross_product(summary):
    assert summary.specs == 10


def test_every_row_is_unique(summary):
    rows, _ = read_traces(RunPaths(summary.root).traces)
    keys = [(r["spec_key"], r["qid"], r["repeat"]) for r in rows]
    assert len(keys) == len(set(keys))


def test_every_row_carries_its_coordinates(summary):
    rows, _ = read_traces(RunPaths(summary.root).traces)
    assert {r["config_hash"] for r in rows} == {summary.run_id.rsplit("-", 1)[1]}
    for r in rows:
        assert {"spec_key", "qid", "repeat", "agent", "mode", "pack", "query",
                "answer", "injected_tokens", "gold_chunk_ids"} <= set(r)


def test_covariates_are_recorded_for_the_report(summary):
    rows, _ = read_traces(RunPaths(summary.root).traces)
    rag = [r for r in rows if r["mode"] == "rag"]
    assert all("injected_tokens" in r and "injected_chunk_ids" in r for r in rag)
    assert all("skipped_chunk_ids" in r for r in rag)
    assert any(r["gold_chunk_ids"] for r in rag)


def test_pack_free_modes_record_no_pack(summary):
    rows, _ = read_traces(RunPaths(summary.root).traces)
    for r in rows:
        if r["mode"] in ("bare", "oracle_ref"):
            assert r["pack"] is None and r["injected_chunk_ids"] == []


# --- resume --------------------------------------------------------------

def test_a_finished_run_redoes_nothing(cfg_out, summary):
    again = run_sweep(cfg_out)
    assert again.resumed == summary.planned and again.written == 0


def test_resume_picks_up_exactly_what_is_missing(cfg_out, summary):
    traces = RunPaths(summary.root).traces
    lines = traces.read_text(encoding="utf-8").splitlines()
    traces.write_text("\n".join(lines[:25]) + "\n", encoding="utf-8")

    again = run_sweep(cfg_out)
    assert again.resumed == 25
    assert again.written == summary.planned - 25

    rows, _ = read_traces(traces)
    assert len({(r["spec_key"], r["qid"], r["repeat"]) for r in rows}) == summary.planned


def test_a_torn_final_line_does_not_corrupt_its_successor(cfg_out, summary):
    """A crash mid-write leaves a row with no newline. Appending after it would
    splice two rows into one that no longer looks broken."""
    traces = RunPaths(summary.root).traces
    lines = traces.read_text(encoding="utf-8").splitlines()
    traces.write_text("\n".join(lines[:20]) + "\n" + lines[20][:40], encoding="utf-8")

    run_sweep(cfg_out)
    rows, bad = read_traces(traces)
    assert bad == 1                                   # the torn tail, isolated
    assert len({(r["spec_key"], r["qid"], r["repeat"]) for r in rows}) == summary.planned


def test_lost_lines_are_counted_not_silently_dropped(tmp_path):
    p = tmp_path / "traces.jsonl"
    p.write_text('{"a":1}\n{"broken\n{"b":2}\n', encoding="utf-8")
    rows, bad = read_traces(p)
    assert len(rows) == 2 and bad == 1


def test_completed_keys_tolerates_a_broken_line(tmp_path):
    p = tmp_path / "traces.jsonl"
    p.write_text('{"spec_key":"s","qid":"q","repeat":0}\n{"trunc\n', encoding="utf-8")
    assert completed_keys(p) == {("s", "q", 0)}


def test_no_resume_starts_over(cfg_out, summary):
    again = run_sweep(cfg_out, resume=False)
    assert again.written == summary.planned


def test_limit_is_per_spec_not_total(cfg_out):
    s = run_sweep(cfg_out, limit=1)
    assert s.planned == s.specs          # one question for each spec


# --- the gate ------------------------------------------------------------

def test_a_pack_that_lost_gold_stops_the_sweep(base_yaml, tmp_path):
    """Rule 9. By report time a missing answer is indistinguishable from a
    model that got worse, and by then the money is spent."""
    cfg = load_config([base_yaml], [f"run.out_dir={tmp_path / 'runs'}",
                                    'source.params={"glob":"bng_*.md"}'])
    with pytest.raises(RuntimeError, match="gold missing"):
        run_sweep(cfg)


def test_the_gate_leaves_evidence_but_no_traces(base_yaml, tmp_path):
    cfg = load_config([base_yaml], [f"run.out_dir={tmp_path / 'runs'}",
                                    'source.params={"glob":"bng_*.md"}'])
    with pytest.raises(RuntimeError):
        run_sweep(cfg)
    paths = RunPaths(Path(cfg.run.out_dir) / cfg.run_id)
    assert paths.coverage.exists()
    assert not paths.traces.exists()


# --- reproducibility artefacts -------------------------------------------

def test_the_run_dir_carries_everything_needed_to_recompute_the_hash(summary, cfg_out):
    paths = RunPaths(summary.root)
    assert paths.resolved.exists() and paths.coverage.exists() and paths.traces.exists()
    assert load_config([paths.resolved]).hash == cfg_out.hash


def test_coverage_is_recorded_for_every_pack(summary, cfg_out):
    report = json.loads(RunPaths(summary.root).coverage.read_text(encoding="utf-8"))
    assert set(report) == set(cfg_out.sweep.pack)
    assert all(r["coverage"] == 1.0 for r in report.values())


def test_the_run_id_separates_different_settings(base_yaml, tmp_path):
    a = load_config([base_yaml], [f"run.out_dir={tmp_path}"])
    b = load_config([base_yaml], [f"run.out_dir={tmp_path}",
                                  "fixed.context_token_budget=500"])
    assert run_sweep(a).root != run_sweep(b).root


def test_operational_settings_do_not_fork_the_run(base_yaml, tmp_path):
    """Raising the thread count must not invalidate comparison with everything
    measured before it."""
    a = load_config([base_yaml], [f"run.out_dir={tmp_path}"])
    b = load_config([base_yaml], [f"run.out_dir={tmp_path}", "run.max_concurrency=1"])
    assert a.hash == b.hash


# --- concurrency ----------------------------------------------------------

def test_concurrency_does_not_change_the_result(base_yaml, tmp_path):
    one = load_config([base_yaml], [f"run.out_dir={tmp_path / 'a'}", "run.max_concurrency=1"])
    many = load_config([base_yaml], [f"run.out_dir={tmp_path / 'b'}", "run.max_concurrency=8"])
    ra, _ = read_traces(RunPaths(run_sweep(one).root).traces)
    rb, _ = read_traces(RunPaths(run_sweep(many).root).traces)

    def norm(rows):
        return sorted((r["spec_key"], r["qid"], r["answer"]) for r in rows)

    assert norm(ra) == norm(rb)


def test_a_worker_crash_is_not_swallowed(cfg_out, monkeypatch):
    from kbeval.llm import mock as mock_mod
    monkeypatch.setattr(mock_mod.MockLLM, "generate",
                        lambda self, s, u: (_ for _ in ()).throw(ZeroDivisionError("boom")))
    with pytest.raises(ZeroDivisionError):
        run_sweep(cfg_out)


def test_provider_errors_are_counted_not_raised(cfg_out, monkeypatch):
    """An outage must finish the sweep and show up as `errors`, never as a
    wrong answer and never as a crash."""
    from kbeval.llm import mock as mock_mod
    from kbeval.llm.base import LLMResponse
    monkeypatch.setattr(mock_mod.MockLLM, "generate",
                        lambda self, s, u: LLMResponse(text="", error="throttled"))
    s = run_sweep(cfg_out)
    assert s.errors == s.written and s.error_samples == ["throttled"]

    rows, _ = read_traces(RunPaths(s.root).traces)
    assert all(r["error"] == "throttled" and r["answer"] == "" for r in rows)


def test_planning_is_independent_of_execution(cfg_out, questions):
    tasks = plan_tasks(cfg_out, expand_sweep(cfg_out), questions)
    assert len(tasks) == len(expand_sweep(cfg_out)) * len(questions)
