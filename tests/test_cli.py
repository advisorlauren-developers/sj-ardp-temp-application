"""The confirm-commands themselves.

Each stage is supposed to add one runnable check, which is only true if the
command still runs. These assert the documented S1-S4 output and, more
importantly, the two places the CLI must FAIL: a broken offset invariant and a
pack that lost gold. A gate that exits 0 is not a gate.
"""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from kbeval.cli import app

runner = CliRunner()


def run(*args):
    return runner.invoke(app, list(args), catch_exceptions=False)


def test_plan_reports_the_non_cross_product():
    r = run("plan")
    assert r.exit_code == 0
    assert "specs       : 10" in r.stdout
    assert "cross product would be 18" in r.stdout


def test_plan_stamps_a_hash():
    r = run("plan")
    assert "config_hash : " in r.stdout
    assert "run_id      : qa-poc-" in r.stdout


def test_docs_reports_the_coordinate_space():
    r = run("docs")
    assert r.exit_code == 0
    assert "docs=2" in r.stdout
    assert "bng_qos.md" in r.stdout and "olt_auth.md" in r.stdout


def test_chunks_asserts_the_offset_invariant():
    r = run("chunks", "mid")
    assert r.exit_code == 0
    assert "offset invariant: OK" in r.stdout
    assert "chunks=6" in r.stdout


@pytest.mark.parametrize("name,n", [("raw", 2), ("mid", 6), ("full", 6)])
def test_pack_builds_and_reports_full_coverage(name, n):
    r = run("pack", name)
    assert r.exit_code == 0
    assert "coverage=1.0" in r.stdout
    assert f"chunks={n}" in r.stdout


def test_pack_exits_nonzero_when_gold_is_gone(tmp_path):
    """Rule 9. A missing answer must stop the pipeline here -- downstream it is
    indistinguishable from the model getting worse."""
    q = tmp_path / "q.jsonl"
    q.write_text(json.dumps({
        "qid": "qa_x", "intent": "qa", "answerable": True, "query": "?",
        "gold": {"doc_id": "olt_auth.md", "char_start": 0, "char_end": 40},
        "gold_answer": "x", "oracle_ref": "x",
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    ok = run("pack", "mid", "-s", f"eval.questions={q}")
    assert ok.exit_code == 0

    # same span, but no chunk can host it once the document is not in the pack
    lost = run("pack", "mid", "-s", f"eval.questions={q}",
               "-s", 'source.params={"glob":"bng_*.md"}')
    assert lost.exit_code == 1
    assert "WARNING gold missing" in lost.stdout


def test_a_config_typo_stops_every_command():
    """The failure that took S1-S4 down: an undeclared block is an error, not a
    silent default."""
    r = runner.invoke(app, ["plan", "-s", "fixed.context_token_budgets=4000"])
    assert r.exit_code != 0


def test_unknown_pack_name_is_a_clean_error():
    r = runner.invoke(app, ["pack", "ghost"])
    assert r.exit_code != 0


# --- S5 ------------------------------------------------------------------

def test_retrieval_eval_reports_every_pack_with_its_covariates():
    r = run("retrieval-eval")
    assert r.exit_code == 0
    assert "hit@1" in r.stdout and "mrr" in r.stdout
    for pack in ("raw", "mid", "full"):
        assert pack in r.stdout
    assert "questions=5" in r.stdout        # unanswerable pair excluded


def test_retrieval_eval_detail_shows_where_gold_landed():
    r = run("retrieval-eval", "--detail")
    assert r.exit_code == 0
    assert "gold_rank=" in r.stdout


def test_search_marks_the_gold_chunk():
    r = run("search", "E-411 조치할 때 장비 리부팅 해도 되나요?", "--pack", "mid")
    assert r.exit_code == 0
    assert "*" in r.stdout                  # the labelled chunk was found
    assert "gold  : ['olt_auth.md" in r.stdout


def test_search_accepts_a_qid_instead_of_retyping_the_query():
    r = run("search", "", "--qid", "qa_003", "--pack", "mid")
    assert r.exit_code == 0
    assert "광 수신 레벨" in r.stdout


def test_search_on_an_unlabelled_query_still_ranks():
    r = run("search", "라이선스", "--pack", "full")
    assert r.exit_code == 0
    assert "no labelled question matched" in r.stdout


# --- S6 ------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["bare", "rag", "oracle_ref"])
def test_ask_runs_every_mode(mode):
    r = run("ask", "qa_001", "--mode", mode, "--pack", "mid")
    assert r.exit_code == 0
    assert "answer  :" in r.stdout


def test_ask_shows_what_the_budget_did():
    r = run("ask", "qa_001", "--mode", "rag", "--pack", "mid",
            "-s", "fixed.context_token_budget=120")
    assert "of 120" in r.stdout and "skipped=" in r.stdout


def test_ask_can_print_the_prompt_it_sent():
    r = run("ask", "qa_001", "--mode", "rag", "--pack", "mid", "--show-prompt")
    assert "--- system ---" in r.stdout and "--- user ---" in r.stdout
    assert "[참고 자료]" in r.stdout


def test_bare_gets_no_material_in_its_prompt():
    r = run("ask", "qa_001", "--mode", "bare", "--show-prompt")
    assert "(제공된 자료 없음)" in r.stdout


def test_ask_rejects_an_unknown_qid():
    r = runner.invoke(app, ["ask", "qa_999"])
    assert r.exit_code == 1


def test_ask_reports_an_error_without_calling_it_an_answer(monkeypatch):
    from kbeval.llm import mock as mock_mod
    monkeypatch.setattr(mock_mod.MockLLM, "generate",
                        lambda self, s, u: __import__("kbeval.llm", fromlist=["x"]).validate("", 1.0, {}))
    r = runner.invoke(app, ["ask", "qa_001", "--mode", "bare"])
    assert r.exit_code == 1
    assert "NOT counted as an answer" in r.stdout


# --- S7 ------------------------------------------------------------------

def test_run_executes_the_whole_sweep(tmp_path):
    r = run("run", "-s", f"run.out_dir={tmp_path}")
    assert r.exit_code == 0
    assert "specs   : 10" in r.stdout and "written=70" in r.stdout
    assert "errors  : 0" in r.stdout


def test_run_is_resumable(tmp_path):
    run("run", "-s", f"run.out_dir={tmp_path}")
    again = run("run", "-s", f"run.out_dir={tmp_path}")
    assert "resumed=70" in again.stdout and "written=0" in again.stdout


def test_run_limit_smoke_tests_one_question_per_spec(tmp_path):
    r = run("run", "--limit", "1", "-s", f"run.out_dir={tmp_path}")
    assert "planned=10" in r.stdout


def test_run_refuses_to_start_on_a_pack_that_lost_gold(tmp_path):
    r = runner.invoke(app, ["run", "-s", f"run.out_dir={tmp_path}",
                            "-s", 'source.params={"glob":"bng_*.md"}'])
    assert r.exit_code == 1
    assert "refusing to run" in r.stdout


# --- S8 ------------------------------------------------------------------

def test_score_then_report(tmp_path):
    out = ["-s", f"run.out_dir={tmp_path}"]
    assert run("run", *out).exit_code == 0

    scored = run("score", *out)
    assert scored.exit_code == 0 and "scored=70" in scored.stdout

    rep = run("report", *out)
    assert rep.exit_code == 0
    assert "## realization" in rep.stdout and "## kendall tau" in rep.stdout
    assert "realiz" in rep.stdout and "gold_inj" in rep.stdout


def test_score_is_resumable(tmp_path):
    out = ["-s", f"run.out_dir={tmp_path}"]
    run("run", *out)
    run("score", *out)
    again = run("score", *out)
    assert "scored=0" in again.stdout and "resumed=70" in again.stdout


def test_report_on_a_missing_run_fails_cleanly(tmp_path):
    r = runner.invoke(app, ["report", "-s", f"run.out_dir={tmp_path}"])
    assert r.exit_code == 1 and "no such run" in r.stdout


def test_report_before_scoring_says_what_to_do(tmp_path):
    out = ["-s", f"run.out_dir={tmp_path}"]
    run("run", *out)
    r = run("report", *out)
    assert "kbeval score" in r.stdout
