"""The eval set itself is an experimental artefact, so it gets tests too.

Two failure modes here are silent and fatal:
  * gold offsets drift after a document edit -- every downstream number stays
    plausible and is wrong;
  * the unanswerable share slips to zero -- injecting knowledge then wins by
    construction, because nothing can penalise a pack for encouraging confident
    invention.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from kbeval.packs import load_questions


def test_qids_are_unique(questions):
    ids = [q.qid for q in questions]
    assert len(ids) == len(set(ids))


def test_unanswerable_share_stays_in_the_designed_band(questions):
    share = sum(1 for q in questions if not q.answerable) / len(questions)
    assert 0.15 <= share <= 0.35, f"{share:.0%} unanswerable"


def test_gold_spans_are_inside_their_document(questions, docs_by_id):
    for q in questions:
        if q.gold is None:
            continue
        d = docs_by_id[q.gold.doc_id]
        assert 0 <= q.gold.char_start < q.gold.char_end <= len(d.text), q.qid


def _bigrams(s: str) -> set[str]:
    s = re.sub(r"\W+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def test_gold_spans_still_point_at_their_answer(questions, docs_by_id):
    """The drift detector, in content terms.

    gold_answer is a paraphrase, so exact containment is the wrong test. What
    must hold is that every identifier the answer names (E-411, autofind, CoA,
    -27dBm, 90) is inside the span, and that the two still overlap heavily as
    text. A span that slid onto the neighbouring section keeps neither.
    """
    for q in questions:
        if q.gold is None or not q.gold_answer:
            continue
        span = docs_by_id[q.gold.doc_id].text[q.gold.char_start:q.gold.char_end]
        assert span.strip(), q.qid

        ids = set(re.findall(r"[A-Za-z][A-Za-z0-9]{2,}|\d+", q.gold_answer))
        assert not (ids - {t for t in ids if t in span}), f"{q.qid}: identifiers drifted"

        a = _bigrams(q.gold_answer)
        assert len(a & _bigrams(span)) / len(a) >= 0.6, f"{q.qid}: span moved off the answer"


def test_answerable_questions_carry_both_ceilings(questions):
    for q in questions:
        if q.answerable:
            assert q.gold is not None, q.qid
            assert q.gold_answer, q.qid          # what the judge grades against
            assert q.oracle_ref, q.qid           # what oracle_ref injects


def test_unanswerable_questions_carry_nothing_to_leak(questions):
    for q in questions:
        if not q.answerable:
            assert (q.gold, q.gold_answer, q.oracle_ref) == (None, None, None), q.qid


def test_oracle_ref_is_pack_independent(cfg):
    """Rule 5. oracle_ref lives on the question, not in any pack directory --
    a pack-specific ceiling would let a bad pack lower its own denominator and
    score a flattering realization rate."""
    raw = [json.loads(ln) for ln
           in Path(cfg.eval.questions).read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert all("pack" not in r for r in raw)
    assert all(isinstance(r.get("oracle_ref"), (str, type(None))) for r in raw)


def test_unanswerable_queries_are_not_answerable_from_the_corpus(questions, docs_by_id):
    """A distractor that the corpus does actually answer is not a distractor."""
    corpus = " ".join(d.text for d in docs_by_id.values())
    for q in questions:
        if not q.answerable:
            assert "E-415" not in corpus if "E-415" in q.query else True
            assert "4.0" not in corpus if "4.0" in q.query else True


def test_regenerating_the_question_set_is_a_no_op(cfg, tmp_path):
    """Offsets must be DERIVED. Re-running the generator against unchanged
    documents has to reproduce the committed file byte for byte; if a number
    was ever hand-typed, this is where it shows up."""
    out = Path(cfg.eval.questions)
    before = out.read_text(encoding="utf-8")
    r = subprocess.run([sys.executable, "scripts/make_questions.py"],
                       capture_output=True, text=True, check=False)
    after = out.read_text(encoding="utf-8")
    assert r.returncode == 0, r.stderr
    assert after == before, "questions.jsonl is not reproducible from its anchors"


def test_loader_skips_blank_lines(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text('{"qid":"a","intent":"qa","query":"?","answerable":false}\n\n',
                 encoding="utf-8")
    assert len(load_questions(p)) == 1
