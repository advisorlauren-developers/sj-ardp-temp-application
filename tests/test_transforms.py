"""S4 -- transforms are the ai-ready factors under test.

Two contracts, and they pull in opposite directions:
  * a transform may rewrite text freely -- that is the whole point;
  * a transform may never touch char_start / char_end -- that is what keeps
    gold mapping automatic.

So the S3 slice invariant is expected to BREAK after a transform, while the
coordinates stay valid. Both halves are asserted here, because a transform that
"helpfully" re-derives offsets from its own output would pass every other test
in the suite and quietly unlabel the corpus.
"""
from __future__ import annotations

import pytest

from kbeval.packs.chunkers import build_chunker
from kbeval.packs.transforms import (
    TRANSFORMS,
    apply_transforms,
    dedup,
    inline_meta,
    normalize_terms,
    strip_boilerplate,
)
from kbeval.types import Chunk

ALL_TRANSFORMS = ["inline_meta", "strip_boilerplate", "normalize_terms", "dedup"]


@pytest.fixture
def chunks(docs):
    ch = build_chunker("heading", 900, 120)
    return [c for d in docs for c in ch.split(d)]


@pytest.mark.parametrize("name", ALL_TRANSFORMS)
def test_transforms_never_move_offsets(name, chunks):
    before = {c.chunk_id: (c.doc_id, c.char_start, c.char_end) for c in chunks}
    for c in TRANSFORMS.get(name)(list(chunks)):
        assert before[c.chunk_id] == (c.doc_id, c.char_start, c.char_end)


@pytest.mark.parametrize("name", ALL_TRANSFORMS)
def test_transforms_do_not_mutate_their_input(name, chunks):
    snapshot = [c.text for c in chunks]
    TRANSFORMS.get(name)(chunks)
    assert [c.text for c in chunks] == snapshot


def test_inline_meta_surfaces_the_breadcrumb(chunks):
    out = {c.chunk_id: c.text for c in inline_meta(chunks)}
    target = next(c for c in chunks if c.meta.get("heading_path") == "E-411 인증 실패")
    assert out[target.chunk_id].startswith("[")
    assert "heading_path: E-411 인증 실패" in out[target.chunk_id]
    assert out[target.chunk_id].endswith(target.text)


def test_inline_meta_breaks_the_slice_invariant_on_purpose(chunks, docs_by_id):
    """Documented consequence: after a transform the text is no longer a slice
    of the source. The offsets still point at where it came from."""
    c = inline_meta(chunks)[0]
    d = docs_by_id[c.doc_id]
    assert d.text[c.char_start:c.char_end].strip() != c.text
    assert d.text[c.char_start:c.char_end].strip() in c.text


def test_strip_boilerplate_removes_document_control_lines(chunks):
    assert any("문서 번호" in c.text for c in chunks)
    assert all("문서 번호" not in c.text for c in strip_boilerplate(chunks))
    assert all("작성자" not in c.text for c in strip_boilerplate(chunks))


def test_strip_boilerplate_keeps_the_answer(chunks):
    out = " ".join(c.text for c in strip_boilerplate(chunks))
    assert "show ont autofind" in out and "-27dBm" in out


def test_normalize_terms_closes_the_user_document_gap():
    """A user types E411; the manual says E-411. To BM25 those are two terms."""
    c = Chunk("x#0", "x", 0, 10, "E411 과 E 412, e_413 모두 같은 코드다.")
    assert normalize_terms([c])[0].text.count("E-4") == 3


def test_normalize_terms_leaves_unrelated_numbers_alone():
    c = Chunk("x#0", "x", 0, 10, "포트 24 와 -27dBm 은 코드가 아니다.")
    assert normalize_terms([c])[0].text == c.text


def test_dedup_collapses_whitespace_variants():
    a = Chunk("a#0", "a", 0, 5, "같은  내용 이다.")
    b = Chunk("b#0", "b", 0, 5, "같은 내용\n이다.")
    assert len(dedup([a, b])) == 1


def test_dedup_can_delete_ground_truth(chunks):
    """The transform that justifies verify_gold_coverage. A duplicated gold
    section survives here only because the ORIGINAL is kept, not the copy --
    reverse the order and the labelled offsets point at a chunk that is gone."""
    gold = next(c for c in chunks if "show ont autofind" in c.text)
    copy = Chunk("other#0", "other.md", 0, len(gold.text), gold.text, {})
    kept = dedup([copy, gold])
    assert len(kept) == 1 and kept[0].chunk_id == "other#0"
    assert gold.chunk_id not in {c.chunk_id for c in kept}


def test_transform_order_is_the_experimenters_choice(chunks):
    """strip_boilerplate AFTER inline_meta would eat the header it just added."""
    good = apply_transforms(list(chunks), ["strip_boilerplate", "inline_meta"])
    assert all(c.text.startswith("[") for c in good if c.meta.get("heading_path"))


def test_apply_transforms_is_a_no_op_when_empty(chunks):
    assert [c.text for c in apply_transforms(list(chunks), [])] == [c.text for c in chunks]


def test_registry_exposes_exactly_the_shipped_transforms():
    assert set(TRANSFORMS.names()) == set(ALL_TRANSFORMS)


def test_unknown_transform_names_itself_in_the_error(chunks):
    with pytest.raises(KeyError, match="summarize"):
        apply_transforms(chunks, ["summarize"])
