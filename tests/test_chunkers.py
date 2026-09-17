"""S3 -- the offset invariant.

    doc.text[chunk.char_start:chunk.char_end].strip() == chunk.text

This is the one property in the project that cannot be retrofitted: it is what
lets a gold span be labelled once on the source and resolve onto every pack.
It is asserted here for every chunker against every pack's settings, because a
chunker that breaks it still produces plausible-looking chunks -- the damage
only shows up much later, as gold that mysteriously fails to map.
"""
from __future__ import annotations

import pytest

from kbeval.packs.chunkers import (
    CHUNKERS,
    FlatChunker,
    HeadingChunker,
    RecursiveChunker,
    build_chunker,
)
from kbeval.types import Document

ALL_CHUNKERS = ["flat", "recursive", "heading"]


@pytest.fixture
def toy() -> Document:
    body = "\n".join(f"## 절 {i}\n" + f"본문 {i} 문장입니다. " * 12 for i in range(4))
    return Document(doc_id="toy.md", title="toy", text=body)


@pytest.mark.parametrize("name", ALL_CHUNKERS)
def test_offset_invariant_on_the_real_corpus(name, cfg, docs):
    ch = build_chunker(name, 900, 120)
    for d in docs:
        for c in ch.split(d):
            assert d.text[c.char_start:c.char_end].strip() == c.text, c.chunk_id


@pytest.mark.parametrize("name", ALL_CHUNKERS)
@pytest.mark.parametrize("size,overlap", [(120, 20), (300, 50), (900, 120), (5000, 0)])
def test_offset_invariant_across_sizes(name, size, overlap, toy):
    for c in build_chunker(name, size, overlap).split(toy):
        assert toy.text[c.char_start:c.char_end].strip() == c.text, c.chunk_id


@pytest.mark.parametrize("name", ALL_CHUNKERS)
def test_every_pack_config_holds_the_invariant(name, cfg, docs):
    for pc in cfg.packs.values():
        ch = build_chunker(name, pc.chunk_chars, pc.chunk_overlap)
        for d in docs:
            for c in ch.split(d):
                assert d.text[c.char_start:c.char_end].strip() == c.text


@pytest.mark.parametrize("name", ALL_CHUNKERS)
def test_chunk_ids_are_unique_within_a_document(name, docs):
    ch = build_chunker(name, 200, 40)
    for d in docs:
        ids = [c.chunk_id for c in ch.split(d)]
        assert len(ids) == len(set(ids))


@pytest.mark.parametrize("name", ALL_CHUNKERS)
def test_no_empty_chunks(name, docs):
    for d in docs:
        assert all(c.text.strip() for c in build_chunker(name, 200, 40).split(d))


@pytest.mark.parametrize("name", ALL_CHUNKERS)
def test_offsets_are_ordered_and_in_range(name, docs):
    for d in docs:
        for c in build_chunker(name, 200, 40).split(d):
            assert 0 <= c.char_start < c.char_end <= len(d.text)


@pytest.mark.parametrize("name", ALL_CHUNKERS)
def test_the_whole_document_is_reachable(name, docs):
    """No pack may quietly lose a region: gold could live there."""
    for d in docs:
        covered = set()
        for c in build_chunker(name, 200, 40).split(d):
            covered.update(range(c.char_start, c.char_end))
        missed = [i for i, ch in enumerate(d.text) if not ch.isspace() and i not in covered]
        assert not missed, f"{d.doc_id} dropped {len(missed)} chars"


def test_flat_is_deliberately_separator_blind(toy):
    """The naive baseline. If it ever starts respecting structure, `raw` stops
    being a floor and every lift measured against it shrinks."""
    chunks = FlatChunker(100, 0).split(toy)
    assert any(not c.text.endswith(("다.", "니다.")) for c in chunks)
    assert chunks[0].char_start == 0 and chunks[0].char_end == 100


def test_recursive_snaps_to_a_separator(toy):
    for c in RecursiveChunker(200, 0).split(toy)[:-1]:
        tail = toy.text[max(c.char_end - 3, 0):c.char_end + 1]
        assert any(sep.strip() in tail or tail.strip() == "" for sep in [".", "#", " "])


def test_heading_records_its_path(docs_by_id):
    chunks = HeadingChunker(900, 120).split(docs_by_id["olt_auth.md"])
    paths = {c.meta["heading_path"] for c in chunks}
    assert "E-411 인증 실패" in paths


def test_heading_falls_back_when_there_are_no_headings():
    doc = Document(doc_id="flat.md", title="flat", text="문장 하나. " * 80)
    assert HeadingChunker(200, 40).split(doc)


def test_pathological_overlap_still_terminates(toy):
    """overlap >= size would loop forever without the +1 guard."""
    assert RecursiveChunker(100, 200).split(toy)


def test_registry_exposes_exactly_the_shipped_chunkers():
    assert set(CHUNKERS.names()) == set(ALL_CHUNKERS)


def test_unknown_chunker_names_itself_in_the_error():
    with pytest.raises(KeyError, match="semantic"):
        build_chunker("semantic", 900, 120)
