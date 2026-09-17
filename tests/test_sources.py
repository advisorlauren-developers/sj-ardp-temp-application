"""S2 -- the adapter defines the coordinate space.

Whatever lands in Document.text is what a gold span can address and what
retrieval can ever reach. Anything the adapter drops is invisible to the rest
of the experiment, so the tests here are mostly about what must NOT silently
disappear -- and about front matter, which must disappear from text while
surviving in meta.
"""
from __future__ import annotations

import pytest

from kbeval.config import SourceCfg
from kbeval.registry import SOURCES
from kbeval.sources import load_documents


def test_corpus_shape(docs):
    assert {d.doc_id for d in docs} == {"bng_qos.md", "olt_auth.md"}


def test_doc_ids_are_relative_posix_paths(docs):
    assert all("/" not in d.doc_id or not d.doc_id.startswith("/") for d in docs)
    assert all(not d.doc_id.startswith("data/") for d in docs)


def test_front_matter_is_lifted_into_meta(docs_by_id):
    d = docs_by_id["olt_auth.md"]
    assert d.title == "OLT 가입자 인증 오류 대응 가이드"
    assert d.meta["component"] == "OLT"
    assert d.meta["updated"] == "2026-03-11"


def test_front_matter_is_removed_from_text(docs_by_id):
    d = docs_by_id["olt_auth.md"]
    assert not d.text.lstrip().startswith("---")
    assert "component: OLT" not in d.text


def test_offsets_index_the_body_not_the_file(docs_by_id):
    """Editing `updated:` must not shift a single gold span."""
    d = docs_by_id["olt_auth.md"]
    raw = (0, d.text.index("E-411은 ONT 시리얼"))
    assert d.text[raw[1]:].startswith("E-411은 ONT 시리얼")


def test_body_keeps_boilerplate_for_the_transform_to_remove(docs_by_id):
    """strip_boilerplate is a measured factor, so the raw pack must still see it."""
    assert "문서 번호" in docs_by_id["olt_auth.md"].text


def test_missing_source_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(SourceCfg(path=str(tmp_path / "nope")))


def test_empty_match_raises_rather_than_returning_nothing(tmp_path):
    (tmp_path / "note.txt").write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError, match="no documents matched"):
        load_documents(SourceCfg(path=str(tmp_path)))


def test_jira_adapter_is_registered_for_later():
    assert {"markdown_docs", "jira_export"} <= set(SOURCES.names())


def test_load_order_is_stable(cfg):
    assert [d.doc_id for d in load_documents(cfg.source)] == \
           [d.doc_id for d in load_documents(cfg.source)]
