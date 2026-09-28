"""S2 -- the PDF adapter, which is where a spec becomes a coordinate space.

Everything here is about what must NOT happen quietly. A PDF has no authored
text layer, so extraction invents the document that gold spans will index,
retrieval will reach and every pack will cut up. Three silent failures would
each be fatal and each look like a model result later:

  * a heading in the wrong place -- inline_meta renders the breadcrumb into the
    chunk, so a false boundary spends injected tokens and feeds BM25 a section
    the chunk is not in;
  * a stale cache -- half the corpus extracted under old rules, offsets moved,
    nothing failing;
  * a page that failed to extract -- content missing from the corpus and
    therefore unanswerable, reported as the model not knowing.

The fake reader keeps these tests independent of any particular PDF: the rules
are what is under test, not pypdf.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

from kbeval.config import SourceCfg
from kbeval.packs import build_chunker
from kbeval.sources import load_documents, pdf_docs


class FakePage:
    def __init__(self, text: str, boom: bool = False) -> None:
        self.text, self.boom = text, boom

    def extract_text(self) -> str:
        if self.boom:
            raise ValueError("no text layer on this page")
        return self.text


class FakeDest:
    def __init__(self, title: str, page: int) -> None:
        self.title, self.page = title, page


class FakeReader:
    """Registered per file name, so several PDFs can differ in one test."""
    by_name: ClassVar[dict[str, tuple[list[FakePage], list]]] = {}
    opened: ClassVar[list[str]] = []

    def __init__(self, path: str) -> None:
        self.pages, self.outline = FakeReader.by_name[Path(path).name]
        FakeReader.opened.append(Path(path).name)

    metadata: ClassVar[dict] = {}

    def get_destination_page_number(self, dest: FakeDest) -> int:
        return dest.page


@pytest.fixture(autouse=True)
def fake_pdfs(monkeypatch):
    FakeReader.by_name = {}
    FakeReader.opened = []
    monkeypatch.setattr(pdf_docs, "PdfReader", FakeReader)
    return FakeReader


def write(root: Path, name: str, pages: list[FakePage], outline: list | None = None) -> Path:
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"%PDF-1.7 not really, the reader is faked\n" + name.encode())
    FakeReader.by_name[Path(name).name] = (pages, outline or [])
    return p


def load(root: Path, **params):
    params.setdefault("cache_dir", str(root / "cache"))
    return load_documents(SourceCfg(adapter="pdf_docs", path=str(root), params=params))


# -- params ------------------------------------------------------------------

def test_a_misspelt_param_is_an_error_not_a_default(tmp_path):
    """Rule 3 reaches inside params. pydantic's extra="forbid" stops at the
    dict boundary, so `heading: true` would otherwise be accepted, ignored, and
    change the corpus without changing the config hash."""
    write(tmp_path, "a.pdf", [FakePage("1 Scope\nbody")])
    with pytest.raises(ValueError, match="unknown pdf_docs params"):
        load(tmp_path, heading=True)


def test_no_pdf_match_raises_instead_of_returning_an_empty_corpus(tmp_path):
    (tmp_path / "note.md").write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError, match="no documents matched"):
        load(tmp_path)


def test_load_order_is_stable(tmp_path):
    for name in ("b.pdf", "a.pdf", "c.pdf"):
        write(tmp_path, name, [FakePage("1 Scope\nbody")])
    order = [d.meta["source_path"] for d in load(tmp_path)]
    assert order == ["a.pdf", "b.pdf", "c.pdf"]


def test_a_spec_number_becomes_the_handle(tmp_path):
    """doc_id is rendered next to every injected chunk (chunk_id is
    f"{doc_id}#NNNN"), so a 110-character file name would spend ~50 of the 3000
    context tokens per chunk -- and spend it more often in a pack with more,
    smaller chunks, which is a pack-dependent cost with no data quality in it.
    """
    write(tmp_path, "L3-1/ETSI_3GPP_5G;_NR;_Physical_layer_measurements_(3GPP_TS_38.215)_V19.pdf",
          [FakePage("body")])
    d = load(tmp_path)[0]
    assert d.doc_id.startswith("ts38.215-")
    assert len(d.doc_id) < 20
    assert d.meta["source_path"].endswith("_V19.pdf")     # the path is not lost
    assert d.meta["taxonomy"] == "L3-1"


def test_a_document_with_no_number_falls_back_to_its_name(tmp_path):
    write(tmp_path, "National_Instruments_(NI)_5G_New_Radio_Introduction.pdf", [FakePage("b")])
    d = load(tmp_path)[0]
    assert d.doc_id.startswith("national-instruments-ni-5g")
    assert "/" not in d.doc_id and " " not in d.doc_id


def test_the_handle_does_not_depend_on_the_rest_of_the_corpus(tmp_path):
    """The tail is a digest of this document's path, not a collision counter:
    an id that moved when a neighbour was added would take every gold label
    labelled against it along."""
    write(tmp_path, "a/spec.pdf", [FakePage("body")])
    alone = load(tmp_path)[0].doc_id
    write(tmp_path, "b/other.pdf", [FakePage("body")])
    assert alone in [d.doc_id for d in load(tmp_path)]


def test_two_documents_cannot_share_a_doc_id(tmp_path, monkeypatch):
    """A shared id would merge two documents into one chunk_id namespace, and
    nothing downstream would report it."""
    write(tmp_path, "one.pdf", [FakePage("body one")])
    write(tmp_path, "two.pdf", [FakePage("body two")])
    monkeypatch.setattr(pdf_docs, "_doc_id", lambda rel, stem: "same-0000")
    with pytest.raises(ValueError, match="doc_id collision"):
        load(tmp_path)


# -- the text layer ----------------------------------------------------------

def test_line_padding_goes_and_line_breaks_stay(tmp_path):
    """pypdf pads lines with trailing spaces: layout, not content, and it would
    otherwise be addressable by gold and payable in injected tokens. The line
    breaks themselves are content and must survive -- they are how the source
    actually reads."""
    write(tmp_path, "a.pdf", [FakePage("ETSI   \nfirst line   \nsecond line\n\n\n\nafter")])
    text = load(tmp_path)[0].text
    assert "   \n" not in text
    assert "first line\nsecond line" in text
    assert "\n\n\n" not in text


def test_invisible_glyphs_go_and_visible_dirt_stays(tmp_path):
    """An unmappable Symbol-font glyph arrives as a C0 control character
    ("\x01f = 15 kHz"): invisible, unsearchable, and it forces a \\uXXXX
    escape into the saved pack. The dirt that a reader can SEE is left alone --
    that is what strip_boilerplate and friends are measured on."""
    write(tmp_path, "a.pdf", [FakePage("LTE with \x01f = 15 kHz\x0c and \uf0ea brackets")])
    text = load(tmp_path)[0].text
    assert "\x01" not in text and "\x0c" not in text
    assert "with f = 15 kHz" in text
    assert "\uf0ea" in text


def test_the_document_opens_with_its_title_so_the_cover_page_survives(tmp_path):
    """The heading chunker starts at the first heading it finds. With nothing
    above the cover page, `mid` and `full` would silently drop what `raw`
    keeps."""
    write(tmp_path, "TS_38.215_v19.pdf", [FakePage("cover page\nno heading here"),
                                          FakePage("1 Scope\nthe scope")])
    d = load(tmp_path)[0]
    assert d.text.startswith("# TS 38.215 v19\n\n")
    assert "cover page" in d.text


def test_page_boundaries_stay_out_of_the_text_and_next_to_it(tmp_path):
    """A page marker in the text would be addressable by gold, retrievable by
    BM25 and payable in tokens -- three costs for provenance that belongs in
    the sidecar."""
    write(tmp_path, "a.pdf", [FakePage("page one body"), FakePage("page two body")])
    d = load(tmp_path)[0]
    assert "page one body" in d.text and "[page" not in d.text
    side = json.loads((tmp_path / "cache" / f"{d.doc_id}.json").read_text(encoding="utf-8"))
    starts = side["page_starts"]
    assert d.text[starts[0]:].startswith("page one body")
    assert d.text[starts[1]:].startswith("page two body")


def test_offsets_index_the_extracted_text(tmp_path):
    """The S3 invariant, checked on this adapter's output: whatever extraction
    produced is what chunk coordinates mean."""
    write(tmp_path, "a.pdf", [FakePage("1 Scope\n" + "body text. " * 40),
                              FakePage("2 References\n" + "more text. " * 40)])
    d = load(tmp_path)[0]
    for c in build_chunker("heading", 300, 40).split(d):
        assert d.text[c.char_start:c.char_end].strip() == c.text


# -- structure ---------------------------------------------------------------

def test_the_outline_decides_the_headings_when_there_is_one(tmp_path):
    write(tmp_path, "a.pdf",
          [FakePage("1 Scope\nthe scope"), FakePage("3.1 Terms\nsome terms")],
          outline=[FakeDest("1 Scope", 0), [FakeDest("3.1 Terms", 1)]])
    d = load(tmp_path)[0]
    assert "## 1 Scope" in d.text
    assert "### 3.1 Terms" in d.text          # level follows the clause number
    assert d.meta["headings"] == "outline 2/2"


def test_a_title_broken_by_the_text_layer_is_still_located(tmp_path):
    """Extraction splits words ("secondary ce ll activation"). An exact search
    would lose the heading and the section boundary with it."""
    write(tmp_path, "a.pdf", [FakePage("4.3 Timing for secondary ce ll\nactivation\nbody")],
          outline=[FakeDest("4.3 Timing for secondary cell activation", 0)])
    d = load(tmp_path)[0]
    assert "### 4.3 Timing for secondary ce ll" in d.text
    assert d.meta["headings"] == "outline 1/1"


def test_an_outline_title_the_text_layer_spells_differently_falls_back_to_the_number(tmp_path):
    """The real misses are dashes and symbols the outline and the text layer
    encode differently ("UE Rx Œ Tx"). The clause number survives both."""
    write(tmp_path, "a.pdf", [FakePage("5.1.15 UE Rx - Tx time difference\nbody")],
          outline=[FakeDest("5.1.15 UE Rx Œ Tx time difference", 0)])
    d = load(tmp_path)[0]
    assert "#### 5.1.15 UE Rx - Tx time difference" in d.text


def test_a_destination_one_page_off_still_lands(tmp_path):
    write(tmp_path, "a.pdf", [FakePage("front matter"), FakePage("6 Link recovery\nbody")],
          outline=[FakeDest("6 Link recovery", 0)])
    assert "## 6 Link recovery" in load(tmp_path)[0].text


def test_a_broken_outline_entry_does_not_cost_the_document_its_structure(tmp_path):
    class Angry(FakeDest):
        @property
        def title(self):                     # type: ignore[override]
            raise KeyError("/Title")

        @title.setter
        def title(self, v): pass

    write(tmp_path, "a.pdf", [FakePage("1 Scope\nbody"), FakePage("2 References\nbody")],
          outline=[Angry("", 0), FakeDest("2 References", 1)])
    d = load(tmp_path)[0]
    assert "## 2 References" in d.text
    assert d.meta["headings"] == "outline 1/2"


def test_without_an_outline_the_pattern_rule_takes_over(tmp_path):
    write(tmp_path, "a.pdf", [FakePage("3.2 Symbols\nsome symbols")])
    d = load(tmp_path)[0]
    assert "### 3.2 Symbols" in d.text
    assert d.meta["headings"] == "pattern 1"


@pytest.mark.parametrize("line", [
    "9.1.2.1 or 16.5.1.1 for Type-1 HARQ-ACK codebook",   # a wrapped sentence
    "0 and applies the second TPC command value",          # ditto, lower case
    "1 Scope ....................................... 6",   # a contents row
    "38.215 version 19.1.0 Release 19",                    # a spec number
    "2. Internet access is available",                     # a numbered list item
])
def test_the_pattern_rule_refuses_the_shapes_that_only_look_like_headings(tmp_path, line):
    """Each of these is a false boundary that would put a wrong breadcrumb into
    the chunk text and a wrong section into the BM25 index."""
    write(tmp_path, "a.pdf", [FakePage(f"{line}\nbody")])
    assert "\n#" not in load(tmp_path)[0].text


def test_headings_can_be_turned_off_entirely(tmp_path):
    write(tmp_path, "a.pdf", [FakePage("1 Scope\nbody")], outline=[FakeDest("1 Scope", 0)])
    d = load(tmp_path, headings=False)[0]
    assert "## 1 Scope" not in d.text
    assert d.meta["headings"] == "off 0"


# -- the cache ---------------------------------------------------------------

def test_the_second_load_reads_the_cache_instead_of_the_pdf(tmp_path):
    write(tmp_path, "a.pdf", [FakePage("1 Scope\nbody")])
    first = load(tmp_path)[0].text
    FakeReader.opened.clear()
    assert load(tmp_path)[0].text == first
    assert FakeReader.opened == []


def test_a_key_mismatch_re_extracts_rather_than_half_matching(tmp_path):
    """The cache pins the coordinate space, so a cache that no longer matches
    the PDF, the extractor or the rules must be rebuilt. Reusing it would leave
    the corpus half in one coordinate system and half in another, with every
    gold offset still looking plausible."""
    write(tmp_path, "a.pdf", [FakePage("1 Scope\nbody")])
    doc_id = load(tmp_path)[0].doc_id
    side = tmp_path / "cache" / f"{doc_id}.json"
    data = json.loads(side.read_text(encoding="utf-8"))
    data["key"] = "stale"
    side.write_text(json.dumps(data), encoding="utf-8")

    FakeReader.opened.clear()
    load(tmp_path)
    assert FakeReader.opened == ["a.pdf"]


def test_the_key_moves_when_the_rules_move(tmp_path, monkeypatch):
    """Editing the normaliser must invalidate every cached document. Without
    this the corpus silently mixes two extraction rule sets, and every gold
    offset in the older half still looks plausible."""
    write(tmp_path, "a.pdf", [FakePage("1 Scope\nbody")])
    load(tmp_path)

    FakeReader.opened.clear()
    monkeypatch.setattr(pdf_docs, "_RULES_DIGEST", "0dd0dd0d")
    load(tmp_path)
    assert FakeReader.opened == ["a.pdf"]


def test_caching_can_be_switched_off(tmp_path):
    write(tmp_path, "a.pdf", [FakePage("1 Scope\nbody")])
    d = load_documents(SourceCfg(adapter="pdf_docs", path=str(tmp_path),
                                 params={"cache_dir": ""}))[0]
    assert "## 1 Scope" in d.text
    assert not (tmp_path / "cache").exists()


# -- failure that must stay visible ------------------------------------------

def test_an_unreadable_page_is_counted_not_swallowed(tmp_path):
    """Missing content is unanswerable content. Counted in meta, it reads as a
    corpus defect; uncounted, it reads later as the model not knowing."""
    write(tmp_path, "a.pdf", [FakePage("good page"), FakePage("", boom=True),
                              FakePage("another good page")])
    d = load(tmp_path)[0]
    assert d.meta["pages_failed"] == 1
    assert d.meta["n_pages"] == 3
    assert "good page" in d.text and "another good page" in d.text


def test_max_pages_trims_the_corpus_and_says_so(tmp_path):
    """A smoke-sized corpus is legitimate, but it is a different corpus: it
    lives in params, so it is inside the config hash."""
    write(tmp_path, "a.pdf", [FakePage("page one"), FakePage("page two")])
    d = load(tmp_path, max_pages=1)[0]
    assert "page two" not in d.text
    assert d.meta["n_pages"] == 2
