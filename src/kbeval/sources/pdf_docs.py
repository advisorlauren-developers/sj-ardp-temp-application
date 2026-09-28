"""PDF manuals and standards as documents -- the vector-RAG side of the corpus.

An adapter decides the coordinate space, and for a PDF that decision weighs
more than it does for markdown: nobody authored or labelled the text layer, so
whatever the extractor emits IS the document as far as gold spans, retrieval
and every pack are concerned.

Four choices follow from that, each deliberate:

* **Extraction is cached on disk** (``params.cache_dir``). The cache is not a
  speed trick, it pins the coordinate space. Gold offsets are labelled against
  extracted text, so a pypdf upgrade that moves one space would otherwise slide
  every span with nothing failing. The key carries the PDF digest, the extractor
  identity and a digest of the normalisation rules below; on a mismatch the
  document is re-extracted instead of half-matched.

* **Page furniture is kept.** "ETSI TS 138 215 V19.1.0" repeated on 400 pages
  is noise -- but dropping it here would drop it from `raw` too, and then
  strip_boilerplate would have nothing left to remove. Cleanup belongs to a
  pack transform, because that is the only place it becomes a number.

* **Section structure comes from the PDF outline, not from a regex.** 16 of the
  20 documents in this corpus ship an outline whose entries are the authored
  clause titles ("5.1.2 CSI reference signal received power (CSI-RSRP)"), and
  97% of them can be located verbatim on their destination page. Guessing the
  same structure from line shapes costs ~15% false positives -- table cells and
  body sentences whose line break landed after a clause cross-reference -- and
  a false boundary is not cosmetic: inline_meta renders the breadcrumb into the
  chunk text, so it spends injected tokens and feeds BM25 a section the chunk
  does not belong to. The pattern rule stays as the fallback for the four
  documents with no outline, and `meta["headings"]` says which path a document
  took, because a pack built on guessed structure is not the same measurement
  as one built on declared structure.

* **A leading ``# <title>`` line is added.** The heading chunker starts at the
  first heading it finds: with nothing above the cover page and the table of
  contents, those pages would exist in `raw` and vanish from `mid` and `full`
  without anything reporting it.

Page boundaries are written to the cache sidecar rather than into the text. A
page marker in the text would be addressable by gold, retrievable by BM25 and
payable in injected tokens, which is three costs for provenance that belongs
next to the corpus and not inside it.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
from pathlib import Path
from typing import Any

from fontTools import version as FONTTOOLS_VERSION
from pypdf import PdfReader
from pypdf import __version__ as PYPDF_VERSION

from ..registry import SOURCES
from ..types import Document

PARAM_KEYS = {"glob", "cache_dir", "headings", "max_pages"}

# Stamped into every cache key. fontTools is what lets pypdf resolve CFF Type1
# encodings; without it pypdf warns and can fall back to a coarser mapping, so
# it is a pinned dependency rather than an optional nicety -- the text layer is
# the coordinate space, and it may not depend on what happens to be installed.
EXTRACTOR = f"pypdf-{PYPDF_VERSION}+fonttools-{FONTTOOLS_VERSION}"

MAX_HEADING_LEVEL = 4        # the heading chunker reads #{1,4}; level 1 is the title
SLUG_CHARS = 28              # see _doc_id: every injected chunk pays for this

# "(3GPP TS 38.215)" is the identity of a standards document, so it makes the
# best handle when it is there. Everything else falls back to the file name.
# No \b before TS: in "ETSI_3GPP_TS_38.215" the character before it is an
# underscore, which is a word character, so \b never matches there.
_DOC_NUMBER = re.compile(r"(?<![A-Za-z])(?:TS|TR)[ _]?(\d{2}\.\d{3})(?![\d.])",
                         re.IGNORECASE)
_NON_SLUG = re.compile(r"[^a-z0-9.]+")

# Unmappable glyphs (a Symbol-font delta in an equation) arrive as C0 control
# characters: invisible to a reader, unsearchable for BM25, and they force
# \uXXXX escapes into the saved pack, which the project asserts stays readable.
# The glyph noise that IS visible -- U+F0xx private use, pypdf's "/g1840"
# fallbacks -- is deliberately left in: it is corpus dirt, and corpus dirt is
# what a pack transform is measured on.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANKS = re.compile(r"\n{3,}")
_LEADING_NUMBER = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,4})\b")

# The fallback rule, for documents with no outline. Every guard exists because
# something that is NOT a heading wears the same shape: numbered list items
# ("2. Internet access"), contents rows with dot leaders, spec references
# ("38.215 version 19.1.0"), formula fragments lifted out of a table, and -- by
# far the commonest -- a body sentence whose line break landed right after a
# clause cross-reference ("... as described in clause\n9.1.2.1 or 16.5.1.1 for
# Type-1 ...").
_CLAUSE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,4})[ \t]+([^\W\d_(][^\n]{0,110})$",
                     re.MULTILINE)
_MAX_CLAUSE_ROOT = 20        # clause 6.3.1 is plausible, "38.215" is a spec number
_MARKED = re.compile(r"^#{2,4} ", re.MULTILINE)


def _doc_id(rel: str, stem: str) -> str:
    """A short handle for a document, and the reason it is not the file path.

    doc_id is not just a key: chunk_id is f"{doc_id}#NNNN", and the chunk
    template renders that id next to every injected chunk. With the file paths
    in this corpus (110 characters of descriptive name) each injected chunk
    spent ~50 of the 3000 context tokens on its own id, the rendered context
    ran over the budget, and a pack with more, smaller chunks paid that overhead
    more often than a pack with fewer -- a pack-dependent cost with no data
    quality in it.

    The 4-hex tail is the digest of the relative path, not a collision
    counter: an id that depended on what else is in the corpus would move when
    a document is added, and gold labels would move with it.
    """
    m = _DOC_NUMBER.search(stem)
    base = f"ts{m.group(1)}" if m else stem
    slug = _NON_SLUG.sub("-", base.lower()).strip("-.")
    if len(slug) > SLUG_CHARS:
        slug = slug[:SLUG_CHARS].rsplit("-", 1)[0]
    tag = hashlib.sha256(rel.encode()).hexdigest()[:4]
    return f"{slug or 'doc'}-{tag}"


def _page_text(raw: str) -> str:
    """Deterministic normalisation of what is invisible, and nothing else.

    pypdf pads most lines with trailing spaces; that padding is layout, not
    content, and leaving it in would spend gold-span coordinates and injected
    tokens on it. Control characters go for the reasons at _CONTROL. Everything
    a reader can see -- the mid-sentence line breaks, the running headers, the
    unmapped glyphs -- is left exactly as extracted, because that is the corpus
    a transform has to be measured against.
    """
    t = raw.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\u00a0", " ").replace("\u00ad", "")   # nbsp, soft hyphen
    t = _CONTROL.sub("", t)
    t = _TRAILING_WS.sub("", t)
    return _BLANKS.sub("\n\n", t).strip()


def _level(title: str, depth: int) -> int:
    """Depth from the clause number when the title has one, outline nesting
    otherwise. The number is what a reader of the standard sees as structure;
    outline nesting is what the publisher's tool happened to emit."""
    m = _LEADING_NUMBER.match(title)
    lvl = m.group(1).count(".") + 2 if m else depth + 2
    return min(lvl, MAX_HEADING_LEVEL)


def _tolerant(title: str) -> re.Pattern[str]:
    """Match a title across the text layer's own line breaks.

    Extraction splits words ("secondary ce ll activation") and wraps long
    titles, so an exact search finds nothing. Allowing optional whitespace
    between every character costs one regex and recovers those hits.
    """
    return re.compile(r"\s*".join(re.escape(ch) for ch in re.sub(r"\s+", "", title)))


def _as_heading(m: re.Match[str]) -> str:
    num, title = m.group(1), m.group(2).rstrip()
    if title.endswith((".", ",", ";")) or "..." in title:
        return m.group(0)                      # a sentence, or a contents row
    if title[:1].islower():
        # "9.1.2.1 or 16.5.1.1 for Type-1 ...", "0 and applies the second TPC
        # command value", "2 y] [0c": 597 of these in this corpus against zero
        # genuine clause titles that open lower case.
        return m.group(0)
    if int(num.partition(".")[0]) > _MAX_CLAUSE_ROOT:
        return m.group(0)
    level = min(num.count(".") + 2, MAX_HEADING_LEVEL)
    return f"{'#' * level} {num} {title}"


def _markdownise(text: str) -> str:
    return _CLAUSE.sub(_as_heading, text)


def _entry(reader: PdfReader, item: Any) -> tuple[str, int] | None:
    """One outline entry resolved to (title, page index), or None.

    A destination can be unresolvable in an otherwise fine PDF, and one broken
    entry must not cost the document its structure.
    """
    try:
        return str(item.title).strip(), reader.get_destination_page_number(item)
    except Exception:  # noqa: BLE001
        return None


def _outline(reader: PdfReader) -> tuple[list[tuple[int, str, int]], int]:
    """((depth, title, page index) per usable entry, entries seen).

    The count is of entries SEEN, not entries kept: an entry that could not be
    resolved has to stay in the denominator, or `meta["headings"]` would report
    "37/37" on a document where one section quietly has no boundary.
    """
    out: list[tuple[int, str, int]] = []
    seen = 0

    def walk(items: Any, depth: int) -> None:
        nonlocal seen
        for it in items:
            if isinstance(it, list):
                walk(it, depth + 1)
                continue
            seen += 1
            resolved = _entry(reader, it)
            if resolved and resolved[0] and resolved[1] is not None:
                out.append((depth, resolved[0], resolved[1]))

    try:
        walk(reader.outline, 0)
    except Exception:  # noqa: BLE001
        return [], seen
    return out, seen


def _find_line(pages: list[str], page: int, title: str) -> tuple[int, int] | None:
    """Locate a title and return (page index, offset of the line holding it).

    Destinations are occasionally one page off, and a title may also occur in
    running prose, so a match that STARTS a line wins over one that does not.
    Falls back to the clause number alone at a line start: the misses are all
    dashes and symbols the outline and the text layer encode differently
    ("UE Rx Œ Tx time difference"), and the number survives both.
    """
    rx = _tolerant(title)
    num = _LEADING_NUMBER.match(title)
    num_rx = re.compile(rf"^{re.escape(num.group(1))}\s", re.MULTILINE) if num else None

    for at_line_start in (True, False):
        for cand in (page, page + 1, page - 1):
            if not 0 <= cand < len(pages):
                continue
            text = pages[cand]
            for pattern in (rx, num_rx):
                if pattern is None:
                    continue
                for m in pattern.finditer(text):
                    line = text.rfind("\n", 0, m.start()) + 1
                    if at_line_start and text[line:m.start()].strip():
                        continue
                    return cand, line
    return None


def _mark(pages: list[str], marks: list[tuple[int, int, int]]) -> list[str]:
    """Insert heading markers, one page at a time, back to front so that every
    offset still points where it was measured."""
    by_page: dict[int, list[tuple[int, int]]] = {}
    for page, line, level in marks:
        by_page.setdefault(page, []).append((line, level))

    out = list(pages)
    for page, items in by_page.items():
        text = out[page]
        seen: set[int] = set()
        for line, level in sorted(items, reverse=True):
            if line in seen or text[line:line + 1] == "#":
                continue                        # two entries on one line, or already marked
            seen.add(line)
            text = f"{text[:line]}{'#' * level} {text[line:]}"
        out[page] = text
    return out


# The cache key must move when the RULES move, not just when the PDF or the
# extractor does: editing anything above would otherwise leave every cached
# document in place and the corpus would be half old text, half new.
_RULES_DIGEST = hashlib.sha256("".join(inspect.getsource(f) for f in (
    _doc_id, _page_text, _level, _tolerant, _as_heading, _markdownise,
    _entry, _outline, _find_line, _mark,
)).encode()).hexdigest()[:8]


@SOURCES.register("pdf_docs")
class PdfDocs:
    def __init__(self, cfg) -> None:
        # params is a free-form dict, so pydantic's extra="forbid" cannot see
        # into it. Checking the keys here restores rule 3: a misspelt setting
        # is an error, never a silent default.
        unknown = set(cfg.params) - PARAM_KEYS
        if unknown:
            raise ValueError(f"unknown pdf_docs params: {sorted(unknown)} "
                             f"(known: {sorted(PARAM_KEYS)})")
        self.root = Path(cfg.path)
        self.glob = cfg.params.get("glob", "**/*.pdf")
        cache = cfg.params.get("cache_dir", "data/extracted")
        self.cache_dir = Path(cache) if cache else None
        self.headings = bool(cfg.params.get("headings", True))
        self.max_pages = int(cfg.params.get("max_pages", 0))   # 0 = all pages

    # -- the adapter contract -------------------------------------------------

    def load(self) -> list[Document]:
        if not self.root.exists():
            raise FileNotFoundError(f"source path not found: {self.root}")
        docs = [self._document(p) for p in sorted(self.root.glob(self.glob))]
        if not docs:
            raise ValueError(f"no documents matched {self.root}/{self.glob}")
        # Two documents under one id would share a chunk_id namespace and merge
        # into one corpus entry with nothing failing anywhere downstream.
        seen: dict[str, str] = {}
        for d in docs:
            clash = seen.setdefault(d.doc_id, d.meta["source_path"])
            if clash != d.meta["source_path"]:
                raise ValueError(f"doc_id collision {d.doc_id!r}: "
                                 f"{clash} and {d.meta['source_path']}")
        return docs

    # -- internals ------------------------------------------------------------

    @property
    def _params(self) -> dict[str, Any]:
        """Only what changes the extracted text. `cache_dir` is where the text
        is kept, not what it says, so it stays out of the key."""
        return {"headings": self.headings, "max_pages": self.max_pages}

    def _document(self, path: Path) -> Document:
        rel = path.relative_to(self.root).as_posix()
        doc_id = _doc_id(rel, path.stem)
        key = self._key(path)
        side = self._cached(doc_id, key)
        if side is None:
            text, side = self._extract(path)
            side["key"] = key
            self._store(doc_id, text, side)
        else:
            text = side.pop("_text")
        return Document(
            doc_id=doc_id,
            title=path.stem.replace("_", " "),
            text=text,
            meta={"source_path": rel,
                  "taxonomy": path.parent.relative_to(self.root).as_posix() or ".",
                  "n_pages": side["n_pages"],
                  "pages_failed": side["pages_failed"],
                  "headings": side["headings"]},
        )

    def _key(self, path: Path) -> str:
        h = hashlib.sha256(path.read_bytes())
        h.update(json.dumps(self._params, sort_keys=True).encode())
        h.update(f"{EXTRACTOR}/{_RULES_DIGEST}".encode())
        return h.hexdigest()[:16]

    def _extract(self, path: Path) -> tuple[str, dict[str, Any]]:
        reader = PdfReader(str(path))
        n_total = len(reader.pages)
        pages = reader.pages[:self.max_pages] if self.max_pages else reader.pages

        parts: list[str] = []
        failed = 0
        for page in pages:
            try:
                raw = page.extract_text() or ""
            except Exception:  # noqa: BLE001  -- one unreadable page must not
                raw, failed = "", failed + 1   # cost the other 399; meta counts it
            parts.append(_page_text(raw))

        if self.headings:
            parts, how, found, wanted = self._headings(reader, parts)
        else:
            how, found, wanted = "off", 0, 0

        head = f"# {path.stem.replace('_', ' ')}\n\n"
        text = head + "\n\n".join(parts)

        starts, cur = [], len(head)
        for t in parts:
            starts.append(cur)
            cur += len(t) + 2
        return text, {
            "source_path": str(path), "n_pages": n_total, "pages_extracted": len(parts), "pages_failed": failed,
            "headings": f"{how} {found}/{wanted}" if wanted else f"{how} {found}",
            "page_starts": starts, "extractor": EXTRACTOR, "rules": _RULES_DIGEST,
            "pdf_title": (reader.metadata or {}).get("/Title") or "",
        }

    def _headings(self, reader: PdfReader,
                  parts: list[str]) -> tuple[list[str], str, int, int]:
        items, seen = _outline(reader)
        if items:
            marks = []
            for depth, title, page in items:
                if self.max_pages and page >= len(parts):
                    continue
                spot = _find_line(parts, page, title)
                if spot is not None:
                    marks.append((*spot, _level(title, depth)))
            if marks:
                return _mark(parts, marks), "outline", len(marks), seen
        marked = [_markdownise(t) for t in parts]
        # count what was ACCEPTED, not what matched: _as_heading rejects most
        # candidates, and a reading that inflates itself is not a reading.
        n = sum(len(_MARKED.findall(t)) for t in marked)
        return marked, "pattern", n, 0

    def _paths(self, doc_id: str) -> tuple[Path, Path]:
        # Built by concatenation, not with_suffix: a doc_id like "ts38.215-9c1d"
        # already has what Path calls a suffix, and replacing it would rename
        # the file.
        base = f"{self.cache_dir / doc_id}"
        return Path(f"{base}.txt"), Path(f"{base}.json")

    def _cached(self, doc_id: str, key: str) -> dict[str, Any] | None:
        if self.cache_dir is None:
            return None
        txt, meta = self._paths(doc_id)
        if not (txt.exists() and meta.exists()):
            return None
        side = json.loads(meta.read_text(encoding="utf-8"))
        if side.get("key") != key:
            return None                      # stale: the PDF or the rules moved
        side["_text"] = txt.read_text(encoding="utf-8")
        return side

    def _store(self, doc_id: str, text: str, side: dict[str, Any]) -> None:
        if self.cache_dir is None:
            return
        txt, meta = self._paths(doc_id)
        txt.parent.mkdir(parents=True, exist_ok=True)
        txt.write_text(text, encoding="utf-8")
        meta.write_text(json.dumps(side, ensure_ascii=False, indent=2), encoding="utf-8")
