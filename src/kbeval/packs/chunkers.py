"""Chunkers MUST preserve source offsets.

This is the one requirement in the project that cannot be retrofitted. With
char_start / char_end pointing back into Document.text, a gold span is labelled
ONCE and maps automatically onto every pack. Without it, each new pack costs a
fresh labelling pass and the factor ablation stops being affordable.

The invariant every chunker must satisfy:

    doc.text[chunk.char_start:chunk.char_end].strip() == chunk.text

Chunkers may only slice. Rewriting text is a transform's job (S4), and even
then offsets stay untouched.
"""
from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from ..registry import CHUNKERS
from ..types import Chunk, Document

_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]


@runtime_checkable
class Chunker(Protocol):
    def split(self, doc: Document) -> list[Chunk]:
        ...


def _snap(text: str, start: int, end: int, size: int) -> int:
    """Pull a window end back to the strongest separator within tolerance.

    Tried strongest-first, so a heading boundary beats a blank line, which
    beats a newline. Never searches before start + size//2, so snapping cannot
    collapse a window to nothing.
    """
    floor = max(start + size // 2, end - size // 3)
    for sep in _SEPARATORS:
        idx = text.rfind(sep, floor, end)
        if idx != -1:
            return idx + len(sep)
    return end


def _spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    n, start = len(text), 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            end = _snap(text, start, end, size)
        out.append((start, end))
        if end >= n:
            break
        start = max(end - overlap, start + 1)   # +1 guards against no progress
    return out


def _emit(doc: Document, pairs: list[tuple[int, int]], extra: dict | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for s, e in pairs:
        body = doc.text[s:e].strip()
        if not body:
            continue
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}#{len(chunks):04d}",
            doc_id=doc.doc_id, char_start=s, char_end=e, text=body,
            meta=dict(doc.meta, title=doc.title, **(extra or {})),
        ))
    return chunks


@CHUNKERS.register("flat")
class FlatChunker:
    """Fixed width, separator-blind. The deliberately naive baseline.

    Cuts mid-sentence and mid-table by design. Pairing this with the `raw` pack
    is what makes structure-aware chunking measurable: without a bad chunker in
    the sweep, there is nothing for a good one to beat.
    """

    def __init__(self, size: int, overlap: int) -> None:
        self.size, self.overlap = size, overlap

    def split(self, doc: Document) -> list[Chunk]:
        pairs, start = [], 0
        while start < len(doc.text):
            end = min(start + self.size, len(doc.text))
            pairs.append((start, end))
            if end >= len(doc.text):
                break
            start = end - self.overlap
        return _emit(doc, pairs)


@CHUNKERS.register("recursive")
class RecursiveChunker:
    """Windowed, but snapped to the strongest nearby separator."""

    def __init__(self, size: int, overlap: int) -> None:
        self.size, self.overlap = size, overlap

    def split(self, doc: Document) -> list[Chunk]:
        return _emit(doc, _spans(doc.text, self.size, self.overlap))


@CHUNKERS.register("heading")
class HeadingChunker:
    """Split on markdown headings, then window anything oversized.

    Records heading_path in meta. That costs nothing now and pays off in S4:
    the inline_meta transform can surface "OLT 가이드 > E-411 인증 실패" into the
    chunk text, so a retrieved fragment carries its own context instead of
    arriving as an orphaned paragraph.
    """

    _RX = re.compile(r"^(#{1,4})\s+(.*)$", re.MULTILINE)

    def __init__(self, size: int, overlap: int) -> None:
        self.size, self.overlap = size, overlap

    def split(self, doc: Document) -> list[Chunk]:
        marks = [(m.start(), len(m.group(1)), m.group(2).strip())
                 for m in self._RX.finditer(doc.text)]
        if not marks:
            return RecursiveChunker(self.size, self.overlap).split(doc)

        bounds = [m[0] for m in marks] + [len(doc.text)]
        chunks: list[Chunk] = []
        # Slot by depth, not by list length: two sibling "##" sections must not
        # nest under each other. A wrong breadcrumb is not cosmetic -- inline_meta
        # renders it into the chunk text, so it spends injected-token budget and
        # feeds BM25 a parent section the chunk does not belong to.
        crumbs: list[str] = [""] * 6
        for (pos, level, title), end in zip(marks, bounds[1:], strict=True):
            crumbs[level - 1] = title
            for i in range(level, len(crumbs)):
                crumbs[i] = ""
            path = [c for c in crumbs[:level] if c]
            section = doc.text[pos:end]
            for s, e in _spans(section, self.size, self.overlap):
                body = section[s:e].strip()
                if not body:
                    continue
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}#{len(chunks):04d}", doc_id=doc.doc_id,
                    char_start=pos + s, char_end=pos + e, text=body,
                    meta=dict(doc.meta, title=doc.title, heading_path=" > ".join(path)),
                ))
        return chunks


def build_chunker(name: str, size: int, overlap: int) -> Chunker:
    return CHUNKERS.get(name)(size, overlap)