"""Pack transforms = the ai-ready factors under test.

Each is one named, independently togglable step, so a pack is a STACK OF
FACTORS rather than an opaque version number. That is what turns "v2 scores 3
points higher" into "inline metadata is worth 3 points" -- the first is a
result, the second is an instruction for whoever maintains the corpus.

Contract: a transform may rewrite chunk TEXT freely, but must never touch
char_start / char_end. Those coordinates belong to the source document and are
what keep gold mapping automatic across every pack.

Consequence: the S3 invariant (doc.text[start:end].strip() == chunk.text) holds
for chunkers only. After a transform the text is intentionally not a slice of
the source any more -- but the offsets still point at where it came from.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ..registry import TRANSFORMS
from ..types import Chunk

GLOSSARY_PATH = Path("data/raw/glossary.tsv")


def _rewrite(c: Chunk, text: str) -> Chunk:
    """Rebuild a chunk with new text and untouched coordinates."""
    return Chunk(c.chunk_id, c.doc_id, c.char_start, c.char_end, text, c.meta)


@TRANSFORMS.register("inline_meta")
def inline_meta(chunks: list[Chunk]) -> list[Chunk]:
    """Prepend a metadata header to the chunk body.

    Metadata held only in a vector-store field is invisible to the model at
    generation time. A retrieved paragraph otherwise arrives orphaned: the
    heading path, the document title and the freshness date are all things the
    reader needs and the chunk does not carry.

    For stale-prone sources (JIRA) this single step is often the difference
    between a confident answer about a bug fixed six months ago and a correct
    one.
    """
    keys = ("title", "heading_path", "status", "resolution", "fix_version", "updated")
    out = []
    for c in chunks:
        bits = [f"{k}: {c.meta[k]}" for k in keys if c.meta.get(k)]
        out.append(_rewrite(c, (f"[{' | '.join(bits)}]\n" if bits else "") + c.text))
    return out


@TRANSFORMS.register("strip_boilerplate")
def strip_boilerplate(chunks: list[Chunk]) -> list[Chunk]:
    """Remove document-control lines that carry no answer.

    They cost budget twice: as injected tokens, and as lexical noise that
    dilutes BM25 term statistics across the corpus.
    """
    rx = re.compile(r"^\s*(문서\s*번호|작성자|개정\s*이력|Confidential|내부용)\s*[::].*$", re.MULTILINE)
    out = []
    for c in chunks:
        t = re.sub(r"\n{3,}", "\n\n", rx.sub("", c.text)).strip()
        if t:
            out.append(_rewrite(c, t))
    return out


@TRANSFORMS.register("normalize_terms")
def normalize_terms(chunks: list[Chunk]) -> list[Chunk]:
    """Pull user phrasing and document phrasing into the same lexical space.

    A user types "E411"; the manual says "E-411". To BM25 those are different
    terms and the gold chunk is unreachable. Error codes are normalised by
    rule; in-house abbreviations are expanded from data/raw/glossary.tsv, kept
    outside the code because the people who maintain it are not the people who
    maintain the harness.
    """
    gloss: dict[str, str] = {}
    if GLOSSARY_PATH.exists():
        for ln in GLOSSARY_PATH.read_text(encoding="utf-8").splitlines():
            if "\t" in ln and not ln.startswith("#"):
                k, _, v = ln.partition("\t")
                gloss[k.strip()] = v.strip()

    code_rx = re.compile(r"\b[Ee][-_ ]?(\d{3,4})\b")
    out = []
    for c in chunks:
        t = code_rx.sub(lambda m: f"E-{m.group(1)}", c.text)
        for k, v in gloss.items():
            if k and k in t and f"{k}(" not in t:
                t = t.replace(k, f"{k}({v})")
        out.append(_rewrite(c, t))
    return out


@TRANSFORMS.register("dedup")
def dedup(chunks: list[Chunk]) -> list[Chunk]:
    """Drop near-identical chunks by normalised-text hash.

    The one transform that can DELETE ground truth: if it removes the only
    chunk carrying a gold span, that is not a cleaner corpus, it is a lower
    ceiling that will read as a model regression. Pack.verify_gold_coverage
    exists to catch exactly this, which is why `kbeval pack` reports coverage
    rather than just chunk counts.
    """
    seen, out = set(), []
    for c in chunks:
        key = hashlib.sha1(re.sub(r"\s+", " ", c.text).strip().encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def apply_transforms(chunks: list[Chunk], names: list[str]) -> list[Chunk]:
    """Order matters and is the experimenter's choice, so it comes from config.
    strip_boilerplate before inline_meta, for instance, or the header is
    stripped as boilerplate the moment it is added."""
    for n in names:
        chunks = TRANSFORMS.get(n)(chunks)
    return chunks