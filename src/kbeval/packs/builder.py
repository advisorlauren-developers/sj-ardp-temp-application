"""Pack assembly, gold mapping, and the coverage check.

The payoff of everything so far lands here: because chunkers preserve source
offsets, a gold span labelled ONCE on the document resolves onto every pack
automatically, no matter how differently each pack chose to cut and rewrite.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import Config
from ..sources import load_documents
from ..types import Chunk, Document, GoldSpan, Question
from .chunkers import build_chunker
from .transforms import apply_transforms

PACK_ROOT = Path("data/packs")


def corpus_stats(chunks: list[Chunk]) -> dict[str, Any]:
    """Reported as COVARIATES, not decoration.

    A pack that grows the index makes retrieval harder for reasons unrelated to
    data quality. Without these numbers, "more chunks to search" and "worse
    chunks to search" are indistinguishable in the final report.
    """
    lens = [len(c.text) for c in chunks] or [0]
    return {
        "n_chunks": len(chunks),
        "n_docs": len({c.doc_id for c in chunks}),
        "total_chars": sum(lens),
        "avg_chunk_chars": round(sum(lens) / max(len(lens), 1), 1),
        "max_chunk_chars": max(lens),
    }


class Pack:
    def __init__(self, name: str, chunks: list[Chunk], manifest: dict[str, Any]) -> None:
        self.name = name
        self.chunks = chunks
        self.manifest = manifest
        self._by_id = {c.chunk_id: c for c in chunks}

    def get(self, chunk_id: str) -> Chunk:
        return self._by_id[chunk_id]

    def has(self, chunk_id: str) -> bool:
        return chunk_id in self._by_id

    def gold_chunk_ids(self, q: Question, min_overlap_ratio: float) -> list[str]:
        """Resolve a source-level gold span onto THIS pack's chunks by offset
        overlap.

        Two ways to qualify, because chunk sizes vary wildly between packs:
        a chunk that covers enough of the span, or a chunk so fully contained
        in the span that it is gold by construction (fine-grained packs).

        Both tests use SOURCE coordinates, never len(chunk.text): a transform
        like inline_meta lengthens the rendered text without covering any more
        of the document, and comparing against it would let a header dilute a
        chunk out of gold status.
        """
        if q.gold is None:
            return []
        span_len = max(q.gold.char_end - q.gold.char_start, 1)
        hits = []
        for c in self.chunks:
            if c.doc_id != q.gold.doc_id:
                continue
            ov = c.overlap(q.gold.char_start, q.gold.char_end)
            if ov == 0:
                continue
            chunk_span = max(c.char_end - c.char_start, 1)
            if ov / span_len >= min_overlap_ratio or ov / chunk_span > 0.9:
                hits.append(c.chunk_id)
        return hits

    def verify_gold_coverage(self, questions: list[Question],
                             min_overlap_ratio: float) -> dict[str, Any]:
        """Fail loudly when a transform has removed ground truth.

        An answerable question whose gold no longer exists in the pack is not a
        hard question -- it is a broken pack. Leaving it in silently would make
        an aggressive dedup or filter look like a model regression.
        """
        answerable = [q for q in questions if q.answerable and q.gold is not None]
        missing = [q.qid for q in answerable if not self.gold_chunk_ids(q, min_overlap_ratio)]
        return {
            "pack": self.name,
            "n_answerable": len(answerable),
            "missing_gold": missing,
            "coverage": round(1 - len(missing) / max(len(answerable), 1), 3),
            "avg_gold_chunks": round(
                sum(len(self.gold_chunk_ids(q, min_overlap_ratio)) for q in answerable)
                / max(len(answerable), 1), 2),
        }

    @property
    def dir(self) -> Path:
        return PACK_ROOT / self.name

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with (self.dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        (self.dir / "manifest.json").write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def build_pack(cfg: Config, pack_name: str, docs: list[Document] | None = None) -> Pack:
    pc = cfg.packs[pack_name]
    docs = load_documents(cfg.source) if docs is None else docs
    chunker = build_chunker(pc.chunker, pc.chunk_chars, pc.chunk_overlap)

    chunks: list[Chunk] = [c for d in docs for c in chunker.split(d)]
    chunks = apply_transforms(chunks, pc.transforms)

    pack = Pack(pack_name, chunks, {
        "name": pack_name,
        "chunker": pc.chunker,
        "chunk_chars": pc.chunk_chars,
        "chunk_overlap": pc.chunk_overlap,
        "transforms": pc.transforms,
        "source": cfg.source.model_dump(mode="json"),
        **corpus_stats(chunks),
    })
    pack.save()
    return pack


def load_pack(name: str) -> Pack:
    d = PACK_ROOT / name
    chunks = [Chunk(**json.loads(ln))
              for ln in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if ln]
    return Pack(name, chunks, json.loads((d / "manifest.json").read_text(encoding="utf-8")))


def load_questions(path: str | Path, intents: list[str] | None = None) -> list[Question]:
    out: list[Question] = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        d = json.loads(ln)
        q = Question(
            qid=d["qid"], intent=d["intent"], query=d["query"],
            answerable=d.get("answerable", True),
            gold=GoldSpan(**d["gold"]) if d.get("gold") else None,
            gold_answer=d.get("gold_answer"), oracle_ref=d.get("oracle_ref"),
            meta=d.get("meta", {}),
        )
        if intents is None or q.intent in intents:
            out.append(q)
    return out