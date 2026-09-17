"""Core data contracts.

Written first, before any implementation. Every axis speaks only these types,
so the axes can be developed and swapped independently.

The one line that matters most: Chunk.char_start / char_end index into
Document.text, NOT into Chunk.text. That is what lets a gold span be labelled
once and mapped onto every pack automatically. It cannot be retrofitted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """Source document before chunking. `text` is the canonical coordinate
    space that every GoldSpan and Chunk offset refers to."""
    doc_id: str
    title: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    char_start: int          # -> Document.text
    char_end: int            # -> Document.text
    text: str                # rendered form shown to retriever and model
    meta: dict[str, Any] = field(default_factory=dict)

    def overlap(self, start: int, end: int) -> int:
        return max(0, min(self.char_end, end) - max(self.char_start, start))


@dataclass(slots=True)
class GoldSpan:
    doc_id: str
    char_start: int
    char_end: int


@dataclass(slots=True)
class Question:
    qid: str
    intent: str
    query: str
    answerable: bool
    gold: GoldSpan | None = None
    gold_answer: str | None = None
    oracle_ref: str | None = None     # pack-independent ceiling context
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Retrieved:
    chunk_id: str
    score: float
    rank: int


@dataclass(slots=True)
class AgentResult:
    """The single contract between host and agent. Nothing leaks around it."""
    text: str
    retrieved: list[Retrieved] = field(default_factory=list)
    injected_chunk_ids: list[str] = field(default_factory=list)
    injected_tokens: int = 0
    skipped_chunk_ids: list[str] = field(default_factory=list)   # retrieved but over budget
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None


@dataclass(slots=True)
class Trace:
    """One row of evidence: append-only, resumable, config-stamped."""
    run_id: str
    config_hash: str
    spec_key: str
    repeat: int
    qid: str
    intent: str
    agent: str
    mode: str
    pack: str | None
    query: str
    answer: str
    retrieved: list[dict[str, Any]]
    injected_chunk_ids: list[str]
    injected_tokens: int
    skipped_chunk_ids: list[str]
    gold_chunk_ids: list[str]
    latency_ms: float
    usage: dict[str, Any]
    error: str | None = None
    scores: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)