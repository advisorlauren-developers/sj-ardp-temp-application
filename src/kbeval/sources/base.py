"""The SourceAdapter contract -- and the JIRA extension point.

An adapter's only job is to produce Documents whose `text` is the canonical
coordinate space for gold labelling. That makes this the highest-stakes layer
in the project: anything an adapter flattens into `text` becomes addressable by
a gold span and retrievable later; anything it drops is invisible to the entire
experiment, and no amount of retrieval tuning downstream can recover it.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..config import SourceCfg
from ..registry import SOURCES
from ..types import Document


@runtime_checkable
class SourceAdapter(Protocol):
    def load(self) -> list[Document]:
        ...


def load_documents(cfg: SourceCfg) -> list[Document]:
    """The only entry point. Callers never name a concrete adapter, so adding
    JIRA later changes one config line and no code outside sources/."""
    return SOURCES.get(cfg.adapter)(cfg).load()