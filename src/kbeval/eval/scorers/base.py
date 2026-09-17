"""The Scorer contract.

A scorer turns one trace row into a dict of numbers. Keeping them separate from
the runner is what makes scoring re-runnable: a judge prompt can be revised and
the whole run re-scored without paying for a single candidate call again.

The hard rule every scorer obeys: a row with an `error` is NOT scored. Rule 8
says a failed call is not an answer, and a scorer that quietly returned 0 for it
would turn an outage into a data-quality finding.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ...config import Config
from ...registry import SCORERS
from ...types import Question


@runtime_checkable
class Scorer(Protocol):
    name: str

    def score(self, row: dict[str, Any], q: Question) -> dict[str, Any]:
        ...


def build_scorer(name: str, cfg: Config) -> Scorer:
    return SCORERS.get(name)(cfg)
