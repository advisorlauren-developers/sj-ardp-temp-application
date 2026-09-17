"""The ceiling: a curated reference passage, identical for every pack.

Rule 5. The denominator of realization_rate has to be pack-INDEPENDENT. If each
pack supplied its own ceiling, a bad pack would lower its own denominator and
score a flattering realization rate -- the worse the data, the better it would
look.

An unanswerable question has no reference by construction, so this mode
degenerates to bare for those. That is correct: there is no ceiling to reach,
and they are scored on abstention instead.
"""
from __future__ import annotations

from ..registry import AGENTS
from ..types import Question
from .base import ContextAgent, Material, count, no_material


@AGENTS.register("oracle_ref")
class OracleRefAgent(ContextAgent):
    def material(self, q: Question) -> Material:
        if not q.oracle_ref:
            return no_material(self.deps)
        return Material(text=q.oracle_ref, tokens=count(q.oracle_ref, self.deps.fixed))
