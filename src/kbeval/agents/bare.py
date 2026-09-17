"""No knowledge injected. The floor of every measurement.

bare is not a control that can be skipped: absolute scores mix "this model is
already good at telecom" with "this data is good", and only the difference from
bare separates them. It is also pack-independent, which is what makes the sweep
N x (2 + M) instead of a cross product.
"""
from __future__ import annotations

from ..registry import AGENTS
from ..types import Question
from .base import ContextAgent, Material, no_material


@AGENTS.register("bare")
class BareAgent(ContextAgent):
    def material(self, q: Question) -> Material:
        return no_material(self.deps)
