"""Retrieve from one pack, then fill the token budget. The measured mode.

Both halves are where a pack can win or lose, and they are kept separate on
purpose:
  * retrieval decides WHETHER the answer was reachable  -> hit@k, already known
    from S5 at zero cost;
  * budget packing decides whether it actually got in   -> injected_chunks.

A pack with great retrieval and chunks too fat for the budget loses here, and
that is a true finding about the data, not an artefact -- as long as nothing is
truncated, which is why budget.fit skips instead of cutting.
"""
from __future__ import annotations

from ..budget import fit
from ..registry import AGENTS
from ..types import Question
from .base import AgentDeps, ContextAgent, Material, count, no_material


@AGENTS.register("rag")
class RagAgent(ContextAgent):
    def __init__(self, deps: AgentDeps) -> None:
        super().__init__(deps)
        if deps.pack is None or deps.retriever is None:
            raise ValueError("rag needs a pack and an indexed retriever")

    def material(self, q: Question) -> Material:
        f = self.deps.fixed
        hits = self.deps.retriever.search(q.query, self.deps.candidate_k)
        packed = fit([self.deps.pack.get(h.chunk_id) for h in hits],
                     f.context_token_budget, f.max_injected_chunks, f.chars_per_token)
        if not packed.chunks:
            return Material(text=no_material(self.deps).text, retrieved=hits,
                            skipped=packed.skipped)

        body = "\n".join(
            self.chunk_template.format(chunk_id=c.chunk_id, text=c.text).strip()
            for c in packed.chunks)
        return Material(text=body, chunk_ids=[c.chunk_id for c in packed.chunks],
                        tokens=count(body, f), retrieved=hits, skipped=packed.skipped)
