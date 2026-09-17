"""The three knowledge-injection modes, and the one thing they must share.

Rule 6 is not a convention here, it is the class layout: `answer()` is written
ONCE, in ContextAgent, and a mode may only override what goes into the context
slot. System prompt, user template, decoding settings and response handling are
therefore identical for bare, rag and oracle_ref by construction.

That matters because realization_rate = (rag - bare) / (oracle_ref - bare) is a
ratio of differences between these three. If oracle_ref quietly got a friendlier
prompt, the denominator would inflate and every pack's realization would sink
together -- a corruption that looks exactly like a real result.

The modes differ in one line each:
    bare        no material at all          -- what the model already knows
    rag         retrieved pack chunks       -- what this pack delivered
    oracle_ref  the curated passage         -- what was there to be had
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..budget import estimate_tokens
from ..config import FixedCfg, PromptsCfg
from ..llm.base import LLMClient
from ..registry import AGENTS
from ..types import AgentResult, Question, Retrieved


@dataclass(frozen=True)
class Material:
    """What lands in the context slot of the shared template."""
    text: str
    chunk_ids: list[str] = field(default_factory=list)
    tokens: int = 0
    retrieved: list[Retrieved] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


@dataclass
class AgentDeps:
    llm: LLMClient
    prompts: PromptsCfg
    fixed: FixedCfg
    pack: object | None = None          # Pack; typed loosely to keep agents/ free of packs/
    retriever: object | None = None     # Retriever, already indexed on that pack
    candidate_k: int = 10               # retrieval depth; the BUDGET decides what gets in


class ContextAgent:
    """Shared body. Subclasses override `material()` and nothing else."""

    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps
        self.system = deps.prompts.read("system_qa")
        self.template = deps.prompts.read("context")
        self.chunk_template = deps.prompts.read("chunk")

    def material(self, q: Question) -> Material:
        raise NotImplementedError

    def render(self, q: Question) -> tuple[str, Material]:
        m = self.material(q)
        return self.template.format(context=m.text, question=q.query), m

    def answer(self, q: Question) -> AgentResult:
        t0 = time.perf_counter()
        user, m = self.render(q)
        r = self.deps.llm.generate(self.system, user)
        return AgentResult(
            text=r.text,
            retrieved=m.retrieved,
            injected_chunk_ids=m.chunk_ids,
            injected_tokens=m.tokens,
            skipped_chunk_ids=m.skipped,
            usage=dict(r.usage, had_thinking=r.had_thinking),
            latency_ms=r.latency_ms or (time.perf_counter() - t0) * 1000,
            error=r.error,
        )


def no_material(deps: AgentDeps) -> Material:
    return Material(text=deps.prompts.no_material)


def count(text: str, fixed: FixedCfg) -> int:
    return estimate_tokens(text, fixed.chars_per_token)


def build_agent(mode: str, deps: AgentDeps) -> ContextAgent:
    return AGENTS.get(mode)(deps)
