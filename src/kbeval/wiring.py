"""Assembling one spec into a runnable agent -- in exactly one place.

The CLI (`kbeval ask`) and the runner (S7) must build agents identically. If
each wired its own, a difference between them would show up as a difference
between an eyeballed answer and a recorded trace, and the recorded one is what
the report is built from.

This is also the only module that knows all three axes at once, which is why it
sits at the top level: agents/ stays ignorant of packs/, packs/ stays ignorant
of retrieval/, and the knowledge of how they fit together lives here rather
than leaking into any of them.
"""
from __future__ import annotations

from .agents import AgentDeps, ContextAgent, build_agent
from .config import Config
from .llm import build_llm
from .packs import Pack, build_pack
from .registry import AGENTS
from .retrieval import build_retriever
from .sweep import PACK_FREE_MODES

_PACKS: dict[tuple[str, str], Pack] = {}


def reset_cache() -> None:
    """Tests redirect where packs are written; a cache that outlives that would
    hand back a Pack pointing at a directory that no longer exists."""
    _PACKS.clear()


def pack_for(cfg: Config, name: str) -> Pack:
    """Packs are deterministic for a given config and a sweep asks for the same
    one N times. Keyed by config hash, so a changed control variable can never
    be served a stale pack."""
    key = (cfg.hash, name)
    if key not in _PACKS:
        _PACKS[key] = build_pack(cfg, name)
    return _PACKS[key]


def build(cfg: Config, agent: str, mode: str, pack: str | None) -> ContextAgent:
    AGENTS.get(mode)          # an unknown mode must say so, not blame the pack
    if agent not in cfg.agents:
        raise KeyError(f"unknown agent '{agent}'. available: {sorted(cfg.agents)}")

    if mode in PACK_FREE_MODES:
        if pack is not None:
            raise ValueError(f"mode '{mode}' is pack-independent; got pack='{pack}'")
        deps = AgentDeps(llm=build_llm(cfg.agents[agent], cfg.llm),
                         prompts=cfg.prompts, fixed=cfg.fixed)
        return build_agent(mode, deps)

    if pack is None:
        raise ValueError(f"mode '{mode}' needs a pack")

    pk = pack_for(cfg, pack)
    retriever = build_retriever(cfg.retrieval)
    retriever.index(pk.chunks)
    deps = AgentDeps(llm=build_llm(cfg.agents[agent], cfg.llm),
                     prompts=cfg.prompts, fixed=cfg.fixed,
                     pack=pk, retriever=retriever,
                     candidate_k=cfg.retrieval.candidate_k)
    return build_agent(mode, deps)
