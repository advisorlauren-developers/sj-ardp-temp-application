"""Sweep expansion.

The sweep is NOT a full cross product. `bare` and `oracle_ref` are
pack-independent by construction -- bare sees no knowledge at all, and
oracle_ref injects a curated reference passage that is identical for every
pack. So 3 models x 3 packs is 3 + 3 + 9 = 15 specs, not 45.

That is a third of the budget, recovered by a definition rather than a
shortcut.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Config

PACK_FREE_MODES = {"bare", "oracle_ref"}


@dataclass(frozen=True)
class RunSpec:
    agent: str
    mode: str
    pack: str | None

    @property
    def key(self) -> str:
        return f"{self.agent}|{self.mode}|{self.pack or '-'}"


def expand_sweep(cfg: Config) -> list[RunSpec]:
    """`kbeval plan` is the cheapest place to catch a typo, so the mode axis is
    validated here rather than halfway through a sweep that has already spent
    money. The local import is what populates the registry."""
    from . import agents  # noqa: F401  (registration side effect)
    from .registry import AGENTS

    for m in cfg.sweep.mode:
        AGENTS.get(m)

    specs: list[RunSpec] = []
    for a in cfg.sweep.agent:
        for m in cfg.sweep.mode:
            if m in PACK_FREE_MODES:
                specs.append(RunSpec(a, m, None))
            else:
                specs.extend(RunSpec(a, m, p) for p in cfg.sweep.pack)
    return specs