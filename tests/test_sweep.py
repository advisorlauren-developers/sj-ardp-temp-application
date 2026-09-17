"""S1 -- the sweep is not a cross product.

`bare` sees no corpus and `oracle_ref` injects a pack-independent passage, so
pairing either with a pack name would buy identical rows at full inference
price. The saving is a definition, not an optimisation, and it is worth a test
because the obvious refactor (three nested loops) silently restores the cost.
"""
from __future__ import annotations

import pytest

from kbeval.config import load_config
from kbeval.sweep import PACK_FREE_MODES, RunSpec, expand_sweep


def test_spec_count_matches_the_documented_formula(cfg):
    specs = expand_sweep(cfg)
    n, m = len(cfg.sweep.agent), len(cfg.sweep.pack)
    assert len(specs) == n * (2 + m) == 10
    assert n * len(cfg.sweep.mode) * m == 18      # what the cross product would cost


def test_pack_free_modes_carry_no_pack(cfg):
    for s in expand_sweep(cfg):
        if s.mode in PACK_FREE_MODES:
            assert s.pack is None


def test_rag_covers_every_pack(cfg):
    rag = {(s.agent, s.pack) for s in expand_sweep(cfg) if s.mode == "rag"}
    assert rag == {(a, p) for a in cfg.sweep.agent for p in cfg.sweep.pack}


def test_spec_keys_are_unique(cfg):
    keys = [s.key for s in expand_sweep(cfg)]
    assert len(keys) == len(set(keys))


def test_adding_a_pack_costs_one_spec_per_agent(cfg, base_yaml):
    grown = load_config([base_yaml], [
        'packs.extra={"name":"extra","chunker":"heading","transforms":[]}',
        'sweep.pack=["raw","mid","full","extra"]',
    ])
    assert len(expand_sweep(grown)) - len(expand_sweep(cfg)) == len(cfg.sweep.agent)


def test_dropping_rag_leaves_only_the_endpoints(cfg, base_yaml):
    ends = load_config([base_yaml], ['sweep.mode=["bare","oracle_ref"]'])
    specs = expand_sweep(ends)
    assert len(specs) == 2 * len(cfg.sweep.agent)
    assert all(s.pack is None for s in specs)


@pytest.mark.parametrize("mode", ["bare", "oracle_ref"])
def test_pack_free_modes_are_declared(mode):
    assert mode in PACK_FREE_MODES


def test_key_format_is_stable():
    assert RunSpec("qwen", "rag", "mid").key == "qwen|rag|mid"
    assert RunSpec("qwen", "bare", None).key == "qwen|bare|-"
