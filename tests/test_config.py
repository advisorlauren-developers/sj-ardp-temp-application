"""S1 -- config is the coordinate system of an experiment.

Two properties are load-bearing and everything else here supports them:
a typo must be an error, and any change to a control variable must change the
hash. If either fails, two different experiments can occupy the same coordinate
and the whole results table becomes unreadable after the fact.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from kbeval.config import Config, dump_resolved, load_config


def test_base_config_loads(cfg):
    assert set(cfg.packs) == {"raw", "mid", "full"}
    assert cfg.eval.questions == "data/eval/sample_questions.jsonl"
    assert cfg.source.adapter == "pdf_docs"


def test_unknown_top_level_key_is_an_error(raw_cfg):
    raw_cfg["retreival"] = {"k": 5}          # plausible typo, silently ignored elsewhere
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Config.model_validate(raw_cfg)


def test_unknown_nested_key_is_an_error(raw_cfg):
    raw_cfg["fixed"]["context_tokens_budget"] = 4000
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Config.model_validate(raw_cfg)


def test_unknown_pack_key_is_an_error(raw_cfg):
    raw_cfg["packs"]["mid"]["chunk_char"] = 500
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Config.model_validate(raw_cfg)


def test_hash_is_deterministic(cfg, base_yaml):
    assert cfg.hash == load_config([base_yaml]).hash


@pytest.mark.parametrize("override", [
    "fixed.context_token_budget=4000",
    "fixed.max_injected_chunks=4",
    "fixed.chars_per_token=2.5",
    "fixed.prompt_version=v2",
    "eval.gold_overlap_ratio=0.5",
    "packs.mid.chunk_chars=600",
    "source.adapter=jira_export",
])
def test_changing_a_control_variable_changes_the_hash(cfg, base_yaml, override):
    other = load_config([base_yaml], [override])
    assert other.hash != cfg.hash, f"{override} left the hash untouched"


def test_run_id_carries_the_hash(cfg):
    assert cfg.run_id == f"{cfg.run.name}-{cfg.hash}"


def test_override_coerces_json_scalars(base_yaml):
    c = load_config([base_yaml], ["fixed.n_repeat=3", "run.name=smoke"])
    assert c.fixed.n_repeat == 3 and isinstance(c.fixed.n_repeat, int)
    assert c.run.name == "smoke"


def test_override_coerces_json_lists(base_yaml):
    c = load_config([base_yaml], ['sweep.pack=["raw","mid"]'])
    assert c.sweep.pack == ["raw", "mid"]


def test_layers_merge_deeply(tmp_path, base_yaml):
    profile = tmp_path / "profile.yaml"
    profile.write_text("fixed:\n  n_repeat: 5\n", encoding="utf-8")
    c = load_config([base_yaml, profile])
    assert c.fixed.n_repeat == 5
    assert c.fixed.context_token_budget == 3000   # sibling keys survive the merge


def test_sweep_must_reference_declared_agents(raw_cfg):
    raw_cfg["sweep"]["agent"] = ["mock_a", "ghost"]
    with pytest.raises(ValidationError, match="ghost"):
        Config.model_validate(raw_cfg)


def test_sweep_must_reference_declared_packs(raw_cfg):
    raw_cfg["sweep"]["pack"] = ["raw", "ghost"]
    with pytest.raises(ValidationError, match="ghost"):
        Config.model_validate(raw_cfg)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_gold_overlap_ratio_is_bounded(raw_cfg, bad):
    raw_cfg["eval"]["gold_overlap_ratio"] = bad
    with pytest.raises(ValidationError):
        Config.model_validate(raw_cfg)


def test_canonical_form_keeps_korean_readable(cfg):
    assert "\\u" not in cfg.canonical()


def test_resolved_dump_round_trips_to_the_same_hash(cfg, tmp_path):
    out = tmp_path / "config.resolved.yaml"
    dump_resolved(cfg, out)
    assert load_config([out]).hash == cfg.hash


# --- the trace coordinate and the grading coordinate are separate ---------

def test_swapping_the_judge_does_not_move_the_run_id(cfg, base_yaml):
    """The judge does not change a single answer -- it changes how answers are
    graded. Folding it into the run id would invalidate the coordinate of
    traces it never touched, and a rerun would have to buy them again."""
    other = load_config([base_yaml],
                        ['judge={"agent":{"provider":"bedrock","id":"us.anthropic.claude-opus-5"}}'])
    assert other.hash == cfg.hash
    assert other.scoring_hash != cfg.scoring_hash


def test_scoring_thresholds_belong_to_the_grading_coordinate(cfg, base_yaml):
    other = load_config([base_yaml], ["scoring.min_headroom=0.2"])
    assert other.hash == cfg.hash
    assert other.scoring_hash != cfg.scoring_hash


def test_editing_the_judge_prompt_moves_only_the_grading_coordinate(cfg, base_yaml, tmp_path):
    edited = tmp_path / "judge.md"
    edited.write_text(cfg.prompts.read("judge") + "\n추가 지시.\n", encoding="utf-8")
    other = load_config([base_yaml], [f"prompts.judge={edited}"])
    assert other.hash == cfg.hash
    assert other.scoring_hash != cfg.scoring_hash


def test_editing_a_candidate_prompt_moves_the_run_id(cfg, base_yaml, tmp_path):
    edited = tmp_path / "system.md"
    edited.write_text(cfg.prompts.read("system_qa") + "\n6. 한 줄 더.\n", encoding="utf-8")
    other = load_config([base_yaml], [f"prompts.system_qa={edited}"])
    assert other.hash != cfg.hash


@pytest.mark.parametrize("override", [
    "fixed.context_token_budget=4000",
    "eval.gold_overlap_ratio=0.5",
    "retrieval.candidate_k=3",
    "run.seed=7",
])
def test_control_variables_still_move_the_run_id(cfg, base_yaml, override):
    assert load_config([base_yaml], [override]).hash != cfg.hash
