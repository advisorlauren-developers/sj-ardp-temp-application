"""Shared fixtures.

Two things every test in this suite depends on:

1. The harness reads relative paths (data/raw/docs, configs/base.yaml) out of
   the repo root, so the suite pins the working directory instead of passing
   absolute paths around. A test that only passes from one directory is not a
   guarantee.
2. build_pack() writes into data/packs/. Tests redirect that at a tmp dir, so
   running the suite can never leave a half-built pack behind for the next
   `kbeval run` to pick up.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from kbeval import wiring
from kbeval.config import Config, load_config
from kbeval.packs import builder
from kbeval.sources import load_documents

ROOT = Path(__file__).resolve().parent.parent
BASE_YAML = ROOT / "configs" / "base.yaml"


@pytest.fixture(autouse=True, scope="session")
def _at_repo_root():
    prev = Path.cwd()
    os.chdir(ROOT)
    yield
    os.chdir(prev)


@pytest.fixture(autouse=True)
def _isolated_pack_root(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "PACK_ROOT", tmp_path / "packs")
    wiring.reset_cache()


@pytest.fixture
def base_yaml() -> Path:
    return BASE_YAML


@pytest.fixture
def cfg() -> Config:
    return load_config([BASE_YAML])


@pytest.fixture
def raw_cfg() -> dict:
    """base.yaml as a plain dict, for tests that need to corrupt it."""
    import yaml
    return yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))


@pytest.fixture
def docs(cfg):
    return load_documents(cfg.source)


@pytest.fixture
def docs_by_id(docs):
    return {d.doc_id: d for d in docs}


@pytest.fixture
def questions(cfg):
    from kbeval.packs import load_questions
    return load_questions(cfg.eval.questions, cfg.eval.intents)
