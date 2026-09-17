"""The axis boundaries, asserted instead of documented.

CLAUDE.md states these as grep commands a human is supposed to run. A grep
nobody runs is a comment, so they live here: the three axes must not be able to
see each other, or a result stops being attributable to one of them.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "kbeval"
PACKAGES = ["sources", "packs", "retrieval"]


def _sources(pkg: str) -> list[Path]:
    d = SRC / pkg
    return sorted(p for p in d.glob("*.py")) if d.exists() else []


def _code(path: Path) -> str:
    """Strip docstrings and comments -- prose may name anything, code may not."""
    s = path.read_text(encoding="utf-8")
    s = re.sub(r'"""(?:.|\n)*?"""', "", s)
    s = re.sub(r"#.*", "", s)
    return s


def test_packs_cannot_see_retrieval():
    """A pack that knew its retriever could tune search to itself, and the pack
    axis would stop measuring the data."""
    for p in _sources("packs"):
        assert not re.search(r"Retriever|retriev", _code(p)), p.name


def test_packs_and_retrieval_cannot_see_the_provider_sdk():
    for pkg in PACKAGES:
        for p in _sources(pkg):
            assert not re.search(r"boto3|modelId|bedrock", _code(p), re.IGNORECASE), p


@pytest.mark.skipif(not (SRC / "agents").exists(), reason="agents/ lands at S6")
def test_agents_cannot_see_the_provider_sdk():
    for p in _sources("agents"):
        assert not re.search(r"boto3|modelId|bedrock", _code(p), re.IGNORECASE), p


def test_the_search_path_never_sees_gold():
    """Scoring knows about gold by definition; searching must not. A retriever
    that can see the answer is not measuring anything, so the split is enforced
    file by file: metrics.py is the scoring half, everything else is search.
    """
    for p in _sources("retrieval"):
        if p.name == "metrics.py":
            continue
        assert "gold" not in _code(p).lower(), p.name


@pytest.mark.parametrize("pkg", PACKAGES)
def test_every_package_follows_the_folder_convention(pkg):
    names = {p.name for p in _sources(pkg)}
    assert "__init__.py" in names
    assert "base.py" in names or pkg == "packs"   # packs splits base into chunkers/transforms


@pytest.mark.parametrize("pkg", PACKAGES)
def test_implementations_are_imported_for_their_side_effects(pkg):
    """The registry only knows what has been imported. A missing line here is
    an ImportError at run time and the commonest mistake in this layout."""
    init = (SRC / pkg / "__init__.py").read_text(encoding="utf-8")
    impls = {p.stem for p in _sources(pkg)} - {"__init__", "base"}
    for impl in impls:
        assert re.search(rf"\b{impl}\b", init), f"{pkg}/{impl}.py is never imported"
