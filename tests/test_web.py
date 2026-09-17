"""S10 -- the viewer, judged on what it cannot do.

The demo's only real risk is that it becomes a second way to produce evidence.
Two runs of the same experiment reached by two different paths cannot both be
the coordinate, so the tests that matter here assert absence: no writes under
runs/, no run ids minted, no scoring, and an explore page that says so in
plain Korean rather than only in the code.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kbeval import web
from kbeval.config import load_config
from kbeval.eval import score_run
from kbeval.runner import run_sweep

ROUTES = ["/", "/retrieval", "/questions", "/q/qa_001", "/q/qa_101", "/explore"]


@pytest.fixture
def site(base_yaml, tmp_path, monkeypatch):
    """A viewer pointed at one finished, scored run inside tmp_path."""
    override = tmp_path / "local.yaml"
    override.write_text(f"run:\n  out_dir: {tmp_path / 'runs'}\n", encoding="utf-8")
    monkeypatch.setattr(web, "CONFIG", [str(base_yaml), str(override)])

    cfg = load_config([base_yaml, override])
    root = run_sweep(cfg).root
    score_run(cfg, root)
    return TestClient(web.app), Path(cfg.run.out_dir), cfg


def snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


# --- the contract ---------------------------------------------------------

@pytest.mark.parametrize("path", ROUTES)
def test_every_page_renders(site, path):
    client, _, _ = site
    r = client.get(path)
    assert r.status_code == 200 and "<title>" in r.text


def test_browsing_never_writes_under_runs(site):
    """The whole point. A viewer that appends is not a viewer."""
    client, runs, _ = site
    before = snapshot(runs)
    for path in ROUTES:
        client.get(path)
    client.get("/explore", params={"q": "E-411 인증 실패 원인은?", "pack": "mid"})
    assert snapshot(runs) == before


def test_exploring_mints_no_run(site):
    client, runs, _ = site
    before = {p.name for p in runs.iterdir()}
    client.get("/explore", params={"q": "광 수신 레벨이 얼마 이하면 문제인가요?", "pack": "full"})
    assert {p.name for p in runs.iterdir()} == before


def test_explore_answers_all_three_modes(site):
    client, _, _ = site
    r = client.get("/explore", params={"q": "E-411 조치 방법", "pack": "mid"})
    assert r.status_code == 200
    for mode in ("bare", "rag", "oracle_ref"):
        assert mode in r.text


def test_explore_says_it_is_not_a_measurement(site):
    """Said on the page, not only in the code: the reader is the one who could
    mistake an exploratory answer for a result."""
    client, _, _ = site
    body = client.get("/explore").text
    assert "측정이 아닙니다" in body
    assert "runs/" in body and "기록되지 않" in body


def test_explore_admits_oracle_ref_is_degenerate_without_gold(site):
    """No labelled reference exists for an ad-hoc question, so oracle_ref falls
    back to bare. Showing it unlabelled would look like a null result."""
    client, _, _ = site
    assert "참조 문단 없음" in client.get("/explore", params={"q": "테스트"}).text


def test_the_viewer_cannot_reach_the_write_path():
    """Enforced by what the module imports, not by intent."""
    code = Path(web.__file__).read_text(encoding="utf-8")
    code = re.sub(r'"""(?:.|\n)*?"""', "", code)
    for forbidden in ("run_sweep", "score_run", "TraceWriter", "open(", "write_text"):
        assert forbidden not in code, forbidden


# --- honesty in the UI ----------------------------------------------------

def test_a_mock_run_is_labelled_as_one(site):
    client, _, _ = site
    assert "mock 실행 결과입니다" in client.get("/").text


def test_a_real_run_carries_no_mock_banner(site, base_yaml, tmp_path, monkeypatch):
    override = tmp_path / "real.yaml"
    override.write_text(
        f"run:\n  out_dir: {tmp_path / 'runs'}\n"
        'agents:\n  mock_a: {provider: bedrock, id: "qwen.qwen3-32b"}\n', encoding="utf-8")
    monkeypatch.setattr(web, "CONFIG", [str(base_yaml), str(override)])
    assert "mock 실행 결과입니다" not in TestClient(web.app).get("/").text


def test_covariates_are_shown_beside_the_retrieval_numbers(site):
    client, _, _ = site
    body = client.get("/retrieval").text
    assert "n_chunks" in body or "chunks" in body
    assert "gold injected" in body and "lost to budget" in body


def test_the_health_strip_is_always_visible(site):
    client, _, _ = site
    body = client.get("/").text
    assert "errors" in body and "format_fail" in body and "unreadable" in body


# --- robustness -----------------------------------------------------------

def test_an_unknown_question_does_not_crash(site):
    client, _, _ = site
    r = client.get("/q/qa_does_not_exist")
    assert r.status_code == 200 and "없는 문항" in r.text


def test_no_runs_yet_explains_what_to_do(base_yaml, tmp_path, monkeypatch):
    override = tmp_path / "empty.yaml"
    override.write_text(f"run:\n  out_dir: {tmp_path / 'nothing'}\n", encoding="utf-8")
    monkeypatch.setattr(web, "CONFIG", [str(base_yaml), str(override)])
    r = TestClient(web.app).get("/")
    assert r.status_code == 200 and "kbeval run" in r.text


def test_an_unscored_run_still_renders(base_yaml, tmp_path, monkeypatch):
    override = tmp_path / "unscored.yaml"
    override.write_text(f"run:\n  out_dir: {tmp_path / 'runs'}\n", encoding="utf-8")
    monkeypatch.setattr(web, "CONFIG", [str(base_yaml), str(override)])
    run_sweep(load_config([base_yaml, override]))
    r = TestClient(web.app).get("/")
    assert r.status_code == 200 and "kbeval score" in r.text


def test_user_text_is_escaped(site):
    """The question box is the one field a reader controls."""
    client, _, _ = site
    r = client.get("/explore", params={"q": "<script>alert(1)</script>"})
    assert "<script>alert(1)</script>" not in r.text
    assert "&lt;script&gt;" in r.text


def test_an_unknown_run_id_falls_back_instead_of_failing(site):
    client, _, _ = site
    assert client.get("/", params={"run": "../etc"}).status_code == 200


def test_an_unknown_pack_falls_back(site):
    client, _, _ = site
    assert client.get("/explore", params={"q": "테스트", "pack": "ghost"}).status_code == 200
