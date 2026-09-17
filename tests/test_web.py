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

ROUTES = ["/", "/qa", "/bug", "/feature", "/report", "/retrieval",
          "/questions", "/q/qa_001", "/q/qa_101"]


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
    client.get("/api/ask", params={"q": "E-411 인증 실패 원인은?", "pack": "mid"})
    assert snapshot(runs) == before


def test_asking_mints_no_run(site):
    client, runs, _ = site
    before = {p.name for p in runs.iterdir()}
    client.get("/api/ask", params={"q": "광 수신 레벨이 얼마 이하면 문제인가요?", "pack": "full"})
    assert {p.name for p in runs.iterdir()} == before


def ask(client, q: str, pack: str = "mid") -> dict:
    return client.get("/api/ask", params={"q": q, "pack": pack}).json()


def test_the_answer_carries_the_passages_it_actually_retrieved(site):
    """The sources ARE the evidence that retrieval happened. An answer with no
    visible provenance is indistinguishable from the model making it up."""
    client, _, _ = site
    a = ask(client, "E-411 인증 실패 조치 방법")
    assert a["sources"]
    assert all({"chunk_id", "heading", "score", "rank", "text"} <= set(s) for s in a["sources"])
    assert any("show ont autofind" in s["text"] for s in a["sources"])


def test_the_answer_carries_its_own_baseline(site):
    """The whole point of the tool: the same model, the same prompt, material
    removed. Without it an answer is just an answer."""
    client, _, _ = site
    a = ask(client, "E-411 인증 실패 조치 방법")
    assert a["baseline"] == "답변 불가"          # bare has nothing to answer from
    assert a["answer"] != a["baseline"]


def test_only_injected_passages_are_offered_as_sources(site):
    """A chunk that lost to the token budget was never seen by the model, so
    listing it as a source would be a lie about provenance."""
    client, _, cfg_ = site
    a = ask(client, "E-411 인증 실패 조치 방법")

    from kbeval.types import Question
    from kbeval.wiring import build
    res = build(cfg_, cfg_.sweep.agent[0], "rag", "mid").answer(
        Question(qid="x", intent="qa", query="E-411 인증 실패 조치 방법", answerable=True))
    assert [s["chunk_id"] for s in a["sources"]] == res.injected_chunk_ids


def test_every_answer_names_the_dataset_level_and_the_budget(site):
    """Which pack answered is the independent variable, and whether the budget
    bound is the other half. Hiding either makes the demo unfalsifiable."""
    client, _, _ = site
    a = ask(client, "E-411 조치", pack="full")
    assert a["pack"] == "full" and "full" in a["pack_label"]
    assert a["injected_tokens"] <= a["budget"]
    assert a["candidates"] >= len(a["sources"])


def test_the_chat_is_single_turn(site):
    """Carrying the thread into the prompt would make the demo show behaviour
    the harness never measures. The endpoint has nowhere to put history."""
    client, _, _ = site
    first = ask(client, "E-411 원인은?")
    ask(client, "그럼 재부팅은요?")
    assert ask(client, "E-411 원인은?") == first      # no state accumulated anywhere


def test_qa_says_it_is_not_a_measurement(site):
    """Said on the page, not only in the code: the reader is the one who could
    mistake an exploratory answer for a result."""
    client, _, _ = site
    body = client.get("/qa").text
    assert "측정이 아닙니다" in body
    assert "runs/" in body and "기록되지 않" in body
    assert "각 질문은 독립적으로" in body


def test_qa_works_without_javascript_or_says_what_to_run(site):
    client, _, _ = site
    body = client.get("/qa").text
    assert "<noscript>" in body and "kbeval ask" in body


def test_an_empty_question_is_refused_rather_than_answered(site):
    client, _, _ = site
    assert ask(client, "   ")["error"] == "empty_question"


# --- the landing page -----------------------------------------------------

def test_the_landing_page_offers_the_three_intents(site):
    client, _, _ = site
    body = client.get("/").text
    for label in ("Q&amp;A", "버그", "기능 추가"):
        assert label in body


def test_only_configured_intents_are_clickable(site):
    """`intent` is a field on Question and eval.intents is a config list, so an
    inactive card is waiting on DATA, not on a screen."""
    client, _, _ = site
    body = client.get("/").text
    assert '<a class="pick" href="/qa">' in body
    assert body.count('class="pick off"') == 2


def test_an_inactive_intent_says_what_it_needs(site):
    client, _, _ = site
    body = client.get("/bug").text
    assert "준비 중" in body
    assert "eval.intents" in body and "questions.jsonl" in body


def test_an_inactive_intent_refuses_to_fake_an_answer(site):
    """An unlabelled answer cannot be judged good or bad -- that is an
    impression, not a measurement."""
    client, _, _ = site
    assert "측정이 아니라 인상" in client.get("/feature").text


def test_the_viewer_cannot_reach_the_write_path():
    """Enforced by what the module imports, not by intent."""
    code = Path(web.__file__).read_text(encoding="utf-8")
    code = re.sub(r'"""(?:.|\n)*?"""', "", code)
    for forbidden in ("run_sweep", "score_run", "TraceWriter", "open(", "write_text"):
        assert forbidden not in code, forbidden


# --- honesty in the UI ----------------------------------------------------

def test_a_mock_run_is_labelled_as_one(site):
    client, _, _ = site
    assert "mock 실행 결과입니다" in client.get("/report").text


def test_a_real_run_carries_no_mock_banner(site, base_yaml, tmp_path, monkeypatch):
    override = tmp_path / "real.yaml"
    override.write_text(
        f"run:\n  out_dir: {tmp_path / 'runs'}\n"
        'agents:\n  mock_a: {provider: bedrock, id: "qwen.qwen3-32b"}\n', encoding="utf-8")
    monkeypatch.setattr(web, "CONFIG", [str(base_yaml), str(override)])
    assert "mock 실행 결과입니다" not in TestClient(web.app).get("/report").text


def test_covariates_are_shown_beside_the_retrieval_numbers(site):
    client, _, _ = site
    body = client.get("/retrieval").text
    assert "n_chunks" in body or "chunks" in body
    assert "gold injected" in body and "lost to budget" in body


def test_the_health_strip_is_always_visible(site):
    client, _, _ = site
    body = client.get("/report").text
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
    r = TestClient(web.app).get("/report")
    assert r.status_code == 200 and "kbeval run" in r.text


def test_an_unscored_run_still_renders(base_yaml, tmp_path, monkeypatch):
    override = tmp_path / "unscored.yaml"
    override.write_text(f"run:\n  out_dir: {tmp_path / 'runs'}\n", encoding="utf-8")
    monkeypatch.setattr(web, "CONFIG", [str(base_yaml), str(override)])
    run_sweep(load_config([base_yaml, override]))
    r = TestClient(web.app).get("/report")
    assert r.status_code == 200 and "kbeval score" in r.text


def test_the_page_never_reflects_a_query_parameter(site):
    """The chat renders answers through textContent, and the shell echoes
    nothing back -- so there is no place for markup to land."""
    client, _, _ = site
    r = client.get("/qa", params={"q": "<script>alert(1)</script>", "pack": "mid"})
    assert "alert(1)" not in r.text


def test_answers_come_back_as_data_not_markup(site):
    """JSON, rendered with textContent. Nothing the corpus contains can become
    an element."""
    client, _, _ = site
    a = ask(client, "<img src=x onerror=alert(1)>")
    assert isinstance(a["answer"], str)
    assert "alert" not in client.get("/qa").text


def test_an_unknown_run_id_falls_back_instead_of_failing(site):
    client, _, _ = site
    assert client.get("/report", params={"run": "../etc"}).status_code == 200


def test_an_unknown_pack_falls_back(site):
    client, _, _ = site
    assert client.get("/api/ask", params={"q": "테스트", "pack": "ghost"}).status_code == 200


def test_the_pack_selector_names_the_transforms(site):
    """"변환 2개" vs "변환 4개" says something differs but not what -- which is
    the one thing the pack axis exists to make legible."""
    client, _, cfg_ = site
    from kbeval.web import pack_label
    body = client.get("/qa").text
    assert "strip_boilerplate" in body and "normalize_terms" in body
    assert pack_label(cfg_, "raw").endswith("변환 없음")


def test_packs_that_came_out_identical_are_called_out(site):
    """If two levels produce the same corpus, a reader who switches and sees
    the same answer reads it as a broken selector. It is neither that nor a
    null result -- those transforms had nothing to do on this corpus."""
    client, _, cfg_ = site
    from kbeval.web import identical_packs
    assert identical_packs(cfg_) == [["mid", "full"]]
    assert "청크가 완전히 동일합니다" in client.get("/qa").text


def test_no_such_note_when_the_packs_really_differ(site, base_yaml, tmp_path, monkeypatch):
    from kbeval.config import load_config
    from kbeval.web import identical_packs
    override = tmp_path / "two.yaml"
    override.write_text('sweep:\n  pack: ["raw", "mid"]\n', encoding="utf-8")
    assert identical_packs(load_config([base_yaml, override])) == []
