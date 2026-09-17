"""S6 -- the provider layer, where an outage must not look like a bad pack.

Rule 8 and rule 10 both live here, and both are about what NEVER reaches the
scorer: a failed call must arrive as an error rather than an empty answer, and
reasoning tokens must not arrive at all.
"""
from __future__ import annotations

import pytest

from kbeval.config import AgentSpec, LLMCfg
from kbeval.llm import build_llm, strip_thinking, validate
from kbeval.llm.bedrock import BedrockLLM, _inference_config, _is_transient
from kbeval.llm.mock import ABSTAIN
from kbeval.registry import LLMS

SPEC_A = AgentSpec(provider="mock", id="mock-a")
SPEC_B = AgentSpec(provider="mock", id="mock-b")


# --- rule 10: thinking never reaches the judge ---------------------------

@pytest.mark.parametrize("tag", ["think", "thinking", "reasoning", "scratchpad"])
def test_tag_wrapped_reasoning_is_removed(tag):
    text, had = strip_thinking(f"<{tag}>속으로 고민한다</{tag}>\n실제 답변이다.")
    assert text == "실제 답변이다." and had is True


def test_an_unclosed_tag_does_not_leak_the_whole_response():
    """A model that hits max_tokens mid-thought never closes the tag. Left
    alone, the entire response would be reasoning scored as an answer."""
    text, had = strip_thinking("<think>토큰이 모자라 여기서 끊긴다")
    assert text == "" and had is True


def test_plain_answers_are_untouched():
    assert strip_thinking("그냥 답변이다.") == ("그냥 답변이다.", False)


def test_stripping_is_case_insensitive():
    assert strip_thinking("<THINK>x</THINK>답")[0] == "답"


# --- rule 8: a failure is an error, not an answer ------------------------

def test_an_empty_body_is_an_error():
    r = validate("", 1.0, {})
    assert r.error == "empty_response" and r.text == ""


def test_a_response_that_was_only_reasoning_is_an_error():
    r = validate("<think>...</think>", 1.0, {})
    assert r.error == "empty_response" and r.had_thinking is True


def test_a_real_answer_carries_no_error():
    assert validate("답변이다.", 1.0, {"tokens": 3}).error is None


def test_abstention_is_not_an_error():
    """`답변 불가` is a deliberate model decision and must survive as an answer
    -- conflating it with an outage would make infra failures look like
    correct caution."""
    assert validate(ABSTAIN, 1.0, {}).error is None


# --- the mock is a first-class citizen -----------------------------------

def test_mock_answers_from_the_material_it_was_given():
    llm = build_llm(SPEC_A, LLMCfg())
    r = llm.generate("sys", "E-411은 ONT 시리얼 미등록이다.\n광 수신 레벨 저하.\n\nE-411 원인은?")
    assert "E-411" in r.text and r.error is None


def test_mock_abstains_with_no_material():
    llm = build_llm(SPEC_A, LLMCfg())
    assert llm.generate("sys", "(제공된 자료 없음)\n\n완전히 무관한 질문?").text == ABSTAIN


def test_mock_is_deterministic():
    llm = build_llm(SPEC_B, LLMCfg())
    prompt = "자료 한 줄.\n\n질문 한 줄?"
    assert {llm.generate("s", prompt).text for _ in range(5)} == {llm.generate("s", prompt).text}


def test_the_two_mocks_do_not_behave_identically():
    """kendall tau needs two models that can disagree; identical mocks would
    make the cross-model check vacuous."""
    a, b = build_llm(SPEC_A, LLMCfg()), build_llm(SPEC_B, LLMCfg())
    prompts = [f"자료 {i} 내용이다.\n\n자료 {i} 내용?" for i in range(30)]
    assert [a.generate("s", p).text for p in prompts] != [b.generate("s", p).text for p in prompts]


def test_the_weak_mock_exercises_the_thinking_stripper():
    r = build_llm(SPEC_B, LLMCfg()).generate("s", "자료.\n\n질문?")
    assert r.had_thinking is True
    assert "<think>" not in r.text


def test_the_candidate_mock_never_sees_a_question_object():
    """It gets the rendered prompt, the same surface a real provider gets, so
    it cannot peek at gold even by accident.

    Scoped to MockLLM, not the module: MockJudge lives beside it and is SUPPOSED
    to see gold_answer -- that is what it grades against."""
    import inspect
    import re

    from kbeval.llm.mock import MockLLM
    code = re.sub(r'"""(?:.|\n)*?"""', "", inspect.getsource(MockLLM))
    code = re.sub(r"#.*", "", code)
    assert "Question" not in code
    assert "gold" not in code.lower()


# --- bedrock: the traps, without AWS -------------------------------------

def test_top_p_is_omitted_when_unset():
    """Several families reject temperature and top_p together, and the refusal
    reads like a bad model id."""
    assert "topP" not in _inference_config(LLMCfg())
    assert _inference_config(LLMCfg(top_p=0.9))["topP"] == 0.9


@pytest.mark.parametrize("msg,transient", [
    ("ThrottlingException: rate exceeded", True),
    ("ServiceUnavailable", True),
    ("Read timeout on endpoint", True),
    ("ValidationException: model not found", False),
    ("AccessDeniedException", False),
])
def test_only_transient_failures_are_retried(msg, transient):
    assert _is_transient(msg) is transient


def test_a_failing_call_returns_an_error_instead_of_raising(monkeypatch):
    llm = BedrockLLM(AgentSpec(provider="bedrock", id="qwen.qwen3-32b"), LLMCfg(max_retries=2))
    monkeypatch.setattr(llm, "_converse",
                        lambda s, u: (_ for _ in ()).throw(RuntimeError("ValidationException")))
    r = llm.generate("sys", "user")
    assert r.error and "ValidationException" in r.error
    assert r.text == ""


def test_a_permanent_failure_is_not_retried(monkeypatch):
    calls = []

    def boom(s, u):
        calls.append(1)
        raise RuntimeError("AccessDeniedException")

    llm = BedrockLLM(AgentSpec(provider="bedrock", id="x"), LLMCfg(max_retries=5))
    monkeypatch.setattr(llm, "_converse", boom)
    llm.generate("s", "u")
    assert len(calls) == 1


def test_a_transient_failure_is_retried_to_the_limit(monkeypatch):
    calls = []

    def boom(s, u):
        calls.append(1)
        raise RuntimeError("ThrottlingException")

    llm = BedrockLLM(AgentSpec(provider="bedrock", id="x"), LLMCfg(max_retries=3))
    monkeypatch.setattr(llm, "_converse", boom)
    monkeypatch.setattr("kbeval.llm.bedrock.time.sleep", lambda s: None)
    llm.generate("s", "u")
    assert len(calls) == 3


def test_a_200_with_an_empty_body_is_an_error_not_an_answer(monkeypatch):
    """The Bedrock trap: HTTP 200 carrying an error document. Scoring that as a
    wrong answer would read as 'this pack made the model worse'."""
    llm = BedrockLLM(AgentSpec(provider="bedrock", id="x"), LLMCfg())
    monkeypatch.setattr(llm, "_converse", lambda s, u: ("", {"provider": "bedrock"}))
    assert llm.generate("s", "u").error == "empty_response"


def test_mantle_refuses_to_run_without_a_token(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    llm = BedrockLLM(AgentSpec(provider="bedrock", endpoint="mantle", id="x"),
                     LLMCfg(max_retries=1))
    assert "AWS_BEARER_TOKEN_BEDROCK" in llm.generate("s", "u").error


def test_both_providers_are_registered():
    assert {"mock", "bedrock"} <= set(LLMS.names())
