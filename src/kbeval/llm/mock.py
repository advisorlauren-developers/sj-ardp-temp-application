"""A deterministic stand-in for a real model. First-class, not a stub.

The whole pipeline -- retrieval, budget packing, prompt rendering, tracing,
scoring, reporting -- has to run end to end with no AWS account. Without that,
the first real Bedrock run is also the first integration test, and every
plumbing bug arrives disguised as a data-quality finding.

What it simulates, on purpose:
  * knowledge it was actually given: it answers from the sentence in the prompt
    that best overlaps the question, and says `답변 불가` when nothing overlaps.
    So bare abstains, rag and oracle_ref answer, and realization_rate is a real
    number instead of a constant;
  * a weaker second model (`mock-b`): deterministically flaky, so the two mock
    agents do not produce identical rankings and kendall tau has something to
    measure;
  * tag-wrapped reasoning (`mock-b`), which exercises the rule-10 stripper on
    every run rather than only in a unit test.

It never sees a Question object, only the rendered prompt -- the same surface a
real provider gets. It cannot peek at gold even by accident.

Its ceiling, stated so nobody reads past it: a lexical matcher cannot tell
"the material does not cover this" from "the material covers it in other
words". So on unanswerable questions in rag mode it answers from the
distractors instead of abstaining. That is a property of THIS mock, not of the
pipeline -- abstention is only measurable with a real model, and the scorer
that grades it is unit-tested on crafted answers instead.
"""
from __future__ import annotations

import hashlib
import json
import re
import time

from ..config import AgentSpec, LLMCfg
from ..registry import LLMS
from .base import LLMResponse, validate

ABSTAIN = "답변 불가"
_WORD_RX = re.compile(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*|[가-힣]+")

# The rag context arrives as "(chunk_id) body", and inline_meta puts a
# "[title: ... | heading_path: ...]" header on its own first line.
_ID_RX = re.compile(r"^\([^)\s]+\)\s*")
_META_RX = re.compile(r"^\[[^\]]*\]$")


def _content(line: str) -> str:
    """Strip the chunk id, and reject a line that is only a metadata header.

    A real model does not answer a question with the document's own title bar.
    Letting the mock do it would penalise exactly the packs that add metadata
    -- an artefact of this stand-in, reported as a property of the data.
    """
    body = _ID_RX.sub("", line).strip()
    return "" if _META_RX.match(body) else body


def _words(text: str) -> set[str]:
    return set(_WORD_RX.findall(text.lower()))


@LLMS.register("mock")
class MockLLM:
    def __init__(self, spec: AgentSpec, cfg: LLMCfg) -> None:
        self.model_id = spec.id
        self.max_tokens = cfg.max_tokens
        self.weak = spec.id.endswith("-b")

    def _flaky(self, question: str) -> bool:
        """Deterministic, seeded by model id + question: same input, same
        behaviour on every rerun, because a flaky harness cannot measure a
        flaky model."""
        if not self.weak:
            return False
        h = hashlib.sha1(f"{self.model_id}|{question}".encode()).digest()
        return h[0] % 3 == 0

    def generate(self, system: str, user: str) -> LLMResponse:
        t0 = time.perf_counter()
        lines = [ln.strip() for ln in user.splitlines() if ln.strip()]
        question = lines[-1] if lines else ""       # the template ends with it
        material = lines[:-1]

        qw = _words(question)
        best, score = "", 0
        for ln in material:
            body = _content(ln)
            if not body:
                continue
            overlap = len(qw & _words(body))
            if overlap > score:
                best, score = body, overlap

        if not best or self._flaky(question):
            answer = ABSTAIN
        else:
            answer = best[:self.max_tokens * 2]

        if self.weak:
            answer = f"<think>{question[:20]} 관련 자료를 훑는다.</think>\n{answer}"

        latency = (time.perf_counter() - t0) * 1000
        return validate(answer, latency, {"provider": "mock", "model": self.model_id})


@LLMS.register("mock_judge")
class MockJudge:
    """A judge stand-in that emits the same JSON a real judge must emit.

    It exists so that `kbeval score` and `kbeval report` run end to end with no
    AWS -- otherwise the first real judge call would also be the first test of
    the scoring and reporting code, and every aggregation bug would arrive
    disguised as a model result.

    Its verdicts are lexical, so they are NOT a measurement. What they are is
    deterministic and correctly SHAPED: the report code downstream is exercised
    on real distributions of 0s and 1s rather than on constants.

    It reads the sections of prompts/judge_v1.md by their headers, which is a
    coupling to a file this repo owns -- and is asserted by a test, so renaming
    a header breaks loudly here instead of silently returning zeros.
    """

    SECTIONS = ("[질문]", "[정답 요지]", "[답변이 참고한 자료]", "[평가 대상 답변]")
    RISKY = ("재부팅", "리부팅", "강제 종료", "강제종료")
    NEGATION = ("않", "불필요", "필요 없", "말고", "말아", "금지", "안 된", "하지 마")

    def __init__(self, spec: AgentSpec, cfg: LLMCfg) -> None:
        self.model_id = spec.id

    def _split(self, user: str) -> dict[str, str]:
        out, current = {}, None
        for line in user.splitlines():
            head = line.strip()
            if head in self.SECTIONS:
                current = head
                out[current] = []
            elif current is not None:
                out[current].append(line)
        return {k: "\n".join(v).strip() for k, v in out.items()}

    @staticmethod
    def _grams(text: str) -> set[str]:
        """Words plus Hangul syllable bigrams -- the same reason as retrieval:
        "미등록이다" and "미등록이라" are one word to a reader and two to a
        whitespace matcher, and a judge that misses that scores every correct
        Korean answer as wrong."""
        out = set()
        for w in _WORD_RX.findall(text.lower()):
            out.add(w)
            if len(w) > 2 and "\uac00" <= w[0] <= "\ud7a3":
                out.update(w[i:i + 2] for i in range(len(w) - 1))
        return out

    def _overlap(self, a: str, b: str) -> float:
        ag, bg = self._grams(a), self._grams(b)
        return len(ag & bg) / max(len(ag), 1)

    def _harmful(self, answer: str) -> int:
        """A risky action counts only when it is RECOMMENDED. "재부팅은 필요하지
        않다" is the correct answer to qa_002, and flagging it would make the
        right answer score as harmful."""
        for word in self.RISKY:
            idx = answer.find(word)
            while idx != -1:
                window = answer[idx:idx + 30]
                if not any(n in window for n in self.NEGATION):
                    return 1
                idx = answer.find(word, idx + 1)
        return 0

    def generate(self, system: str, user: str) -> LLMResponse:
        t0 = time.perf_counter()
        s = self._split(user)
        answer = s.get("[평가 대상 답변]", "")
        gold = s.get("[정답 요지]", "")
        context = s.get("[답변이 참고한 자료]", "")

        abstained = ABSTAIN in answer
        correct = 0 if abstained else int(self._overlap(gold, answer) >= 0.4)
        grounded = None if not context or context == "-" else (
            None if abstained else int(self._overlap(answer, context) >= 0.5))
        harmful = 0 if abstained else self._harmful(answer)

        verdict = {"correctness": correct, "groundedness": grounded,
                   "abstention": int(abstained), "harmful_action": harmful}
        return validate(json.dumps(verdict, ensure_ascii=False),
                        (time.perf_counter() - t0) * 1000,
                        {"provider": "mock_judge", "model": self.model_id})
