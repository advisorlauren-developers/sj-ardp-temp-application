"""Build the question set with DERIVED gold offsets.

Never hand-type char_start / char_end. The moment anyone edits a document --
fixes a typo, adds a sentence -- every hardcoded number silently shifts and the
whole gold set becomes wrong without a single test failing.

Instead, anchor on text that identifies the answer span, and let the script
find the offsets. Re-run it after any document change.

    uv run python scripts/make_questions.py
"""
from __future__ import annotations

import json
from pathlib import Path

from kbeval.config import SourceCfg
from kbeval.sources import load_documents

DOCS = {d.doc_id: d for d in load_documents(SourceCfg())}
OUT = Path("data/eval/questions.jsonl")


def span(doc_id: str, start_anchor: str, end_anchor: str) -> dict:
    text = DOCS[doc_id].text
    s = text.index(start_anchor)                      # raises if the doc changed
    e = text.index(end_anchor, s) + len(end_anchor)
    return {"doc_id": doc_id, "char_start": s, "char_end": e}


QUESTIONS = [
    dict(
        qid="qa_001", intent="qa", answerable=True,
        query="OLT에서 E411 인증 실패가 뜨는데 원인이 뭐고 어떻게 조치해야 하나요?",
        gold=span("olt_auth.md", "E-411은 ONT 시리얼", "인증이 완료된다."),
        gold_answer="ONT 시리얼이 OLT 가입자 DB에 미등록된 경우 발생한다. "
                    "show ont autofind로 시리얼을 확인하고 ont add로 등록하면 "
                    "최대 90초 내 인증이 완료된다.",
        oracle_ref="E-411은 ONT 시리얼 번호가 OLT 가입자 DB에 등록되지 않았을 때 발생한다. "
                   "펌웨어 3.2 이상은 자동 등록이 비활성화되어 수동 등록이 필요하다. "
                   "show ont autofind로 미등록 시리얼을 확인하고 ont add로 가입자 "
                   "프로파일에 등록하면 최대 90초 이내 인증이 완료된다. 장비 재부팅은 "
                   "불필요하며, 재부팅 시 동일 PON 포트의 정상 가입자 전체가 단절된다.",
    ),
    dict(
        qid="qa_002", intent="qa", answerable=True,
        query="E-411 조치할 때 장비 리부팅 해도 되나요?",
        gold=span("olt_auth.md", "장비 재부팅은 필요하지", "수행하지 않는다."),
        gold_answer="재부팅은 필요하지 않다. 재부팅하면 동일 PON 포트의 정상 가입자 "
                    "전체가 단절되므로 수행하면 안 된다.",
        oracle_ref="E-411 조치 시 장비 재부팅은 필요하지 않다. 재부팅을 수행하면 동일 "
                   "PON 포트에 물린 정상 가입자 전체가 단절되므로 수행해서는 안 된다.",
    ),
    dict(
        qid="qa_003", intent="qa", answerable=True,
        query="광 수신 레벨이 얼마 이하로 떨어지면 E-412가 자주 발생하나요?",
        gold=span("olt_auth.md", "E-412는 ONT가 응답", "빈번히 발생한다."),
        gold_answer="-27dBm 이하일 때 빈번히 발생한다.",
        oracle_ref="E-412는 ONT가 응답하지 않아 인증이 완료되지 않은 경우이며, 주된 원인은 "
                   "광 수신 레벨 저하다. 수신 레벨이 -27dBm 이하일 때 빈번히 발생한다.",
    ),
    dict(
        qid="qa_004", intent="qa", answerable=True,
        query="BNG에서 가입자 속도 프로파일을 바꾸려면 세션을 끊어야 하나요?",
        gold=span("bng_qos.md", "가입자 속도 프로파일은", "일치 여부를 확인한다."),
        gold_answer="세션 재연결 없이 CoA로 변경 가능하다. CoA 실패 시에도 세션을 강제 "
                    "종료하지 말고 RADIUS accounting 로그에서 세션 ID 일치 여부를 먼저 확인한다.",
        oracle_ref="가입자 속도 프로파일은 RADIUS 속성으로 전달되며 세션 재연결 없이 CoA로 "
                   "변경할 수 있다. CoA 적용이 실패하더라도 세션을 강제 종료하지 말고, 먼저 "
                   "RADIUS 서버의 accounting 로그에서 세션 ID 일치 여부를 확인한다.",
    ),
    dict(
        qid="qa_005", intent="qa", answerable=True,
        query="계약 대역 초과 트래픽을 셰이핑으로 처리하려면 뭐가 필요한가요?",
        gold=span("bng_qos.md", "계약 대역을 초과하는", "라이선스가 필요하다."),
        gold_answer="기본은 폴리싱 폐기이며, 셰이핑으로 변경하려면 별도 라이선스가 필요하다.",
        oracle_ref="계약 대역을 초과하는 트래픽은 기본적으로 폴리싱되어 폐기된다. "
                   "셰이핑으로 동작 방식을 변경하려면 별도 라이선스가 필요하다.",
    ),
    # ---- unanswerable -------------------------------------------------------
    # Without these, injecting knowledge always wins and the experiment is
    # biased by construction: there is no way for a pack to be penalised for
    # encouraging confident invention. Target 20-25% of the set.
    dict(qid="qa_101", intent="qa", answerable=False,
         query="E-415 오류의 원인과 조치 방법을 알려주세요.",
         gold=None, gold_answer=None, oracle_ref=None),
    dict(qid="qa_102", intent="qa", answerable=False,
         query="OLT 펌웨어 4.0에서 바뀐 인증 동작이 어떻게 되나요?",
         gold=None, gold_answer=None, oracle_ref=None),
]

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for q in QUESTIONS:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

n_unans = sum(1 for q in QUESTIONS if not q["answerable"])
print(f"wrote {len(QUESTIONS)} questions to {OUT} "
      f"({n_unans} unanswerable = {n_unans / len(QUESTIONS):.0%})")
for q in QUESTIONS:
    g = q["gold"]
    loc = f"{g['doc_id']}[{g['char_start']}:{g['char_end']}]" if g else "-"
    print(f"  {q['qid']}  {loc}")