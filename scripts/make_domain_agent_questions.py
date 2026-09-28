"""Question set aimed at what domain-agent ADDED, not at what the manual says.

First version asked "what does SS-RSRQ mean?". That barely separated the packs:
a 32B model already knows 3GPP definitions, so da_raw answered from its own
knowledge and looked as if it had read the expansion. The expansion's real
contribution is not the definition -- it is the definition sitting NEXT TO the
product screen that shows it.

The obvious fix -- describe the concept, ask which screen shows it -- failed
at RETRIEVAL: "XCAL-Mobile 어느 화면에서…" pulled the generic overview chunks
to the top under every retriever, the gold chunk landed at rank 7-10, and the
budget then dropped it from da_full (2-3 big chunks fit) but not from da_raw
(7 small ones). That would have measured chunk size, not meaning.

So the answerable questions NAME the screen (meta.shape = screen_anchored) and
ask what one of its items means or does. Naming the screen puts the gold
chunk at rank 1-2 in BOTH packs under bm25/dense/hybrid (checked), so
retrieval and budget are equalised and the remaining difference is the one
the expansion makes while the answer is written.

Caveat, stated so nobody reads past it: the expansion is a generic 3GPP
glossary, which a 32B model largely knows already. Expect small gaps on most
of these and visible ones on precision (RSRQ vs SINR, the "Absolute" in
ARFCN, which node an SCG belongs to) -- and on the harm set.

Two more shapes keep the set honest:
    harm     the expansion is wrong/imprecise here (MAC address -> Medium Access
             Control; SCG Pcell -> MCG PCell). Enrichment should lose.
    control  the gold chunk has no expansion. The packs should tie; a gap is
             budget / chunk size, not meaning.

Gold is the RAW span where the answer's screen/term appears. Both packs share
chunk boundaries, so coverage is 1.0 by construction (rule 1 keeps offsets on
the original text; the expansion itself has no coordinates).

oracle_ref is written independently of the expansion (rule 5) -- it has to be,
because the harm questions exist precisely where the expansion is wrong.
gold_answer / oracle_ref need a domain review before the numbers are cited.

    uv run python scripts/make_domain_agent_questions.py
"""
from __future__ import annotations

import json
from pathlib import Path

from kbeval.config import SourceCfg
from kbeval.sources import load_documents

SRC = SourceCfg(adapter="domain_agent", path="data/domain-agent-export", params={})
DOCS = {d.doc_id: d for d in load_documents(SRC)}
DOC = "XCAL-Mobile_User_Guide_v5_xx_rev34"
OUT = Path("data/eval/da_expansion_questions.jsonl")


def span(start_anchor: str, end_anchor: str, doc_id: str = DOC) -> dict:
    text = DOCS[doc_id].text
    s = text.index(start_anchor)                      # raises if the doc changed
    e = text.index(end_anchor, s) + len(end_anchor)
    return {"doc_id": doc_id, "char_start": s, "char_end": e}


def q(qid: str, query: str, term: str, gold: dict, gold_answer: str,
      oracle_ref: str, **meta) -> dict:
    return dict(qid=qid, intent="qa", answerable=True, query=query, gold=gold,
                gold_answer=gold_answer, oracle_ref=oracle_ref,
                meta={"target": "expansion", "term": term, **meta})


def unanswerable(qid: str, query: str, why: str) -> dict:
    return dict(qid=qid, intent="qa", answerable=False, query=query,
                gold=None, gold_answer=None, oracle_ref=None,
                meta={"target": "absent", "why": why})


NR_CA = span("5G NR CA shows", "SSB Index.")
NR_BEAM = span("The 5G NR Cell screen shows Set", "SS-SINR.")
SCG_SUMMARY = span("The 5G NR Summary(SCG) screen", "activate the menu.")
OVERVIEW = span("XCAL-Mobile is a handheld", "5GNR technologies.")
LICENSE = span("Request a license by sending", "is unavailable.")

# Three shapes, tagged in meta.shape so the report can be read per shape:
#
#   concept_to_screen  the question DESCRIBES a concept in Korean and asks which
#                      XCAL-Mobile screen/item shows it. No English term in the
#                      query, so BM25 cannot shortcut it, and the model's own
#                      3GPP knowledge cannot name a product screen. The link
#                      concept -> screen exists only in the expansion text.
#                      This is where enrichment should be visible.
#   harm               the expansion is wrong or imprecise for this context.
#                      Enrichment should LOSE here; if it does not, the model
#                      is ignoring the context.
#   control            the gold chunk carries no expansion. da_full and da_raw
#                      should tie; a gap here is budget/chunk-size, not meaning.

def anchored(qid, query, term, gold, gold_answer, oracle_ref):
    return q(qid, query, term, gold, gold_answer, oracle_ref, shape="screen_anchored")


QUESTIONS = [
    anchored("da_s01", "5G NR CA 화면에서 간섭 대비 신호 품질을 나타내는 항목은 무엇이고, 어떻게 계산된 값인가요?",
             "SS-RSRQ", NR_CA,
             "SS-RSRQ다. SS-RSRP를 같은 대역폭의 전체 수신 전력으로 나눈 값으로, 절대 세기가 아니라 간섭 대비 "
             "품질을 나타낸다.",
             "5G NR CA 화면은 PCI, NR-ARFCN, Raster Frequency, SS-RSRP, SS-RSRQ, SSB Index를 보여준다. 이 중 "
             "간섭 대비 품질은 SS-RSRQ로, SS-RSRP를 같은 대역폭에서 측정한 전체 수신 전력으로 나눈 값이다."),
    anchored("da_s02", "5G NR CA 화면의 항목 중 주파수를 MHz 대신 정수 채널 번호로 나타낸 것은 무엇인가요?",
             "NR-ARFCN", NR_CA,
             "NR-ARFCN(절대 무선 주파수 채널 번호)이다. 3GPP 래스터로 중심 주파수에 대응된다.",
             "5G NR CA 화면의 NR-ARFCN(NR Absolute Radio-Frequency Channel Number)은 캐리어 주파수를 가리키는 "
             "정수로, 3GPP가 정한 고정 래스터로 중심 주파수에 대응된다."),
    anchored("da_s03", "5G NR Summary(SCG) 화면의 SCG는 이중 연결에서 어느 쪽 노드가 제공하는 셀 묶음인가요?",
             "SCG", SCG_SUMMARY,
             "보조 노드(secondary node)가 제공하는 서빙 셀 묶음(Secondary Cell Group)이다.",
             "SCG(Secondary Cell Group)는 이중 연결에서 보조 노드가 제공하는 서빙 셀 집합이다. 마스터 노드가 "
             "제공하는 쪽은 MCG다. 5G NR Summary(SCG) 화면은 왼쪽 위 SCG 버튼으로 활성화한다."),
    anchored("da_s04", "5G NR CA(SCG) 화면의 CA는 어떤 방식으로 속도를 높이는 기술인가요?",
             "CA", span("On 5G NR CA(SCG)", "is displayed."),
             "여러 컴포넌트 캐리어를 묶어 합친 대역폭으로 송수신해 단일 캐리어보다 최대 처리량을 높인다.",
             "CA(Carrier Aggregation)는 여러 컴포넌트 캐리어를 묶어 한 단말이 합산된 대역폭으로 송수신하게 하는 "
             "기술로, 단일 캐리어보다 최대 처리량을 높인다."),
    anchored("da_s05", "5G NR Beam 화면의 항목 중 셀 선택과 핸드오버 판단에 쓰는 빔 단위 세기 지표는 무엇이고 단위는 뭔가요?",
             "SS-RSRP", NR_BEAM,
             "SS-RSRP이고 단위는 dBm이다.",
             "5G NR Beam 항목의 5G NR Cell 화면은 Set, PCI, SSB Idx, SS-RSRP, SS-RSRQ, SS-SINR을 보여준다. 셀 "
             "선택과 이동성 판단에 쓰는 셀·빔 단위 세기 지표는 SS-RSRP이며 단위는 dBm이다."),
    anchored("da_s06", "5G NR Beam 화면의 SS-RSRP, SS-SINR에서 SS는 단말이 무엇을 할 때 쓰는 신호인가요?",
             "SS", NR_BEAM,
             "동기 신호(Synchronization Signal)다. 단말이 셀을 찾고 그 셀의 타이밍·주파수에 맞출 때 쓴다.",
             "SS는 동기 신호(Synchronization Signal)로, 단말이 브로드캐스트 정보를 읽기 전에 셀을 탐색하고 "
             "타이밍과 주파수를 맞추기 위해 찾는 신호다. 5G NR에서는 SSB로 전송된다."),
    anchored("da_s07", "5G NR RACH 화면은 단말이 어떤 상황에서 셀에 접속하는 과정을 보여주나요?",
             "RACH", span("5G NR RACH", "for 5G NR in the table."),
             "할당된 상향 자원이 없는 단말이 셀에 처음 접속(랜덤 액세스)하는 과정이다. 그 주요 KPI를 표로 보여준다.",
             "RACH(Random Access Channel)는 할당받은 상향 자원이 없는 단말이 셀에 처음 접속할 때 쓰는 상향링크 "
             "채널이다. 5G NR RACH 화면은 그 과정의 주요 KPI(RACH Type, RACH Result 등)를 표로 보여준다."),
    anchored("da_s08", "5G NR Inter RAT 화면에서 말하는 두 무선 기술은 각각 몇 세대 기술인가요?",
             "Inter RAT", span("5G NR Inter RAT\nDisplay Cell Search", "5G NR / LTE."),
             "5G NR은 5세대, LTE는 4세대 3GPP 무선 접속 기술이다.",
             "5G NR Inter RAT 화면은 5G NR / LTE에 걸친 Cell Search 정보를 보여준다. 5G NR은 3GPP 5세대, LTE는 "
             "4세대 무선 접속 기술이다."),
    anchored("da_s09", "RF Measurement Summary 화면에 나오는 5G NR과 LTE는 각각 어떤 특징의 기술인가요?",
             "LTE/5G NR", span("RF Measurement Summary", "key KPIs."),
             "5G NR은 유연한 부반송파 간격·넓은 대역폭·빔 기반 동작을 지원하는 5세대 기술이고, LTE는 하향 OFDMA·"
             "상향 SC-FDMA와 all-IP 코어를 쓰는 4세대 기술이다.",
             "Summary Viewer 화면은 5G NR과 LTE의 핵심 KPI를 보여준다. 5G NR은 유연한 부반송파 간격, 넓은 대역폭, "
             "빔 기반 동작을 지원하는 3GPP 5세대 기술이고, LTE는 하향 OFDMA, 상향 SC-FDMA, all-IP 코어의 4세대 "
             "기술이다."),
    anchored("da_s10", "RF Information 메뉴에서 말하는 RF 정보는 구체적으로 어떤 측정값을 뜻하나요?",
             "RF", span("RF Information (Qualcomm Chipset)", "on smartphones."),
             "무선 주파수 영역, 즉 아날로그 신호 경로와 그 위에서 잰 전력·주파수 같은 측정값이다.",
             "RF(Radio Frequency) 정보는 아날로그 신호 경로와 그 위에서 측정한 전력·주파수 같은 값을 가리킨다. "
             "XCAL-Mobile은 Qualcomm 칩셋 스마트폰에서 실시간 RF 정보를 보여준다."),
    anchored("da_s11", "XCAL-Mobile 개요에 나오는 GSM은 몇 세대 기술이고 측정 도구에서 어떤 의미인가요?",
             "GSM", OVERVIEW,
             "2세대 디지털 셀룰러 표준이다. 측정 도구에서는 단말이 폴백하는 레거시 무선 접속 기술로 나타난다.",
             "GSM은 2세대(2G) 디지털 셀룰러 표준이다. 측정 도구에서는 단말이 상위 기술을 쓸 수 없을 때 폴백하는 "
             "레거시 무선 접속 기술로 나타난다. XCAL-Mobile은 WCDMA, GSM, LTE, 5GNR의 QoS·QoE 테스트를 지원한다."),

    q("da_s12", "5G NR Summary(SCG) 화면의 'SCG Pcell'은 단말이 처음 접속한 셀인가요?",
      "Pcell", SCG_SUMMARY,
      "아니다. SCG 쪽 주 셀(PSCell)이다. 단말의 최초 접속을 담당하는 셀은 MCG의 PCell이다.",
      "이중 연결에서 SCG의 주 셀은 PSCell(Primary SCG Cell)이다. 보조 노드 쪽 셀 그룹을 대표하며, "
      "단말의 최초 접속을 담당하는 MCG의 PCell과 구별된다.",
      shape="harm", note="확장본은 Pcell을 MCG의 PCell(최초 접속 셀)로 풀었다"),
    q("da_s13", "라이선스 요청할 때 보내는 MAC 주소는 어디서 확인하나요? 인터넷이 끊겨 있어도 보이나요?",
      "MAC", LICENSE,
      "단말 네트워크 인터페이스의 하드웨어 주소다. 인터넷(WiFi 또는 3G/4G/5G)에 연결되어 있지 않으면 "
      "표시되지 않는다.",
      "라이선스 요청에 쓰는 MAC 주소는 단말 네트워크 인터페이스(주로 Wi-Fi)의 하드웨어 주소다. "
      "XCAL-Mobile에서는 인터넷(WiFi 또는 3G/4G/5G)에 연결되어 있지 않으면 MAC 주소가 표시되지 않는다.",
      shape="harm", note="확장본은 MAC을 L2 Medium Access Control로 풀었다. 여기서는 하드웨어 주소다"),
    q("da_s14", "XCAL-Mobile에서 MAC이라는 말이 라이선스 쪽과 처리량 화면 쪽에 둘 다 나오는데 같은 뜻인가요?",
      "MAC", LICENSE,
      "다르다. 라이선스 요청의 MAC 주소는 단말의 하드웨어 주소이고, 처리량 화면의 DL/UL MAC TP는 "
      "L2 MAC(Medium Access Control) 계층의 처리량이다.",
      "라이선스 요청에 쓰는 MAC 주소는 단말 네트워크 인터페이스의 하드웨어 주소다. 반면 5G NR Dual "
      "Connectivity 화면의 DL MAC TP / UL MAC TP는 L2 MAC(Medium Access Control) 계층에서 잰 처리량이다. "
      "이름만 같고 가리키는 대상이 다르다.",
      shape="harm", note="확장은 라이선스 청크에만 있고 그 풀이는 처리량 쪽 뜻이다"),

    q("da_s15", "Dual Connectivity 화면에서 'LTE : Pcell + Scell'은 무슨 뜻인가요?",
      "Pcell", span("Current Throughput\nTotal: LTE+5G", "5G NR: Pcell + Scell"),
      "CA 구성에서 LTE 주 셀(PCell)과 보조 셀(SCell)의 처리량을 합친 값이다.",
      "5G NR Dual Connectivity 화면의 Current Throughput에서 'LTE : Pcell + Scell'은 캐리어 집성 구성의 "
      "LTE 주 셀(PCell)과 보조 셀(SCell) 처리량을 합친 값이다. Total은 LTE+5G 합계다.",
      shape="control", note="이 청크에는 확장이 없다. 두 pack이 비슷해야 정상"),
    q("da_s16", "5G NR Dual Connectivity 화면의 DL MAC TP와 DL PHY TP는 어떻게 다른가요?",
      "MAC", span("DL PHY TP\nUL PHY TP\nDL MAC TP", "UL PDCP TP"),
      "DL PHY TP는 물리계층에서 잰 하향 처리량이고, DL MAC TP는 그 위 L2 MAC 계층에서 잰 하향 처리량이다.",
      "5G NR Dual Connectivity 화면은 PHY, MAC, RLC, PDCP 계층별 처리량(DL/UL)을 보여준다. PHY TP는 "
      "물리계층, MAC TP는 L2 MAC 계층에서 잰 처리량이다.",
      shape="control", note="이 청크엔 확장이 없다. 다른 청크(라이선스)의 MAC 풀이가 검색되면 da_full이 유리"),

    # 5 / 21 = 24% -- rule 4. Product-specific facts that neither the manual
    # nor the expansion contains; a model that answers them is inventing.
    unanswerable("da_x101", "5G NR Beam 화면에서 SS-SINR 경고 임계값은 기본 몇 dB로 설정되어 있나요?",
                 "임계값 설정이 매뉴얼에 없다"),
    unanswerable("da_x102", "XCAL-Mobile에서 CRI-RSRP 측정 주기를 바꾸려면 어떻게 하나요?",
                 "측정 주기 설정이 매뉴얼에 없다"),
    unanswerable("da_x103", "XCAL-Mobile이 계산하는 QoE 점수는 어떤 공식으로 산출되나요?",
                 "QoE 산출식이 없다(확장본도 'QoE 미정의'로 관찰)"),
    unanswerable("da_x104", "SSB Idx가 바뀔 때 XCAL-Mobile 로그에 남는 이벤트 이름은 무엇인가요?",
                 "로그 이벤트 명세가 없다"),
    unanswerable("da_x105", "XCAL-Mobile에서 SCell이 추가·해제될 때 PCell 변경 이벤트도 따로 기록되나요?",
                 "'SCell Event'는 메뉴 이름만 있고 동작 설명이 없다"),
]

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for row in QUESTIONS:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

n_unans = sum(1 for row in QUESTIONS if not row["answerable"])
print(f"wrote {len(QUESTIONS)} questions to {OUT} "
      f"({n_unans} unanswerable = {n_unans / len(QUESTIONS):.0%})")
for row in QUESTIONS:
    g = row["gold"]
    loc = f"[{g['char_start']}:{g['char_end']}]" if g else "-"
    print(f"  {row['qid']:8s} {loc:14s} {row['meta'].get('shape', '-'):17s} {row['query'][:44]}")
