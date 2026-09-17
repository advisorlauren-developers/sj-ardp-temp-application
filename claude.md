# kbeval — AI-Ready Dataset Downstream Evaluation Harness

## 이 프로젝트가 무엇인가

**측정 장치다. 챗봇 애플리케이션이 아니다.**

사내 문서(추후 JIRA)를 RAG 파이프라인에 넣었을 때, **데이터 전처리 수준(ai-ready 정도)이
downstream 성능을 얼마나 바꾸는가**를 통제된 조건에서 측정한다.
기술지원 Q&A 형태를 띠지만 목적은 답변 품질이 아니라 **데이터 품질의 정량화**다.

따라서:
- UI/UX 품질은 목표가 아니다. 데모는 읽기 전용이며 측정에 관여하지 않는다.
- 기능 추가보다 **통제변수 보존**이 우선이다.
- 예쁜 출력보다 **재현 가능한 숫자**가 우선이다.
- 애매하면 기능을 빼라. PoC다.

지식 주입 방식은 **in-context RAG**로 고정한다. fine-tuning은 데이터 품질과 학습
하이퍼파라미터가 교락되어 측정이 불가능해진다.

---

## 실험 설계 (코드를 바꾸기 전에 반드시 이해할 것)

### 세 축

| 축 | 의미 | 담당 디렉터리 |
|---|---|---|
| `agent` | 기반 모델 (Qwen3-32B, Gemma3-27B, …) | `llm/` |
| `mode` | 지식 주입 방식 (`bare` / `rag` / `oracle_ref`) | `agents/` |
| `pack` | ai-ready 처리 수준 (`raw` / `mid` / `full`) | `sources/` + `packs/` |

세 축은 **서로를 몰라야 한다.** 이건 `tests/test_architecture.py`가 매 실행마다 검증한다.

```bash
grep -r "boto3\|modelId\|bedrock" src/kbeval/agents/    # 0건
grep -r "Retriever\|retriev" src/kbeval/packs/          # docstring 산문만 나온다
```

raw grep은 거칠다 — docstring이 "왜 이렇게 했는가"를 설명하며 다른 축을 언급하는 건
정상이고 오히려 필요하다. 테스트는 **docstring과 주석을 제거한 뒤** 검사하므로
금지되는 것은 실제 import·호출뿐이다.

### 측정량

절대 점수는 "모델이 원래 잘함"과 "데이터가 좋음"이 섞여 있어 쓸 수 없다.

```
headroom     = oracle_ref − bare              지식으로 얻을 수 있는 총량
lift         = rag − bare                     이 pack이 실제로 얻은 양
realization  = lift / headroom                실현율 ← 1차 관측량
kendall τ    = 모델 간 pack 순위 일치도        ← 온프렘 전이 가능성
```

- `headroom < 0.05`인 모델은 분모가 불안정하므로 해석에서 제외한다.
- τ ≥ 0.7이면 pack 순위가 모델 무관 → Bedrock 대리 실험 결론을 온프렘으로 이전 가능.
- 통계는 **문항 단위 paired**(Wilcoxon). 모델 간 분산이 pack 효과보다 크므로 unpaired는 무의미.

### 배포 맥락

최종 목표는 **온프렘 30B급 서빙**이다. Bedrock은 대리 환경이다.
따라서 `context_token_budget`은 Bedrock의 능력이 아니라 **온프렘 현실(8K 컨텍스트, 양자화)**
기준으로 잡는다. 현재 3000 토큰.

---

## 절대 규칙 (위반하면 실험이 무효가 된다)

이 10가지는 편의나 리팩터링을 이유로 바꾸지 않는다. 바꿔야 한다면 먼저 사용자에게 묻는다.

1. **offset 보존.** `Chunk.char_start/char_end`는 `Document.text` 좌표를 가리킨다.
   chunker 불변식: `doc.text[start:end].strip() == chunk.text`.
   transform은 텍스트를 고쳐도 좌표는 건드리지 않는다(그래서 transform 후엔 이 불변식이
   적용되지 않는다 — chunker에만 적용).
   → gold를 한 번만 라벨링하고 모든 pack에 자동 매핑하기 위한 유일한 장치. 나중에 못 붙인다.

2. **통제변수는 토큰 예산이지 `top_k`가 아니다.** pack마다 청크 길이가 다르면 프롬프트 길이가
   달라지고, 30B급의 long-context 저하가 데이터 품질 효과로 둔갑한다.
   예산에 안 들어가는 청크는 **자르지 말고 건너뛴다**(자르면 측정 대상이 훼손됨).

3. **config는 `extra="forbid"`.** 오타는 에러여야 한다. 조용히 무시되면 안 된다.
   config 해시가 모든 trace 행에 박힌다. 통제변수를 바꾸면 해시가 바뀌어야 한다.

4. **unanswerable 문항 20~25% 유지.** 없으면 지식 주입이 항상 이기는 편향된 실험이 된다.
   이 문항들은 `realization_rate` 계산에서 제외하고 abstention으로만 평가한다.

5. **`oracle_ref`는 pack 독립.** 정제된 참조 문단을 주입하며 모든 pack에서 동일하다.
   pack별 gold를 분모로 쓰면 나쁜 pack이 자기 천장도 낮춰서 실현율이 부당하게 좋아진다.

6. **`oracle_ref`와 `rag`는 같은 system prompt·같은 context template을 쓴다.**
   컨텍스트 내용만 다르다. 이게 어긋나면 realization_rate가 조용히 무효가 된다.

7. **스윕은 교차곱이 아니다.** `bare`와 `oracle_ref`는 pack 독립이므로
   모델 N × pack M → `N × (2 + M)` specs. 예산이 1/3로 줄어든다.

8. **응답 본문 검증.** Bedrock은 HTTP 200에 에러 문서를 담아 반환할 수 있다.
   빈 응답을 오답으로 집계하면 "pack이 성능을 떨어뜨렸다"로 보인다.
   실패는 반드시 `error` 필드로 기록하고 답변으로 취급하지 않는다.

9. **gold 생존 검증.** `Pack.verify_gold_coverage`가 `coverage=1.0`이어야 한다.
   `dedup` 같은 transform이 정답 청크를 지우면 그건 깨끗해진 코퍼스가 아니라 **삭제된 정답**이고,
   리포트에서는 모델 성능 저하로 보인다. `kbeval pack`이 coverage < 1이면 exit 1.

10. **thinking 블록 제거.** reasoning 토큰이 judge 채점에 섞이면 안 된다.
    태그로 감싸는 모델은 제거하고, 인라인 추론 모델(Nemotron 등)은 분리 불가하므로
    `thinking` 플래그로 표시하고 별도 층으로 보고한다.

---

## 코드 구조 규약

### 최상위 = "누가 바꾸는가"

```
src/kbeval/   개발자. 실험마다 바뀌지 않음
configs/      실험자. 실험마다 바뀜
prompts/      실험자. 버전 고정 대상 (코드에 문자열로 박지 말 것 — 해시에 안 잡힌다)
data/         도메인 전문가
runs/         프로그램만 append. 손대지 않음
tests/        단계별 보증
```

### 폴더 = 교체 가능한 것, 파일 = 하나뿐인 것

```
llm/ sources/ packs/ retrieval/ agents/ eval/scorers/   → 구현이 여럿이므로 폴더
config.py budget.py sweep.py types.py registry.py cli.py → 하나뿐이므로 파일
```

### 모든 폴더의 내부 규약

```
<pkg>/
├── base.py        Protocol 정의 + 팩토리 함수
├── impl_a.py      구현 1개 (@REGISTRY.register("name") 데코레이터)
├── impl_b.py
└── __init__.py    구현 import (등록 부작용) + export
```

**새 구현 추가 = 파일 하나 + 데코레이터 한 줄 + `__init__.py` 한 줄.**
다른 곳은 건드리지 않는다. `__init__.py` 갱신을 빠뜨리면 `ImportError`가 난다 — 자주 하는 실수다.

### 단계마다 Protocol 하나

기능을 추가할 때는 Protocol을 하나만 도입하고, 그 단계가 끝나면 **실행 가능한 확인 명령**이
하나 늘어야 한다. "전부 만든 뒤 처음 실행"은 실패 지점을 못 찾는다.

---

## 현재 상태

| 단계 | 내용 | 확인 명령 | 상태 |
|---|---|---|---|
| S0 | uv 뼈대, 진입점 | `kbeval --help` | 완료 |
| S1 | types / registry / config / sweep | `kbeval plan` | 완료 |
| S2 | `SourceAdapter` | `kbeval docs` | 완료 |
| S3 | `Chunker` (offset 보존) | `kbeval chunks <pack>` | 완료 |
| S4 | `Transform` + pack 영속화 + gold 검증 | `kbeval pack <name>` | 완료 |
| S5 | `Retriever` (BM25, 한글 bigram) | `kbeval search` / `kbeval retrieval-eval` | 완료 |
| S6 | `LLMClient` + `Agent` + 토큰 예산 | `kbeval ask` | 완료 |
| S7 | Runner (재개 가능) | `kbeval run` | 완료 |
| S8 | `Scorer` + Report | `kbeval score` / `kbeval report` | 완료 |
| S9 | Bedrock 실측 | | **다음** |
| S10 | 읽기 전용 FastAPI 데모 | `uvicorn kbeval.web:app` | (측정 후) |

### S1–S8 기대 출력 (회귀 확인용)

```
$ kbeval plan
specs : 10   (cross product would be 18)

$ kbeval docs
docs=2   bng_qos.md chars=292   olt_auth.md chars=570

$ kbeval chunks mid
chunks=6  avg_chars=142  offset invariant: OK

$ kbeval pack raw|mid|full
chunks=2|6|6   coverage=1.0 (세 pack 모두)

$ kbeval retrieval-eval
questions=5  depth=10
raw  chunks=2  hit@1=1.000  mrr=1.000
mid  chunks=6  hit@1=0.800  mrr=0.900
full chunks=6  hit@1=0.800  mrr=0.900

$ kbeval ask qa_001 --mode bare        -> answer : 답변 불가
$ kbeval ask qa_001 --mode rag --pack mid
injected: 4 chunks / 533 tok of 3000   skipped=0
$ kbeval ask qa_001 --mode oracle_ref  -> injected: 0 chunks / 107 tok

$ kbeval run
specs   : 10   planned=70  resumed=0  written=70   errors : 0
$ kbeval run            # 그대로 다시 → resumed=70  written=0

$ kbeval score
rows    : 70   scored=70  resumed=0   skipped: 0   format: 0

$ kbeval report
mock_a  raw   bare=0.000 rag=0.200 oracle=1.000 headroom=1.000 realiz=0.200
(mock 기준. 숫자는 배관이 도는 증거일 뿐 데이터 품질의 증거가 아니다)

$ pytest -q
329 passed
```

### 알려진 한계 — S9 실측 전에 데이터를 키울 것

현재 코퍼스는 문서 2개 862자다. **이 크기에서는 세 축이 측정되지 않는다.**

- `raw`가 문서 전체를 한 청크로 담아서(900자 > 문서 길이) 검색이 항상 맞는다.
  hit@1=1.0은 raw가 좋아서가 아니라 후보가 2개뿐이라서다.
- `mid`와 `full`의 청크가 **바이트 단위로 동일하다.** 문서에 중복이 없어 `dedup`이,
  표기가 이미 일관돼서 `normalize_terms`가 각각 아무 일도 하지 않는다.
  현재 스윕에서 `full`은 `mid`의 복제본이고 두 pack을 비교해도 0이 나온다.
- 전체 코퍼스가 ~431 토큰이라 `context_token_budget=3000`이 한 번도 걸리지 않는다.
  통제변수가 작동하지 않는 상태다.

배관 검증이 목적인 샘플 데이터셋이므로 S8까지는 이대로 간다.
**실측(S9) 직전에** 문서를 수십 건으로 키운다. 그때까지 나오는 숫자는
파이프라인이 도는지의 증거이지 데이터 품질의 증거가 아니다.

---

## 남은 단계 지침

### S5 — `Retriever` (완료)

**LLM을 붙이기 전에 검색을 먼저 확정한다.** retrieval이 깨진 채로 생성을 붙이면 원인 분리가
불가능하고 토큰만 태운다. 이 단계는 **토큰 0원**으로 hit@k를 측정한다.

- `retrieval/base.py`: `Retriever` Protocol (`index(chunks)`, `search(query, k)`), 한글 토크나이저
- `retrieval/bm25.py`: 순수 파이썬 BM25. **한글은 공백 토큰만으로는 recall이 참혹하다**
  (조사·어미가 붙음). 음절 bigram을 함께 넣는다. 형태소 분석기는 나중에, 단 **pack 간 고정**으로.
- `retrieval/dense.py`, `hybrid.py`(RRF): 나중에. PoC 기본값은 bm25.
- `kbeval search "질문"` → 랭킹 + gold 적중 표시
- `kbeval retrieval-eval` → pack별 hit@1/3/5/10, MRR
- retriever 알고리즘·임베더는 **pack 간 고정**. 인덱스만 pack 종속(그게 측정 대상).
  → `retrieval:` 블록은 최상위다. `PackCfg`에 넣지 않는다.
- **토크나이저는 term normalization을 하지 않는다.** 여기서 `E411`→`E-411`을 접으면
  모든 pack에 동시에 적용되어 `normalize_terms` 인자가 실험에서 조용히 사라진다.
  정규화는 pack transform의 일이고, 거기여야 측정된다.
  (그래서 `E-411`은 토큰 하나로 유지한다 — `e`+`411`로 쪼개지면 우연히 매칭된다.)
- 2음절 한글 단어는 bigram == 단어 자신이므로 bigram을 추가하지 않는다(tf 이중 계산).

### S6 — `LLMClient` + `Agent` (완료)

- `llm/base.py`: Protocol + `build_llm()`. `llm/mock.py`가 1급 시민 —
  AWS 없이 전 파이프라인이 돈다.
- `llm/bedrock.py`: runtime(Converse) / mantle(OpenAI 호환) 이원화. 본문 검증. 백오프.
- `agents/`: `bare`, `rag`, `oracle_ref`. (`oracle_pack` 생략.)
- `budget.py`: 토큰 예산 채우기. **자르지 않고 건너뛴다. 건너뛴 뒤에도 계속 내려간다**
  (한 개의 비대한 청크가 그 아래 전부를 지우면 청크 크기가 고른 pack이 이중으로 유리해진다).
  `skipped_chunk_ids`를 trace에 남긴다 — 예산이 실제로 걸렸는지가 이걸로만 보인다.
- `wiring.py`: 세 축을 조립하는 **유일한** 자리. CLI와 runner가 같은 경로를 쓴다
  (각자 조립하면 눈으로 본 답변과 기록된 trace가 갈라진다).

#### S6에서 확정한 것

- **프롬프트 내용이 config 해시에 들어간다.** 경로만 해시하면 `system_qa.md`를 고쳐도
  해시가 그대로여서 서로 다른 실험이 같은 좌표를 갖는다. `Config.canonical()`이
  프롬프트 3개 파일의 내용 digest를 함께 직렬화한다.
- **규칙 6은 클래스 구조로 강제한다.** `agents/base.py:ContextAgent.answer()`가 유일한
  구현이고, mode는 `material()`만 오버라이드한다. system prompt·템플릿·디코딩 설정이
  구조적으로 동일해진다. bare는 컨텍스트 자리에 `prompts.no_material`이 들어간다.
- **후보 답변은 평문이다. JSON을 요구하지 않는다.** 20~30B급의 JSON 미준수가
  지식 효과로 둔갑하는 경로를 아예 없앤다. 파싱 부담은 S8의 judge(강한 모델)만 진다.
  따라서 `format_fail`은 judge 출력 파싱 실패를 센다.
- **기권은 `답변 불가` 문자열 하나로 고정.** 빈 응답은 기권이 아니라 `error`다
  (`llm/base.py:validate`). 둘을 섞으면 장애가 "모델이 신중했다"로 보인다.

#### mock의 한계 (알고 쓸 것)

어휘 매처는 "자료가 이 주제를 다루지 않는다"와 "다른 말로 다룬다"를 구별하지 못한다.
그래서 **unanswerable 문항의 rag 모드에서 mock은 기권하지 않고 distractor로 답한다.**
파이프라인 문제가 아니라 이 mock의 천장이다. abstention은 실모델에서만 측정되며,
채점기 자체는 S8에서 조작된 답변으로 단위 테스트한다.

### S7 — Runner (완료)

- `runs/<run_id>/traces.jsonl` append-only, 행마다 flush.
  버퍼링은 재개의 유일한 전제("디스크에 행이 있으면 그 단위는 끝났다")를
  이 규모에서 아무도 필요 없는 처리량과 맞바꾼다.
- `(spec_key, qid, repeat)` 기준 **재개**.
- `config.resolved.yaml`, `gold_coverage.json` 동시 기록.
- `ThreadPoolExecutor(run.max_concurrency)`. 백오프는 `llm/bedrock.py`에 있다(이중으로 걸지 않는다).

#### S7에서 확정한 것

- **운영 설정은 해시에서 뺀다.** `run.name` / `run.out_dir` / `run.max_concurrency`는
  답변을 한 글자도 바꾸지 않는다. 해시에 넣으면 스레드 수만 올려도 이전 측정 전체와
  비교가 금지된다. `run.seed`는 통제변수이므로 해시에 남는다.
  → `run_id = {name}-{hash}`이므로 이름이 달라도 해시가 같으면 비교 가능하다.
- **찢어진 마지막 줄을 먼저 개행으로 닫는다** (`_heal_torn_line`).
  쓰기 도중 죽으면 개행 없는 행이 남고, 그 뒤에 append하면 두 행이 한 행으로 접합된다.
  찢어진 꼬리는 읽는 사람 눈에 띄지만 **접합된 행은 멀쩡해 보인다.** 1바이트로 막는다.
- **읽기는 손상 행을 세어서 보고한다** (`read_traces -> (rows, bad)`).
  "증거 일부가 소실됐다"와 "증거가 원래 그만큼이었다"는 다른 주장이고 하나만 정직하다.
- **gold 게이트가 스윕 시작 전에 걸린다.** coverage < 1이면 `gold_coverage.json`만 남기고
  `traces.jsonl`은 만들지 않은 채 exit 1. 리포트 시점엔 이미 돈을 쓴 뒤다.
- **worker 예외는 삼키지 않는다.** 프로바이더 장애는 `error` 필드로 집계되지만
  코드 버그(예: TypeError)는 스윕을 세운다. 둘을 섞으면 버그가 "모델 실패율"이 된다.

### S8 — `Scorer` + Report (완료)

- `eval/scorers/retrieval.py`: hit@k, MRR, **gold_injected**, **gold_lost_to_budget**.
  추론 없이 trace에서 계산한다.
- `eval/scorers/judge.py`: LLM-as-judge. 네 항목 **분리 집계**.
- `eval/scoring.py`: `scores.jsonl`에 append. 재개 가능.
- `eval/report.py`: realization / retrieval+공변량 / abstention / kendall tau / health.

#### S8에서 확정한 것

- **점수는 traces.jsonl에 되쓰지 않는다.** 별도 `scores.jsonl`이다.
  trace는 증거이고 증거는 제자리에서 고치지 않는다. 덕분에 **judge 프롬프트를 고쳐
  전체를 재채점해도 candidate 호출은 단 한 번도 다시 사지 않는다.**
- **`hit@k`와 `gold_injected`는 다른 질문이다.** 검색이 찾았는가 / 모델에게 닿았는가.
  예산이 물리는 순간 둘이 갈라진다(gold가 1위인데 위쪽 뚱뚱한 청크에 밀려 탈락).
  한쪽만 보면 예산 효과를 검색 탓으로 돌리게 된다. `gold_lost_to_budget`이 그 경우를 센다.
- **judge 컨텍스트는 저장하지 않고 chunk_id로 복원한다.** pack은 config 해시에 대해
  결정적이므로 정확히 복원된다. trace가 읽을 만한 크기로 유지된다.
  `oracle_ref`와 `bare` 행은 의도적으로 `-`로 복원된다(규칙 5).
- **judge 파싱 실패는 `format_fail`이고 0점이 아니다.** 평균에서 빠진다.
  0으로 세면 "포맷 능력"이 "지식"으로 둔갑한다.
- **provider 에러 행은 채점 자체를 건너뛴다**(`skipped: error`). 분모에서도 빠진다.
  전 스펙 장애 시 realization은 0이 아니라 **None**으로 나와야 한다 — 확인됨.
- **headroom < `scoring.min_headroom`이면 realization을 계산하지 않는다.**
  분모가 0에 가까우면 잡음이 화려한 비율로 둔갑한다. 대시와 사유를 출력한다.
- **음수 realization은 남기고 라벨을 붙인다.** 지우면 모든 pack이 무해해 보인다.
- **Wilcoxon은 문항 단위 paired.** n<5거나 차이가 전부 0이면 p를 내지 않는다.
- **kendall tau는 pack 순위가 전부 동률이면 계산하지 않는다**(정의되지 않음).

### S9 — Bedrock 실측

순서를 지킨다.

1. 모델 ID 확인: `aws bedrock list-foundation-models --region <r>`
   - 30B급 후보: `qwen.qwen3-32b`(dense 32B), `google.gemma-3-27b`(PT/IT 확인 필수),
     `openai.gpt-oss-20b`(MoE active ~3B), `nvidia.nemotron-nano-3-30b`(MoE)
   - **dense와 MoE를 같은 층으로 평균내지 말 것.** 활성 파라미터가 10배 차이난다.
2. `--limit 1`로 모델별 호출만 스모크 테스트 (judge는 mock으로)
3. judge를 frontier 모델로 교체
4. 전체 스윕

Bedrock 함정: runtime/mantle ID 상이, `us.`/`global.` 접두사, HTTP 200+에러본문,
temperature/top_p 동시 지정 거부(top_p None이면 payload에서 제외), throttling.

---

### S10 — 읽기 전용 FastAPI 데모

**측정이 끝난 뒤에 붙인다.** 지금 붙이면 보여줄 숫자가 없다(S8 미구현).

제약:
- **읽기 전용.** `runs/*/traces.jsonl`과 `data/packs/*/manifest.json`을 읽어서 보여주기만 한다.
  웹에서 실험을 실행하지 않는다 — 실행 경로가 둘이 되면 재현 좌표가 갈라진다.
- FastAPI 하나, 템플릿 최소, 빌드 스텝 없음. `src/kbeval/web.py` 파일 하나.
- 라우트는 리포트 축과 1:1: `/` (realization_rate 표), `/retrieval` (pack별 hit@k),
  `/trace/{run_id}/{qid}` (문항 하나의 세 mode 답변 나란히).
- 측정에 관여하지 않으므로 config 해시에 들어가지 않고, 테스트 대상도 아니다.
- **UI를 다듬지 않는다.** 이건 계측기 눈금이지 제품이 아니다.

---

## 명령어

```bash
uv sync
uv run python scripts/make_questions.py    # 문서 수정 시마다 재실행 (gold offset 재유도)

uv run kbeval plan                          # 스윕/해시 확인, 토큰 0
uv run kbeval docs --show 120
uv run kbeval chunks mid --show 6
uv run kbeval pack raw|mid|full

uv run kbeval search "E411 인증 실패" --pack mid      # 랭킹 + gold(*) 표시
uv run kbeval search "" --qid qa_003 --pack full
uv run kbeval retrieval-eval --detail                 # pack별 hit@k / MRR, 토큰 0

uv run kbeval ask qa_001 --mode bare                  # 세 mode 한 문항씩
uv run kbeval ask qa_001 --mode rag --pack mid --show-prompt
uv run kbeval ask qa_001 --mode oracle_ref
uv run kbeval ask qa_001 --mode rag --pack mid -s fixed.context_token_budget=120

uv run kbeval run --limit 1                 # 스펙당 1문항 스모크
uv run kbeval run                           # 전체 스윕 (재개 가능, 그냥 다시 실행)
uv run kbeval score                         # traces -> scores.jsonl (재개 가능)
uv run kbeval report                        # realization / retrieval / abstention / tau

uv run pytest -q
uv run ruff check src tests
```

---

## 하지 말 것

- **gold offset 하드코딩.** 반드시 `scripts/make_questions.py`의 anchor 방식으로 유도한다.
  손으로 쓴 숫자는 문서가 바뀌면 조용히 밀리고 테스트도 안 깨진다.
- **프롬프트를 `.py`에 문자열로 박기.** config 해시에 안 잡혀 서로 다른 실험이 같은 좌표를 갖는다.
- **UI 다듬기.** 데모는 읽기 전용이고 측정에 관여하지 않는다.
- **`top_k`로 컨텍스트 양 통제.** 토큰 예산으로만 한다.
- **에러를 오답으로 집계.** `error` 필드로 분리한다.
- **파싱 실패를 오답으로 집계.** `format_fail`로 분리한다. 20~30B급은 JSON 준수율이 낮아서
  포맷 능력이 지식 효과로 둔갑한다.
- **retriever를 pack마다 다르게.** 알고리즘은 고정, 인덱스만 pack 종속.
- **합성 질문 남용.** 청크에서 역생성한 질문은 어휘 중복이 높아 검색이 부당하게 쉬워진다.
  JIRA reporter 원문 같은 실제 질의를 우선한다.
- **음수 realization을 버리기.** 컨텍스트를 줬는데 오히려 헷갈리는 건 30B급에서 흔하고,
  그 자체가 중요한 발견이다. 기록한다.