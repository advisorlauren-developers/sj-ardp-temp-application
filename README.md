# kbeval

**AI-Ready 데이터셋이 RAG 성능을 얼마나 바꾸는가**를 통제된 조건에서 재는 측정 장치.

사내 문서를 RAG 파이프라인에 넣었을 때, **데이터 전처리 수준(ai-ready 정도)이
downstream 성능을 얼마나 끌어올리는지** 숫자로 답하는 것이 목적이다.
기술지원 Q&A 형태를 띠지만 목적은 답변 품질이 아니라 **데이터 품질의 정량화**다.

> 챗봇 애플리케이션이 아니다. 동봉된 웹 앱은 결과를 보여주고 감을 잡게 해주는
> 계측기 눈금이지, 측정에 관여하지 않는다.

---

## 왜 절대 점수를 쓰지 않는가

"RAG 붙였더니 정답률 80%"는 쓸모가 없다. **모델이 원래 잘하는 것**과
**데이터가 좋은 것**이 섞여 있어 분리되지 않기 때문이다.

그래서 같은 문항을 세 가지 조건으로 물어보고 **차이**만 본다.

| mode | 주는 것 | 재는 것 |
|---|---|---|
| `bare` | 아무것도 안 줌 | 모델이 원래 아는 양 |
| `rag` | 이 pack이 검색해온 청크 | 이 pack이 실제로 전달한 양 |
| `oracle_ref` | 사람이 정제한 참조 문단 | 지식으로 얻을 수 있는 총량 |

```
headroom    = oracle_ref − bare      지식으로 얻을 수 있는 총량
lift        = rag        − bare      이 pack이 실제로 얻은 양
realization = lift / headroom        실현율  ← 1차 관측량
kendall τ   = 모델 간 pack 순위 일치도  ← 온프렘 전이 가능성
```

`realization`은 "줄 수 있었던 것 중 실제로 준 비율"이다. 모델이 원래 잘하든 못하든
분모·분자에서 같이 상쇄되므로, **데이터에 대한 진술**로 읽을 수 있다.

τ ≥ 0.7이면 pack 순위가 모델과 무관하다는 뜻이고, 그때 Bedrock 대리 실험의 결론을
온프렘 모델로 옮겨도 된다.

### 세 축

| 축 | 값 | 담당 |
|---|---|---|
| `agent` | 기반 모델 (Qwen3-32B, Gemma3-27B, …) | `llm/` |
| `mode` | 지식 주입 방식 (`bare`/`rag`/`oracle_ref`) | `agents/` |
| `pack` | ai-ready 처리 수준 (`raw`/`mid`/`full`) | `sources/` + `packs/` |

세 축은 서로를 모른다. `tests/test_architecture.py`가 매 실행마다 검증한다.

스윕은 교차곱이 **아니다**. `bare`와 `oracle_ref`는 pack 독립이므로
`N × (2 + M)` — 모델 4 × pack 3이면 36이 아니라 **20 specs**다.

---

## 빠른 시작

```bash
uv sync
uv run kbeval run      # 스윕 (기본 config는 mock 모델 — AWS 불필요, 토큰 0원)
uv run kbeval score    # 채점
uv run kbeval web      # http://127.0.0.1:8000
```

포트가 쓰이는 중이면 `--port 8010`. 빈 포트를 알려주고 종료하므로,
동작하지 않는 주소를 먼저 출력하는 일은 없다.

mock은 1급 시민이다. AWS 없이 전 파이프라인이 돌아야 **깨진 배관을 나쁜 데이터로
오해하지 않는다.**

### 웹 화면

표면이 둘이고 섞지 않는다. 앱에서 하는 일은 증거를 만들지 않고, 측정의 모든 숫자는
`kbeval run`에서 나왔다.

| 앱 | |
|---|---|
| `/` | 랜딩 — Q&A · 버그 · 기능추가 |
| `/qa` | **채팅.** 질문 → 답변 → 근거 문단 → baseline 비교. 기록하지 않음 |

| 측정 | |
|---|---|
| `/lab` | 실현율 + health + 기권/유해 + kendall τ |
| `/lab/retrieval` | pack별 hit@k · gold_injected · 공변량 |
| `/lab/questions` | 문항별 3-mode 비교 + judge 판정 |

`/qa`에서 코퍼스 밖 질문을 던지면 이렇게 나온다 — 검색이 **빈손으로 정직하게**
돌아오고, 지어내지 않는다:

```
Q: 중력방정식에 대해 설명해줘
A: 이 질문은 통신 장비 기술지원 범위와 관련성이 없습니다.
   중력방정식은 물리학에서 다루는 내용입니다.
   기술지원 엔지니어로서 답변은 제공할 수 없습니다.
   → 검색 후보 0건, 주입 0청크
```

---

## 실측 (Bedrock)

`.env`에 `BEDROCK_API_KEY`를 넣으면 된다. **키는 config에 절대 들어가지 않는다** —
들어가면 run_id에 해시되고 `config.resolved.yaml`에 복사되고 config 해시를 찍는
웹 페이지에 렌더링된다.

```bash
BED="-c configs/base.yaml -c configs/bedrock.yaml -c configs/judge.yaml"

uv run kbeval ask qa_001 --agent qwen32 --mode rag --pack mid $BED  # 단발, 가장 쌈
uv run kbeval run --limit 1 $BED     # 스펙당 1문항 스모크
uv run kbeval run $BED               # 20 spec × 7 문항 = 140 호출
uv run kbeval score $BED             # judge = claude-sonnet-5
uv run kbeval report $BED
```

**순서를 지킨다.** 단발 → `--limit 1` → 전체. 1차 실측에서 함정 5개가 전부
스모크 단계에서 잡혔다 (빈 system 블록 거부, `temperature` deprecated,
reasoning이 출력 예산 잠식, `us.` 접두사 필수, …). 자세한 내용은
[`claude.md`](claude.md)의 S9 절.

---

## 현재 상태

| 단계 | 내용 | 확인 명령 |
|---|---|---|
| S1–S4 | types / config / source / chunker / transform / gold 검증 | `kbeval pack <name>` |
| S5 | BM25 검색 (한글 음절 bigram) | `kbeval retrieval-eval` |
| S6 | LLM · Agent · 토큰 예산 | `kbeval ask` |
| S7 | Runner (재개 가능) | `kbeval run` |
| S8 | Scorer · Report | `kbeval score` / `kbeval report` |
| S9 | Bedrock 실측 (1차) | |
| S10 | 읽기 전용 뷰어 | `kbeval web` |

```
$ uv run pytest -q
414 passed
```

### 1차 실측 결과 — 배관은 맞고, 데이터가 부족하다

실모델 4종(qwen3-32b, gemma-3-27b, gpt-oss-20b, nemotron-nano-3-30b) × 7문항,
judge는 claude-sonnet-5. candidate 에러 0, judge 실패 0, 파싱 실패 0.

```
agent       pack   bare    rag  oracle  headroom  lift  realiz
qwen32      raw   0.200  1.000   1.000     0.800  0.800  1.000
gemma27     raw   0.000  1.000   1.000     1.000  1.000  1.000
gptoss20    raw   0.000  1.000   1.000     1.000  1.000  1.000
nemotron30  raw   0.000  1.000   0.800     0.800  1.000  1.250   ← rag > oracle
                          (mid/full 전부 동일)
kendall τ: 전 쌍 undefined — 모든 pack 동률
```

**모든 모델이 모든 pack에서 만점이다.** 지식만 주면 다 맞히므로 전처리가 이길
여지가 없다. 리포트가 이걸 스스로 진단해 출력한다 — 표만 던져두면 "전처리는 효과
없다"로 읽히기 때문이다:

> every pack scored identically for every model — the pack axis did not
> discriminate. Read this as a property of the **QUESTION SET**, not of the packs.

곁가지로 나온 것 하나: `gemma27`·`nemotron30`이 unanswerable 문항에서 `raw` pack일
때는 100% 기권했는데 `mid`/`full`일 때는 50%만 기권했다. **잘 정리된 컨텍스트가
오히려 자신 있게 지어내게 만들었다.** n=2라 결론은 아니지만, unanswerable 문항이
존재하는 이유가 정확히 이 메커니즘이다.

### 다음에 필요한 것은 코드가 아니다

현재 코퍼스는 문서 2개 862자, 문항 7개다. **문항이 먼저다.**

1. **어려운 문항** — 여러 문단에 흩어진 근거, 표·절차 참조, 유사 오류코드 혼동
2. **문서 수십 건** — 검색이 실패할 수 있어야 hit@k가 의미를 갖고,
   토큰 예산이 실제로 걸려야 통제변수가 작동한다
3. **중복과 표기 흔들림** — 지금은 `dedup`·`normalize_terms`가 할 일이 없어
   `full`이 `mid`의 바이트 단위 복제본이다

---

## 구조

```
src/kbeval/   개발자. 실험마다 바뀌지 않음
configs/      실험자. 실험마다 바뀜
prompts/      실험자. 버전 고정 대상 (파일 내용이 config 해시에 들어간다)
data/         도메인 전문가
runs/         프로그램만 append. 손대지 않음
tests/        단계별 보증
```

폴더는 교체 가능한 것(`llm/ sources/ packs/ retrieval/ agents/ eval/scorers/`),
파일은 하나뿐인 것(`config.py budget.py sweep.py runner.py wiring.py web.py`).

새 구현 추가 = **파일 하나 + 데코레이터 한 줄 + `__init__.py` 한 줄.**

---

## 설계에서 양보하지 않은 것들

전체 목록과 이유는 [`claude.md`](claude.md)에 있다. 성격만 옮기면:

- **offset 보존.** 청크 좌표가 원본 문서를 가리키므로 gold를 **한 번만** 라벨링하면
  모든 pack에 자동 매핑된다. 나중에 못 붙이는 장치다.
- **통제변수는 토큰 예산이지 `top_k`가 아니다.** 예산에 안 들어가는 청크는
  자르지 않고 건너뛴다 — 자르면 측정 대상이 훼손된다.
- **세 mode가 같은 system prompt·같은 템플릿을 쓴다.** 클래스 구조로 강제한다.
  실현율의 모든 항이 이 셋의 차이라서, 어긋나면 조용히 무효가 된다.
- **에러는 오답이 아니다.** 프로바이더 실패는 `error`, judge 파싱 실패는
  `format_fail`로 분리한다. 0점으로 세면 장애가 "데이터 품질"로 둔갑한다.
- **gold 생존 검증.** transform이 정답 청크를 지우면 스윕이 시작조차 하지 않는다.
  리포트 시점엔 이미 돈을 쓴 뒤다.
- **judge는 run 해시에서 뺀다.** judge는 답변을 바꾸지 않고 채점 방식만 바꾼다.
  넣으면 judge 교체가 건드리지도 않은 trace의 좌표를 무효화한다.
- **음수 realization을 버리지 않는다.** 컨텍스트가 오히려 헷갈리게 하는 건
  30B급에서 흔하고, 그 자체가 중요한 발견이다.

---

## 배포 맥락

최종 목표는 **온프렘 30B급 서빙**이다. Bedrock은 대리 환경이다.
그래서 `context_token_budget`은 Bedrock의 능력이 아니라 **온프렘 현실(8K 컨텍스트,
양자화)** 기준으로 잡는다 — 현재 3000 토큰.

지식 주입 방식은 **in-context RAG로 고정**한다. fine-tuning은 데이터 품질과 학습
하이퍼파라미터가 교락되어 측정이 불가능해진다.
