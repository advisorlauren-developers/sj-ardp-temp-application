"""Read-only viewer for finished runs. One file, no build step, no state.

What this is NOT: a second way to run the experiment. Measurement has exactly
one entry point -- the CLI -- because two paths that both produce traces would
fork the reproduction coordinate, and a number whose provenance is ambiguous is
worth less than no number.

So the contract is narrow and enforced by what the code can reach:

    it READS   runs/<run_id>/{traces,scores}.jsonl, gold_coverage.json,
               config.resolved.yaml, and pack manifests
    it WRITES  nothing under runs/, ever

`/explore` is the one interactive page. It answers an ad-hoc question through
the same agents the sweep uses, and throws the result away -- nothing is
appended, no run id is minted, and the page says so in the banner. It exists so
a reader can feel the bare / rag / oracle_ref difference on their own question,
which a table cannot convey.

The styling is deliberate and deliberately small: this is an instrument
readout, so it gets whitespace, tabular figures and one accent colour, and
nothing that moves.
"""
from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .config import Config, load_config
from .eval import build_report, load_scores
from .eval.report import Report
from .packs import load_questions
from .runner import RunPaths, read_traces
from .wiring import build as build_agent_for

app = FastAPI(title="kbeval", docs_url=None, redoc_url=None)

CONFIG = os.environ.get("KBEVAL_CONFIG", "configs/base.yaml").split(",")

CSS = """
:root{
  --ink:#1D1D1F; --accent:#0A2540; --bg:#FBFBFD; --surface:#FFFFFF;
  --line:#E8E8ED; --muted:#6E6E76; --warn:#B4232A; --track:#F0F0F3;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Pretendard","Apple SD Gothic Neo",
    "Segoe UI",Roboto,"Helvetica Neue",sans-serif;
  font-size:14px;line-height:1.6;-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
header{position:sticky;top:0;z-index:10;background:rgba(251,251,253,.82);
  backdrop-filter:saturate(180%) blur(20px);border-bottom:1px solid var(--line)}
.bar{max-width:1080px;margin:0 auto;padding:14px 24px;display:flex;
  align-items:baseline;gap:28px}
.brand{font-size:17px;font-weight:600;letter-spacing:-.02em;color:var(--ink)}
nav{display:flex;gap:22px;font-size:13px}
nav a{color:var(--muted)}
nav a.on{color:var(--ink);font-weight:500}
main{max-width:1080px;margin:0 auto;padding:40px 24px 96px}
h1{font-size:32px;line-height:1.2;font-weight:600;letter-spacing:-.025em;margin:0 0 6px}
h2{font-size:15px;font-weight:600;letter-spacing:-.01em;margin:44px 0 4px}
.sub{color:var(--muted);font-size:13px;margin:0 0 18px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:14px;
  padding:4px 20px 8px;margin-top:12px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th{text-align:right;font-size:11px;font-weight:500;color:var(--muted);
  text-transform:uppercase;letter-spacing:.05em;padding:14px 10px 10px;
  border-bottom:1px solid var(--line);white-space:nowrap}
td{text-align:right;padding:11px 10px;border-bottom:1px solid var(--line)}
tr:last-child td{border-bottom:none}
th:first-child,td:first-child{text-align:left}
td.k{font-weight:500}
.dim{color:var(--muted)}
.note{color:var(--muted);font-size:12px;text-align:left}
.neg{color:var(--warn)}
.track{display:inline-block;width:90px;height:5px;border-radius:3px;
  background:var(--track);vertical-align:middle;margin-left:10px;overflow:hidden}
.fill{display:block;height:100%;border-radius:3px;background:var(--accent)}
.fill.neg{background:var(--warn)}
.pills{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0 0}
.pill{border:1px solid var(--line);background:var(--surface);border-radius:980px;
  padding:5px 13px;font-size:12px;color:var(--muted)}
.pill b{color:var(--ink);font-weight:600}
.pill.bad b{color:var(--warn)}
.banner{border:1px solid var(--line);border-left:3px solid var(--accent);
  background:var(--surface);border-radius:10px;padding:13px 16px;margin:20px 0;
  font-size:13px;color:var(--muted)}
.banner b{color:var(--ink)}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:14px}
.mode{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:18px}
.mode h3{margin:0 0 2px;font-size:13px;font-weight:600;letter-spacing:-.01em}
.mode .meta{font-size:11px;color:var(--muted);margin-bottom:12px}
.answer{font-size:13px;line-height:1.7;white-space:pre-wrap;word-break:break-word}
.chunks{margin-top:14px;padding-top:12px;border-top:1px solid var(--line);
  font-size:11px;color:var(--muted);font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
form{display:flex;gap:10px;margin-top:16px;flex-wrap:wrap}
input[type=text],select{font:inherit;padding:11px 14px;border:1px solid var(--line);
  border-radius:10px;background:var(--surface);color:var(--ink);min-width:0}
input[type=text]{flex:1}
input[type=text]:focus,select:focus{outline:none;border-color:var(--accent)}
button{font:inherit;font-weight:500;padding:11px 20px;border:none;border-radius:10px;
  background:var(--accent);color:#fff;cursor:pointer}
button:hover{opacity:.9}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.empty{color:var(--muted);padding:26px 0;text-align:center}
@media (max-width:760px){.grid{grid-template-columns:1fr}main{padding:28px 18px 64px}
  h1{font-size:26px}.bar{gap:16px;flex-wrap:wrap}}
"""


# --- plumbing ------------------------------------------------------------

def cfg() -> Config:
    """Loaded per request: the viewer must never hold a stale coordinate."""
    return load_config(CONFIG)


def runs_root(c: Config) -> Path:
    return Path(c.run.out_dir)


def list_runs(c: Config) -> list[str]:
    root = runs_root(c)
    if not root.exists():
        return []
    return sorted((d.name for d in root.iterdir() if (d / "traces.jsonl").exists()),
                  key=lambda n: (runs_root(c) / n / "traces.jsonl").stat().st_mtime,
                  reverse=True)


def pick_run(c: Config, run: str | None) -> str | None:
    available = list_runs(c)
    if run and run in available:
        return run
    if c.run_id in available:
        return c.run_id
    return available[0] if available else None


def e(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def num(v: float | None, nd: int = 3) -> str:
    if v is None:
        return '<span class="dim">–</span>'
    cls = ' class="neg"' if v < 0 else ""
    return f"<span{cls}>{v:.{nd}f}</span>"


def bar(v: float | None, lo: float = 0.0, hi: float = 1.0) -> str:
    if v is None:
        return ""
    pct = max(0.0, min(1.0, (abs(v) - lo) / (hi - lo) if hi > lo else 0.0)) * 100
    cls = "fill neg" if v < 0 else "fill"
    return f'<span class="track"><span class="{cls}" style="width:{pct:.1f}%"></span></span>'


def page(title: str, active: str, body: str, run: str | None, runs: list[str]) -> HTMLResponse:
    def link(href: str, label: str, key: str) -> str:
        q = f"?run={e(run)}" if run else ""
        return f'<a class="{"on" if active == key else ""}" href="{href}{q}">{label}</a>'

    picker = ""
    if len(runs) > 1:
        opts = "".join(f'<option value="{e(r)}"{" selected" if r == run else ""}>{e(r)}</option>'
                       for r in runs)
        picker = (f'<form method="get" style="margin:0"><select name="run" '
                  f'onchange="this.form.submit()">{opts}</select></form>')

    return HTMLResponse(f"""<!doctype html><html lang="ko"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)} · kbeval</title><style>{CSS}</style></head><body>
<header><div class="bar">
  <span class="brand">kbeval</span>
  <nav>{link('/', '개요', 'overview')}{link('/retrieval', '검색', 'retrieval')}
       {link('/questions', '문항', 'questions')}{link('/explore', '탐색', 'explore')}</nav>
  <span style="margin-left:auto">{picker}</span>
</div></header><main>{body}</main></body></html>""")


def no_runs(active: str) -> HTMLResponse:
    body = ("<h1>기록된 run이 없습니다</h1>"
            '<p class="sub">측정은 CLI에서만 실행됩니다. 이 화면은 결과를 읽기만 합니다.</p>'
            '<div class="banner"><b>uv run kbeval run</b> 로 스윕을 돌리고 '
            '<b>uv run kbeval score</b> 로 채점한 뒤 새로고침하세요.</div>')
    return page("no runs", active, body, None, [])


def health_pills(rep: Report) -> str:
    def pill(label: str, value: Any, bad: bool = False) -> str:
        return f'<span class="pill{" bad" if bad else ""}">{label} <b>{e(value)}</b></span>'
    return ('<div class="pills">'
            + pill("traces", rep.n_traces)
            + pill("errors", rep.errors, rep.errors > 0)
            + pill("judge format_fail", rep.format_fails, rep.format_fails > 0)
            + pill("unreadable", rep.malformed, rep.malformed > 0)
            + pill("config", rep.config_hash)
            + "</div>")


def mock_banner(c: Config) -> str:
    providers = {a.provider for a in c.agents.values()} | {c.judge.agent.provider}
    if not providers <= {"mock", "mock_judge"}:
        return ""
    return ('<div class="banner"><b>mock 실행 결과입니다.</b> 배관이 도는지의 증거이지 '
            '데이터 품질의 증거가 아닙니다. 숫자를 pack 비교로 읽지 마세요.</div>')


# --- pages ---------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def overview(run: str | None = None) -> HTMLResponse:
    c = cfg()
    rid = pick_run(c, run)
    if not rid:
        return no_runs("overview")
    rep = build_report(c, runs_root(c) / rid)

    rows = ""
    for r in rep.realization:
        note = r.note or ""
        rows += (f"<tr><td class='k'>{e(r.agent)}</td><td class='k'>{e(r.pack)}</td>"
                 f"<td>{num(r.bare)}</td><td>{num(r.rag)}</td><td>{num(r.oracle)}</td>"
                 f"<td>{num(r.headroom)}</td><td>{num(r.lift)}</td>"
                 f"<td>{num(r.realization)}{bar(r.realization)}</td>"
                 f"<td class='dim'>{r.n}</td><td class='note'>{e(note)}</td></tr>")

    tau = ""
    for t in rep.tau:
        tau += (f"<tr><td class='k'>{e(t['pair'])}</td><td>{num(t.get('tau'))}</td>"
                f"<td class='note'>{e(t.get('note') or '')}</td></tr>")

    abst = ""
    for a in rep.abstention:
        abst += (f"<tr><td class='k'>{e(a['agent'])}</td><td>{e(a['mode'])}</td>"
                 f"<td>{e(a['pack'] or '–')}</td><td class='dim'>{a['n']}</td>"
                 f"<td>{num(a['abstention'])}</td><td>{num(a['harmful_action'])}</td></tr>")

    notes = "".join(f'<div class="banner">{e(n)}</div>' for n in rep.notes)

    body = f"""<h1>실현율</h1>
<p class="sub">answerable 문항만. <code>realization = (rag − bare) / (oracle_ref − bare)</code></p>
{health_pills(rep)}{mock_banner(c)}{notes}
<div class="card"><table>
<tr><th>agent</th><th>pack</th><th>bare</th><th>rag</th><th>oracle</th>
    <th>headroom</th><th>lift</th><th>realization</th><th>n</th><th style="text-align:left">note</th></tr>
{rows or '<tr><td colspan="10" class="empty">채점 결과가 없습니다 — kbeval score</td></tr>'}
</table></div>

<h2>기권 · 유해 조치</h2>
<p class="sub">unanswerable 문항만. 실현율 계산에서 제외되고 기권으로만 평가합니다.</p>
<div class="card"><table>
<tr><th>agent</th><th>mode</th><th>pack</th><th>n</th><th>abstention</th><th>harmful</th></tr>
{abst or '<tr><td colspan="6" class="empty">–</td></tr>'}
</table></div>

<h2>모델 간 pack 순위 일치도</h2>
<p class="sub">kendall τ ≥ 0.7 이면 순위가 모델 무관 → 온프렘 전이 가능.</p>
<div class="card"><table>
<tr><th>pair</th><th>τ</th><th style="text-align:left">note</th></tr>
{tau or '<tr><td colspan="3" class="empty">–</td></tr>'}
</table></div>"""
    return page("개요", "overview", body, rid, list_runs(c))


@app.get("/retrieval", response_class=HTMLResponse)
def retrieval(run: str | None = None) -> HTMLResponse:
    c = cfg()
    rid = pick_run(c, run)
    if not rid:
        return no_runs("retrieval")
    rep = build_report(c, runs_root(c) / rid)

    ks = [k for k in (rep.retrieval[0] if rep.retrieval else {}) if k.startswith("hit@")]
    head = "".join(f"<th>{e(k)}</th>" for k in ks)
    rows = ""
    for x in rep.retrieval:
        cells = "".join(f"<td>{num(x[k])}</td>" for k in ks)
        rows += (f"<tr><td class='k'>{e(x['pack'])}</td><td class='dim'>{x['n_chunks']}</td>"
                 f"<td class='dim'>{x['avg_chunk_chars']}</td>{cells}"
                 f"<td>{num(x['mrr'])}</td><td>{num(x['gold_injected'])}</td>"
                 f"<td>{num(x['gold_lost_to_budget'])}</td>"
                 f"<td class='dim'>{num(x['injected_tokens'], 1)}</td>"
                 f"<td class='dim'>{num(x['injected_chunks'], 2)}</td></tr>")

    body = f"""<h1>검색과 예산</h1>
<p class="sub">두 개는 다른 질문입니다 — 검색이 <b>찾았는가</b>(hit@k)와
모델에게 <b>닿았는가</b>(gold_injected). 예산이 물리면 갈라집니다.</p>
{mock_banner(c)}
<div class="banner">공변량이 옆에 붙어 있습니다. 청크가 많고 작은 pack은 맞히기도 쉽고
주입도 쌉니다 — <b>n_chunks</b>와 <b>avg_chars</b> 없이는 hit@k 차이를 읽을 수 없습니다.</div>
<div class="card"><table>
<tr><th>pack</th><th>chunks</th><th>avg chars</th>{head}<th>mrr</th>
    <th>gold injected</th><th>lost to budget</th><th>inj tokens</th><th>inj chunks</th></tr>
{rows or '<tr><td colspan="12" class="empty">–</td></tr>'}
</table></div>"""
    return page("검색", "retrieval", body, rid, list_runs(c))


@app.get("/questions", response_class=HTMLResponse)
def questions(run: str | None = None) -> HTMLResponse:
    c = cfg()
    rid = pick_run(c, run)
    if not rid:
        return no_runs("questions")
    scores = load_scores(runs_root(c) / rid)

    rows = ""
    for q in load_questions(c.eval.questions, c.eval.intents):
        got = [s for s in scores if s["qid"] == q.qid]
        correct = [(s.get("judge") or {}).get("correctness") for s in got]
        correct = [v for v in correct if v is not None]
        rate = sum(correct) / len(correct) if correct else None
        kind = "answerable" if q.answerable else "unanswerable"
        rows += (f"<tr><td class='k'><a href='/q/{e(q.qid)}?run={e(rid)}'>{e(q.qid)}</a></td>"
                 f"<td style='text-align:left'>{e(q.query)}</td>"
                 f"<td class='dim'>{kind}</td>"
                 f"<td>{num(rate)}{bar(rate)}</td></tr>")

    body = f"""<h1>문항</h1>
<p class="sub">문항 하나를 열면 bare · rag · oracle_ref 답변이 나란히 보입니다.</p>
{mock_banner(c)}
<div class="card"><table>
<tr><th>qid</th><th style="text-align:left">질문</th><th>유형</th><th>정답률(전 스펙)</th></tr>
{rows}
</table></div>"""
    return page("문항", "questions", body, rid, list_runs(c))


@app.get("/q/{qid}", response_class=HTMLResponse)
def question_detail(qid: str, run: str | None = None) -> HTMLResponse:
    c = cfg()
    rid = pick_run(c, run)
    if not rid:
        return no_runs("questions")
    root = runs_root(c) / rid
    traces, _ = read_traces(RunPaths(root).traces)
    scores = {(s["spec_key"], s["qid"]): s for s in load_scores(root)}

    q = next((x for x in load_questions(c.eval.questions, c.eval.intents) if x.qid == qid), None)
    if q is None:
        return page("없는 문항", "questions",
                    f"<h1>없는 문항</h1><p class='sub'>{e(qid)}</p>", rid, list_runs(c))

    cards = ""
    for t in sorted((t for t in traces if t["qid"] == qid),
                    key=lambda t: (t["agent"], ["bare", "rag", "oracle_ref"].index(t["mode"]),
                                   t["pack"] or "")):
        v = (scores.get((t["spec_key"], qid)) or {}).get("judge") or {}
        verdict = ""
        if v.get("skipped"):
            verdict = f"<span class='neg'>채점 제외 · {e(v['skipped'])}</span>"
        elif v.get("format_fail") == 1.0:
            verdict = "<span class='neg'>judge format_fail</span>"
        elif v:
            verdict = (f"correct {e(v.get('correctness'))} · "
                       f"grounded {e(v.get('groundedness'))} · "
                       f"abstain {e(v.get('abstention'))} · "
                       f"harmful {e(v.get('harmful_action'))}")
        answer = (f"<span class='neg'>error · {e(t['error'])}</span>" if t["error"]
                  else e(t["answer"]))
        chunks = ""
        if t["injected_chunk_ids"]:
            chunks = (f"<div class='chunks'>injected {len(t['injected_chunk_ids'])} · "
                      f"{t['injected_tokens']} tok"
                      + (f" · skipped {len(t['skipped_chunk_ids'])}"
                         if t["skipped_chunk_ids"] else "")
                      + "<br>" + e(", ".join(t["injected_chunk_ids"])) + "</div>")
        cards += (f"<div class='mode'><h3>{e(t['mode'])}"
                  + (f" · {e(t['pack'])}" if t["pack"] else "")
                  + f"</h3><div class='meta'>{e(t['agent'])} · {verdict}</div>"
                  f"<div class='answer'>{answer}</div>{chunks}</div>")

    gold = (f"<div class='banner'><b>정답 요지</b><br>{e(q.gold_answer)}</div>"
            if q.gold_answer else
            "<div class='banner'><b>unanswerable</b> — 코퍼스에 근거가 없습니다. "
            "올바른 행동은 기권이며, 실현율 계산에서 제외됩니다.</div>")

    body = f"""<h1>{e(q.qid)}</h1>
<p class="sub">{e(q.query)}</p>{gold}
<div class="grid">{cards}</div>"""
    return page(q.qid, "questions", body, rid, list_runs(c))


@app.get("/explore", response_class=HTMLResponse)
def explore(q: str = "", pack: str = "", run: str | None = None) -> HTMLResponse:
    c = cfg()
    rid = pick_run(c, run)
    pack = pack if pack in c.packs else next(iter(c.sweep.pack))
    agent = c.sweep.agent[0]

    opts = "".join(f'<option value="{e(p)}"{" selected" if p == pack else ""}>{e(p)}</option>'
                   for p in c.sweep.pack)
    form = (f'<form method="get"><input type="text" name="q" value="{e(q)}" '
            f'placeholder="질문을 입력하세요 — 예: E411 인증 실패 원인이 뭔가요?">'
            f'<select name="pack">{opts}</select>'
            f'<button type="submit">비교</button></form>')

    cards = ""
    if q.strip():
        from .types import Question
        probe = Question(qid="explore", intent="qa", query=q.strip(), answerable=True)
        for mode in ("bare", "rag", "oracle_ref"):
            a = build_agent_for(c, agent, mode, pack if mode == "rag" else None)
            res = a.answer(probe)
            answer = (f"<span class='neg'>error · {e(res.error)}</span>" if res.error
                      else e(res.text))
            chunks = ""
            if res.injected_chunk_ids:
                chunks = (f"<div class='chunks'>injected {len(res.injected_chunk_ids)} · "
                          f"{res.injected_tokens} tok of {c.fixed.context_token_budget}"
                          + (f" · skipped {len(res.skipped_chunk_ids)}"
                             if res.skipped_chunk_ids else "")
                          + "<br>" + e(", ".join(res.injected_chunk_ids)) + "</div>")
            label = "oracle_ref (참조 문단 없음 — bare와 동일)" if mode == "oracle_ref" else mode
            cards += (f"<div class='mode'><h3>{e(label)}"
                      + (f" · {e(pack)}" if mode == "rag" else "")
                      + f"</h3><div class='meta'>{e(agent)} · "
                      f"{e(c.agents[agent].provider)}</div>"
                      f"<div class='answer'>{answer}</div>{chunks}</div>")
        cards = f"<div class='grid'>{cards}</div>"

    body = f"""<h1>탐색</h1>
<p class="sub">같은 질문을 세 mode에 동시에 던져 지식 주입의 차이를 직접 봅니다.</p>
<div class="banner"><b>측정이 아닙니다.</b> 이 질의는 <code>runs/</code>에 기록되지 않고
run id도 만들지 않습니다. 라벨된 gold가 없으므로 채점도 없습니다 —
<code>oracle_ref</code>는 참조 문단이 없어 <code>bare</code>와 같은 답을 냅니다.
측정은 <code>kbeval run</code> 한 경로뿐입니다.</div>
{form}{cards}"""
    return page("탐색", "explore", body, rid, list_runs(c))
