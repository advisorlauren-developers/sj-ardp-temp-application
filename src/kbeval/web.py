"""Read-only viewer for finished runs. One file, no build step, no state.

What this is NOT: a second way to run the experiment. Measurement has exactly
one entry point -- the CLI -- because two paths that both produce traces would
fork the reproduction coordinate, and a number whose provenance is ambiguous is
worth less than no number.

So the contract is narrow and enforced by what the code can reach:

    it READS   runs/<run_id>/{traces,scores}.jsonl, gold_coverage.json,
               config.resolved.yaml, and pack manifests
    it WRITES  nothing under runs/, ever

`/qa` is the one interactive page. It answers an ad-hoc question through the
same agents the sweep uses, and throws the result away -- nothing is appended,
no run id is minted, and the page says so. It exists so a reader can feel the
baseline difference on their own question, which a table cannot convey, and it
shows the retrieved passages alongside the answer: the sources ARE the evidence
that retrieval happened.

The landing page offers three intents -- Q&A, bug, feature. That is not
decoration: `intent` is a field on Question and `eval.intents` is a config
list, so the two inactive cards are waiting on data, not on a screen. They say
exactly what has to arrive for them to open.

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
from .wiring import pack_for

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
.hero{padding:56px 0 8px}
.hero h1{font-size:40px;letter-spacing:-.03em}
.hero p{color:var(--muted);font-size:15px;margin:0}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:34px}
.pick{display:block;background:var(--surface);border:1px solid var(--line);
  border-radius:16px;padding:26px 22px 22px;color:inherit;transition:.18s ease;
  transition-property:border-color,transform}
a.pick:hover{border-color:var(--accent);transform:translateY(-2px);text-decoration:none}
.pick .name{font-size:17px;font-weight:600;letter-spacing:-.015em}
.pick .desc{color:var(--muted);font-size:13px;margin-top:6px;line-height:1.55}
.pick .go{margin-top:18px;font-size:12px;color:var(--accent);font-weight:500}
.pick.off{opacity:.55}
.pick.off .go{color:var(--muted)}
.ask{background:var(--surface);border:1px solid var(--line);border-radius:16px;
  padding:20px;margin-top:18px}
.ask form{margin:0}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}
.chip{border:1px solid var(--line);border-radius:980px;padding:6px 13px;
  font-size:12px;color:var(--muted);background:var(--bg)}
a.chip:hover{border-color:var(--accent);color:var(--accent);text-decoration:none}
.reply{background:var(--surface);border:1px solid var(--line);border-radius:16px;
  padding:24px 26px;margin-top:22px}
.reply .lead{font-size:11px;font-weight:500;color:var(--muted);
  text-transform:uppercase;letter-spacing:.06em}
.reply .text{font-size:16px;line-height:1.75;margin-top:10px;white-space:pre-wrap}
.reply .foot{margin-top:16px;padding-top:14px;border-top:1px solid var(--line);
  font-size:11px;color:var(--muted);font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.src{border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-top:10px;
  background:var(--surface)}
.src .head{display:flex;gap:10px;align-items:baseline;font-size:11px;color:var(--muted);
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.src .head b{color:var(--ink);font-family:inherit;font-size:12px;font-weight:600}
.src .body{font-size:13px;line-height:1.65;margin-top:8px;white-space:pre-wrap}
.base{background:var(--bg);border:1px solid var(--line);border-radius:12px;
  padding:16px 18px;margin-top:10px}
.base .text{font-size:14px;line-height:1.7;color:var(--muted);white-space:pre-wrap}
.chat{display:flex;flex-direction:column;gap:18px;margin-top:20px;min-height:46vh}
.turn{display:flex;flex-direction:column;gap:8px}
.turn.me{align-items:flex-end}
.bubble{max-width:min(78%,640px);padding:13px 17px;border-radius:20px;
  font-size:14px;line-height:1.7;white-space:pre-wrap;word-break:break-word}
.turn.me .bubble{background:var(--accent);color:#fff;border-bottom-right-radius:6px}
.turn.bot .bubble{background:var(--surface);border:1px solid var(--line);
  border-bottom-left-radius:6px}
.turn.bot .bubble.err{color:var(--warn)}
.meta{font-size:11px;color:var(--muted);
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.turn.bot details{max-width:min(78%,640px);font-size:12px}
.turn.bot summary{cursor:pointer;color:var(--muted);padding:5px 2px;list-style:none}
.turn.bot summary::-webkit-details-marker{display:none}
.turn.bot summary::before{content:"▸ ";color:var(--muted)}
.turn.bot details[open] summary::before{content:"▾ "}
.turn.bot summary:hover{color:var(--accent)}
.drawer{border:1px solid var(--line);border-radius:12px;background:var(--surface);
  padding:4px 14px 12px;margin-top:4px}
.composer{position:sticky;bottom:0;background:linear-gradient(to bottom,
  rgba(251,251,253,0) 0%,var(--bg) 26%);padding:20px 0 8px;margin-top:8px}
.composer form{margin:0;align-items:center}
.thinking{color:var(--muted);font-size:13px;padding:6px 2px}
.empty-chat{text-align:center;color:var(--muted);padding:34px 0 10px;font-size:13px}
.ghost{background:var(--surface);color:var(--ink);border:1px solid var(--line)}
.ghost:hover{border-color:var(--accent);opacity:1}
@media (max-width:760px){.grid,.cards{grid-template-columns:1fr}
  .bubble,.turn.bot details{max-width:88%}
  main{padding:28px 18px 64px}h1{font-size:26px}.hero h1{font-size:30px}
  .bar{gap:16px;flex-wrap:wrap}}
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
  <nav>{link('/', '홈', 'home')}{link('/qa', 'Q&amp;A', 'qa')}
       {link('/report', '실현율', 'overview')}{link('/retrieval', '검색', 'retrieval')}
       {link('/questions', '문항', 'questions')}</nav>
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

@app.get("/report", response_class=HTMLResponse)
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


INTENTS = [
    ("qa", "Q&A", "/qa",
     "사내 문서를 근거로 기술 질문에 답합니다. 지식 주입 전후를 나란히 보여줍니다."),
    ("bug", "버그", "/bug",
     "장애 신고를 과거 처리 이력에 연결합니다."),
    ("feature", "기능 추가", "/feature",
     "요청을 기존 스펙·중복 요청과 대조합니다."),
]


def pack_label(c: Config, name: str) -> str:
    """Name the transforms, not their count.

    "변환 2개" vs "변환 4개" tells a reader that something differs but not what,
    which is the one thing the pack axis is supposed to make legible.
    """
    pc = c.packs[name]
    steps = ", ".join(pc.transforms) if pc.transforms else "변환 없음"
    return f"{name} · {pc.chunker} · {steps}"


def identical_packs(c: Config) -> list[list[str]]:
    """Packs whose chunks came out byte-identical.

    Worth saying out loud: if two levels produce the same corpus, switching
    between them changes nothing, and a reader who sees the same answer twice
    will read it as a broken selector or as a null result. It is neither -- it
    means those transforms had nothing to do on THIS corpus.
    """
    groups: dict[tuple[str, ...], list[str]] = {}
    for name in c.sweep.pack:
        sig = tuple(ch.text for ch in pack_for(c, name).chunks)
        groups.setdefault(sig, []).append(name)
    return [g for g in groups.values() if len(g) > 1]


@app.get("/", response_class=HTMLResponse)
def home(run: str | None = None) -> HTMLResponse:
    c = cfg()
    live = set(c.eval.intents)

    cards = ""
    for key, name, href, desc in INTENTS:
        on = key in live
        go = "열기 →" if on else "준비 중"
        tag = "a" if on else "div"
        attr = f' href="{href}"' if on else ""
        cards += (f'<{tag} class="pick{"" if on else " off"}"{attr}>'
                  f'<div class="name">{e(name)}</div>'
                  f'<div class="desc">{e(desc)}</div>'
                  f'<div class="go">{go}</div></{tag}>')

    body = f"""<div class="hero">
<h1>무엇을 물어보시겠습니까?</h1>
<p>사내 문서를 RAG에 넣었을 때 답변이 실제로 얼마나 좋아지는지 측정하는 도구입니다.</p>
</div>
<div class="cards">{cards}</div>
<div class="banner">활성화된 유형은 <b>{e(", ".join(sorted(live)))}</b> 하나입니다.
나머지는 화면이 아니라 <b>데이터를 기다리는 중</b>입니다 —
<code>data/eval/questions.jsonl</code>에 해당 <code>intent</code> 문항이 들어오고
<code>configs/base.yaml</code>의 <code>eval.intents</code>에 추가되면 열립니다.</div>"""
    return page("kbeval", "home", body, pick_run(c, run), list_runs(c))


@app.get("/bug", response_class=HTMLResponse)
@app.get("/feature", response_class=HTMLResponse)
def not_yet(run: str | None = None) -> HTMLResponse:
    c = cfg()
    body = """<h1>준비 중</h1>
<p class="sub">이 유형의 문항이 아직 평가 세트에 없습니다.</p>
<div class="banner">빈 화면을 먼저 만들지 않습니다. 라벨된 문항 없이 답변만 보여주면
좋아 보이는지 아닌지를 판단할 근거가 없고, 그건 측정이 아니라 인상입니다.<br><br>
필요한 것: <code>scripts/make_questions.py</code>에 해당 intent 문항과 gold 앵커 추가 →
<code>data/eval/questions.jsonl</code> 재생성 →
<code>configs/base.yaml</code>의 <code>eval.intents</code>에 추가.</div>
<p class="sub"><a href="/qa">Q&amp;A로 가기 →</a></p>"""
    return page("준비 중", "home", body, pick_run(c, run), list_runs(c))


def answer_payload(c: Config, question: str, pack: str) -> dict[str, Any]:
    """One question, answered exactly the way the sweep answers it.

    Single turn, no conversation history. Carrying the thread into the prompt
    would make the demo show behaviour the harness never measures -- the
    control is one question against one controlled context, and a screen that
    quietly does something else is worse than no screen.
    """
    from .types import Question
    probe = Question(qid="qa-live", intent="qa", query=question, answerable=True)
    agent = c.sweep.agent[0]

    rag_agent = build_agent_for(c, agent, "rag", pack)
    rag = rag_agent.answer(probe)
    base = build_agent_for(c, agent, "bare", None).answer(probe)
    pk = rag_agent.deps.pack
    hits = {h.chunk_id: h for h in rag.retrieved}

    sources = []
    for cid in rag.injected_chunk_ids:
        ch = pk.get(cid) if pk.has(cid) else None
        text = ch.text if ch else ""
        if text.startswith("["):            # inline_meta header sits on its own line
            text = text.split("\n", 1)[-1]
        sources.append({
            "chunk_id": cid,
            "heading": (ch.meta.get("heading_path") or ch.meta.get("title") or "") if ch else "",
            "score": round(hits[cid].score, 2) if cid in hits else None,
            "rank": hits[cid].rank if cid in hits else None,
            "text": text.strip(),
        })

    return {
        "answer": rag.text, "error": rag.error,
        "baseline": base.text, "baseline_error": base.error,
        "agent": agent, "provider": c.agents[agent].provider,
        "pack": pack, "pack_label": pack_label(c, pack),
        "injected_tokens": rag.injected_tokens,
        "budget": c.fixed.context_token_budget,
        "candidates": len(rag.retrieved),
        "skipped": len(rag.skipped_chunk_ids),
        "sources": sources,
    }


@app.get("/api/ask")
def api_ask(q: str, pack: str = "") -> dict[str, Any]:
    """The chat's only backend call. Reads packs, writes nothing."""
    c = cfg()
    pack = pack if pack in c.packs else next(iter(c.sweep.pack))
    if not q.strip():
        return {"answer": "", "error": "empty_question", "sources": []}
    return answer_payload(c, q.strip(), pack)


CHAT_JS = """
const KEY = 'kbeval.thread';
const thread = document.getElementById('thread');
const form = document.getElementById('composer');
const input = document.getElementById('q');
const packSel = document.getElementById('pack');
const empty = document.getElementById('empty');
let turns = [];

function store() {
  try { sessionStorage.setItem(KEY, JSON.stringify(turns)); } catch (_) {}
}
function restore() {
  try { turns = JSON.parse(sessionStorage.getItem(KEY) || '[]'); } catch (_) { turns = []; }
  turns.forEach(t => render(t.q, t.a));
  if (turns.length) empty.hidden = true;
}
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;   // never innerHTML: answers are data
  return n;
}
function drawer(summaryText, build) {
  const d = el('details');
  d.appendChild(el('summary', null, summaryText));
  const box = el('div', 'drawer');
  build(box);
  d.appendChild(box);
  return d;
}
function render(question, a) {
  const me = el('div', 'turn me');
  me.appendChild(el('div', 'bubble', question));
  thread.appendChild(me);

  const bot = el('div', 'turn bot');
  const body = a.error ? ('오류 · ' + a.error) : a.answer;
  bot.appendChild(el('div', 'bubble' + (a.error ? ' err' : ''), body));
  bot.appendChild(el('div', 'meta',
    `${a.agent} (${a.provider}) · ${a.pack_label} · 주입 ${a.sources.length}청크 `
    + `${a.injected_tokens}/${a.budget} tok`
    + (a.skipped ? ` · 예산 초과 ${a.skipped}건 제외` : '')
    + ` · 검색 후보 ${a.candidates}건`));

  if (a.sources.length) {
    bot.appendChild(drawer(`근거로 쓰인 문단 ${a.sources.length}개`, box => {
      a.sources.forEach((s, i) => {
        const h = el('div', 'head');
        h.appendChild(el('span', null, String(i + 1)));
        h.appendChild(el('b', null, s.heading || s.chunk_id));
        h.appendChild(el('span', null, s.chunk_id));
        if (s.score !== null) h.appendChild(el('span', null, 'score ' + s.score));
        const card = el('div', 'src');
        card.appendChild(h);
        card.appendChild(el('div', 'body', s.text));
        box.appendChild(card);
      });
    }));
  } else {
    bot.appendChild(el('div', 'meta', '검색된 문단이 없습니다 — 질문의 어휘가 코퍼스와 겹치지 않습니다.'));
  }

  bot.appendChild(drawer('지식 주입 없이 같은 질문을 하면', box => {
    box.appendChild(el('div', 'base',
      a.baseline_error ? ('오류 · ' + a.baseline_error) : a.baseline));
    box.appendChild(el('div', 'meta',
      '같은 모델, 같은 프롬프트, 자료만 뺀 baseline입니다. 이 차이가 측정 대상입니다.'));
  }));
  thread.appendChild(bot);
}
async function ask(question) {
  empty.hidden = true;
  const me = el('div', 'turn me');
  me.appendChild(el('div', 'bubble', question));
  thread.appendChild(me);
  const wait = el('div', 'thinking', '문서를 검색하는 중…');
  thread.appendChild(wait);
  window.scrollTo(0, document.body.scrollHeight);

  try {
    const r = await fetch('/api/ask?q=' + encodeURIComponent(question)
                          + '&pack=' + encodeURIComponent(packSel.value));
    const a = await r.json();
    wait.remove();
    me.remove();
    render(question, a);
    turns.push({ q: question, a: a });
    store();
  } catch (err) {
    wait.textContent = '요청이 실패했습니다 · ' + err;
  }
  window.scrollTo(0, document.body.scrollHeight);
}
form.addEventListener('submit', ev => {
  ev.preventDefault();
  const v = input.value.trim();
  if (!v) return;
  input.value = '';
  ask(v);
});
document.querySelectorAll('[data-sample]').forEach(b =>
  b.addEventListener('click', ev => { ev.preventDefault(); ask(b.dataset.sample); }));
document.getElementById('clear').addEventListener('click', () => {
  turns = []; store(); thread.replaceChildren(); empty.hidden = false;
});
restore();
"""


@app.get("/qa", response_class=HTMLResponse)
def qa(pack: str = "", run: str | None = None) -> HTMLResponse:
    c = cfg()
    rid = pick_run(c, run)
    pack = pack if pack in c.packs else next(iter(c.sweep.pack))

    opts = "".join(
        f'<option value="{e(p)}"{" selected" if p == pack else ""}>{e(pack_label(c, p))}</option>'
        for p in c.sweep.pack)
    samples = "".join(
        f'<button type="button" class="chip" data-sample="{html.escape(x.query, quote=True)}">'
        f'{e(x.query[:30])}…</button>'
        for x in load_questions(c.eval.questions, c.eval.intents)[:3])

    twins = identical_packs(c)
    twin_note = ""
    if twins:
        pairs = " / ".join(" = ".join(g) for g in twins)
        twin_note = (f'<div class="banner"><b>{e(pairs)}</b> 는 현재 코퍼스에서 '
                     f'청크가 완전히 동일합니다. 선택을 바꿔도 답이 같은 것은 고장이 아니라, '
                     f'그 전처리 단계가 이 문서들에 대해 <b>할 일이 없다</b>는 뜻입니다 — '
                     f'중복이 없어 <code>dedup</code>이, 표기가 이미 일관돼 '
                     f'<code>normalize_terms</code>가 바꿀 것을 찾지 못합니다.</div>')

    body = f"""<h1>기술지원 Q&amp;A</h1>
<p class="sub">사내 문서(OLT·BNG 가이드)를 근거로 답합니다. 근거 문단과 baseline을 함께 펼쳐 볼 수 있습니다.</p>
{mock_banner(c)}{twin_note}
<div class="banner"><b>측정이 아닙니다.</b> 여기서 던진 질문은 <code>runs/</code>에
기록되지 않고 run id도 만들지 않습니다. 라벨된 gold가 없어 채점도 없습니다 —
점수가 필요하면 <code>kbeval run</code> 경로 하나뿐입니다.<br>
<b>각 질문은 독립적으로 답합니다.</b> 앞의 대화를 컨텍스트로 넣지 않습니다.
그래야 화면에 보이는 동작이 실제로 측정하는 동작과 같습니다.</div>

<noscript><div class="banner">이 화면은 자바스크립트가 필요합니다.
같은 일을 터미널에서 하려면 <code>kbeval ask qa_001 --mode rag --pack mid</code>.</div></noscript>

<div class="chat" id="thread"></div>
<div class="empty-chat" id="empty">질문을 입력해 보세요.<div class="chips"
  style="justify-content:center">{samples}</div></div>

<div class="composer"><form id="composer" autocomplete="off">
<input type="text" id="q" placeholder="예: OLT에서 E411 인증 실패가 뜨는데 어떻게 조치하나요?" autofocus>
<select id="pack">{opts}</select>
<button type="submit">전송</button>
<button type="button" class="ghost" id="clear">초기화</button>
</form></div>
<script>{CHAT_JS}</script>"""
    return page("Q&A", "qa", body, rid, list_runs(c))
