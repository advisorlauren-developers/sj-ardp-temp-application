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
.brand{font-size:17px;font-weight:600;letter-spacing:-.02em;color:var(--ink);white-space:nowrap}
.brand .tag{font-size:11px;font-weight:500;color:var(--muted);border:1px solid var(--line);
  border-radius:6px;padding:2px 6px;margin-left:4px;vertical-align:2px}
.right{margin-left:auto;display:flex;gap:14px;align-items:center}
a.quiet{color:var(--muted);font-size:12px;white-space:nowrap}
.footnote{margin-top:44px;padding-top:16px;border-top:1px solid var(--line);
  color:var(--muted);font-size:12px;line-height:1.7}
.footnote b{color:var(--ink);font-weight:600}
textarea{font:inherit;padding:11px 14px;border:1px solid var(--line);border-radius:12px;
  background:var(--surface);color:var(--ink);resize:none;flex:1;min-height:44px;
  max-height:160px;line-height:1.55}
textarea:focus{outline:none;border-color:var(--accent)}
button[disabled]{opacity:.45;cursor:default}
.copy{background:none;border:none;color:var(--muted);font-size:11px;padding:4px 2px;
  cursor:pointer;font-family:inherit}
.copy:hover{color:var(--accent)}
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
.turn.bot .drawer details{max-width:none}
.turn.bot .drawer .bubble{max-width:none;margin-top:10px}   /* the compared answer reads like the answer */
.src details.full{margin-top:6px}
.src details.full .drawer{background:var(--bg)}
.attr{max-width:min(78%,640px);margin-top:2px}
.hitline.added{border-left-style:dashed}
.hitline{font-size:13px;line-height:1.6;margin-top:6px;padding-left:10px;
  border-left:2px solid var(--accent);white-space:pre-wrap;word-break:break-word}
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


FAVICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'"
           "%3E%3Crect width='32' height='32' rx='8' fill='%230A2540'/%3E%3Cpath d='M9 22V10h2.6v5.2"
           "L16.4 10h3.2l-4.9 5.4 5.1 6.6h-3.3l-3.7-4.9-1.2 1.3V22z' fill='white'/%3E%3C/svg%3E")


def page(title: str, active: str, body: str, run: str | None, runs: list[str],
         surface: str = "app") -> HTMLResponse:
    """One shell, two surfaces.

    The product and the instrument are kept apart on screen because they are
    apart in fact: nothing a visitor does in the app produces evidence, and
    every number in the lab came from `kbeval run`. A single nav holding both
    "Q&A" and "실현율" invites the reader to assume the page they are looking at
    is feeding the table -- which would be false, and expensively so.
    """
    def link(href: str, label: str, key: str) -> str:
        q = f"?run={e(run)}" if run and surface == "lab" else ""
        return f'<a class="{"on" if active == key else ""}" href="{href}{q}">{label}</a>'

    if surface == "lab":
        brand = '<span class="brand">kbeval <span class="tag">측정</span></span>'
        nav = (link("/lab", "실현율", "overview") + link("/lab/retrieval", "검색", "retrieval")
               + link("/lab/questions", "문항", "questions"))
        right = ""
        if len(runs) > 1:
            opts = "".join(f'<option value="{e(r)}"{" selected" if r == run else ""}>'
                           f'{e(r)}</option>' for r in runs)
            right = (f'<form method="get" style="margin:0"><select name="run" '
                     f'onchange="this.form.submit()">{opts}</select></form>')
        right += '<a class="quiet" href="/">앱 →</a>'
    else:
        brand = '<span class="brand">기술지원 어시스턴트</span>'
        nav = (link("/", "홈", "home") + link("/qa", "Q&amp;A", "qa")
               + link("/bug", "버그", "bug") + link("/feature", "기능 추가", "feature"))
        right = '<a class="quiet" href="/lab">측정 →</a>'

    return HTMLResponse(f"""<!doctype html><html lang="ko"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)}</title><link rel="icon" href="{FAVICON}">
<style>{CSS}</style></head><body>
<header><div class="bar">{brand}<nav>{nav}</nav>
  <span class="right">{right}</span>
</div></header><main>{body}</main></body></html>""")


def no_runs(active: str) -> HTMLResponse:
    body = ("<h1>기록된 run이 없습니다</h1>"
            '<p class="sub">측정은 CLI에서만 실행됩니다. 이 화면은 결과를 읽기만 합니다.</p>'
            '<div class="banner"><b>uv run kbeval run</b> 로 스윕을 돌리고 '
            '<b>uv run kbeval score</b> 로 채점한 뒤 새로고침하세요.</div>')
    return page("기록 없음 · kbeval 측정", active, body, None, [], "lab")


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


def mock_chip(c: Config) -> str:
    """The app surface still has to admit what is answering it.

    A banner would shout; a chip is enough. What it must not do is disappear --
    a reader who takes a stand-in's output for a model's is being misled by the
    screen, not by the harness.
    """
    providers = {a.provider for a in c.agents.values()}
    if not providers <= {"mock", "mock_judge"}:
        return ""
    return ('<div class="pills"><span class="pill">응답 모델 <b>mock</b> — '
            '실제 모델이 아닙니다. 답변 품질을 평가하지 마세요.</span></div>')


def mock_banner(c: Config) -> str:
    providers = {a.provider for a in c.agents.values()} | {c.judge.agent.provider}
    if not providers <= {"mock", "mock_judge"}:
        return ""
    return ('<div class="banner"><b>mock 실행 결과입니다.</b> 배관이 도는지의 증거이지 '
            '데이터 품질의 증거가 아닙니다. 숫자를 pack 비교로 읽지 마세요.</div>')


# --- pages ---------------------------------------------------------------

@app.get("/lab", response_class=HTMLResponse)
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
    return page("개요 · kbeval 측정", "overview", body, rid, list_runs(c), "lab")


@app.get("/lab/retrieval", response_class=HTMLResponse)
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
    return page("검색 · kbeval 측정", "retrieval", body, rid, list_runs(c), "lab")


@app.get("/lab/questions", response_class=HTMLResponse)
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
        rows += (f"<tr><td class='k'><a href='/lab/q/{e(q.qid)}?run={e(rid)}'>{e(q.qid)}</a></td>"
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
    return page("문항 · kbeval 측정", "questions", body, rid, list_runs(c), "lab")


@app.get("/lab/q/{qid}", response_class=HTMLResponse)
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
        return page("없는 문항 · kbeval 측정", "questions",
                    f"<h1>없는 문항</h1><p class='sub'>{e(qid)}</p>",
                    rid, list_runs(c), "lab")

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
    return page(f"{q.qid} · kbeval 측정", "questions", body, rid, list_runs(c), "lab")


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
    return page("기술지원 어시스턴트", "home", body, pick_run(c, run), list_runs(c))


@app.get("/bug", response_class=HTMLResponse)
def bug(run: str | None = None) -> HTMLResponse:
    return not_yet("bug", "버그", run)


@app.get("/feature", response_class=HTMLResponse)
def feature(run: str | None = None) -> HTMLResponse:
    return not_yet("feature", "기능 추가", run)


def not_yet(key: str, label: str, run: str | None) -> HTMLResponse:
    c = cfg()
    body = f"""<h1>{e(label)}</h1>
<p class="sub">준비 중 — 이 유형의 문항이 아직 평가 세트에 없습니다.</p>
<div class="banner">빈 화면을 먼저 만들지 않습니다. 라벨된 문항 없이 답변만 보여주면
좋아 보이는지 아닌지를 판단할 근거가 없고, 그건 측정이 아니라 인상입니다.<br><br>
필요한 것: <code>scripts/make_questions.py</code>에 해당 intent 문항과 gold 앵커 추가 →
<code>data/eval/questions.jsonl</code> 재생성 →
<code>configs/base.yaml</code>의 <code>eval.intents</code>에 추가.</div>
<p class="sub"><a href="/qa">Q&amp;A로 가기 →</a></p>"""
    return page(f"{label} · 기술지원 어시스턴트", key, body, pick_run(c, run), list_runs(c))


def match_evidence(query: str, text: str, n_lines: int = 3) -> tuple[list[str], list[str]]:
    """Why BM25 picked this chunk: the query words it shares, and the lines they sit on.

    The retriever is lexical. A chunk shown with no reason reads as "the
    model's evidence"; shown with the words that matched, it reads as what it
    is -- a word overlap, which a Korean question over an English manual
    mostly finds only on the English terms. Display only: it re-derives the
    overlap with the retriever's own tokenizer and changes nothing upstream.
    """
    from .retrieval.base import tokenize
    ctoks = set(tokenize(text))
    matched = list(dict.fromkeys(w for w in tokenize(query, bigrams=False) if w in ctoks))
    if not matched:
        return [], []
    want = set(matched)
    scored = []
    for i, ln in enumerate(text.splitlines()):
        k = len(want & set(tokenize(ln, bigrams=False)))
        if k:
            scored.append((-k, i, ln.strip()[:240]))
    top = sorted(sorted(scored)[:n_lines], key=lambda t: t[1])
    return matched, [ln for _, _, ln in top]


_DOCS: dict[str, dict[str, str]] = {}


def doc_texts(c: Config) -> dict[str, str]:
    """Document text by id -- the coordinate space every chunk offset points into."""
    from .sources import load_documents
    if c.hash not in _DOCS:
        _DOCS[c.hash] = {d.doc_id: d.text for d in load_documents(c.source)}
    return _DOCS[c.hash]


def origin(line: str, raw: str) -> str:
    """'raw' if the line is in the document at this chunk's offsets, else 'added'.

    This is how a reader tells a transform's output from the source: rule 1
    keeps offsets on the original text, so anything in a chunk that is not in
    doc.text[start:end] was put there by a transform (e.g. domain_agent_enrich).
    Checked by content, not by pack name -- a label that trusted the name would
    stay "enriched" even if the transform silently did nothing.
    """
    t = line.strip()
    return "raw" if t and t in raw else "added"


def injected_sources(c: Config, rag_agent, rag, query: str) -> list[dict[str, Any]]:
    """Only what reached the model. A chunk the budget dropped is not a source."""
    docs = doc_texts(c)
    pk = rag_agent.deps.pack
    hits = {h.chunk_id: h for h in rag.retrieved}
    sources = []
    for cid in rag.injected_chunk_ids:
        ch = pk.get(cid) if pk.has(cid) else None
        text = ch.text if ch else ""
        if text.startswith("["):            # inline_meta header sits on its own line
            text = text.split("\n", 1)[-1]
        raw = docs.get(ch.doc_id, "")[ch.char_start:ch.char_end] if ch else ""
        matched, lines = match_evidence(query, text)
        sources.append({
            "chunk_id": cid,
            "heading": (ch.meta.get("heading_path") or ch.meta.get("title") or "") if ch else "",
            "score": round(hits[cid].score, 2) if cid in hits else None,
            "rank": hits[cid].rank if cid in hits else None,
            "text": text.strip(),
            "matched": matched,
            "lines": [{"text": ln, "origin": origin(ln, raw)} for ln in lines],
            "n_added": sum(origin(ln, raw) == "added" for ln in text.splitlines() if ln.strip()),
            "_raw": raw,
        })
    return sources


# The one abstention string (CLAUDE.md S6). Nothing to attribute when the model declined.
ABSTAIN_TEXT = "답변 불가"

RETRIEVER_NOTE = {
    "bm25": "검색은 단어 일치(BM25)입니다. \"겹친 단어\"가 이 문단이 뽑힌 이유의 전부입니다.",
    "dense": "검색은 다국어 임베딩 유사도입니다. 단어가 겹치지 않아도 뜻이 가까우면 뽑힙니다.",
    "hybrid": "검색은 단어 일치(BM25)와 다국어 임베딩 순위를 합친 것(RRF)입니다. "
              "\"겹친 단어\"는 그중 BM25 쪽 이유입니다.",
}


def attribute(c: Config, answer: str, sources: list[dict[str, Any]],
              n: int = 3) -> list[dict[str, Any]]:
    """Which injected lines the answer sits closest to -- an ESTIMATE, after the fact.

    The candidate model is not asked to cite: that would change the prompt,
    and the prompt is a controlled variable (rule 6, and the prompt digest in
    the config hash). So the screen does the next honest thing -- it compares
    the finished answer against the lines the model was given, and labels the
    result as an estimate. It is display-only and never reaches a trace.

    Uses the retriever's own embedder when there is one, so "close" means the
    same thing here as it does in search; plain token overlap otherwise.
    """
    from .retrieval.base import tokenize
    if not answer or answer.strip() == ABSTAIN_TEXT:
        return []
    # Short lines stay: this manual is mostly table rows ("DL MAC TP"), and
    # those are exactly what a model quotes. Repeats keep their first place.
    seen: set[str] = set()
    cands = []
    for s in sources:
        for ln in s["text"].splitlines():
            t = ln.strip()
            if len(t) >= 3 and t not in seen:
                seen.add(t)
                cands.append((s["chunk_id"], s["rank"], t, origin(t, s.get("_raw", ""))))
    if not cands:
        return []
    if c.retrieval.retriever in ("dense", "hybrid"):
        from .embed import build_embedder
        emb = build_embedder(c.retrieval.embedder)
        q = emb.embed([answer], "query")[0]
        vs = emb.embed([t for _, _, t, _ in cands], "document")
        norm = lambda v: sum(x * x for x in v) ** .5 or 1.0
        qn = norm(q)
        sims = [sum(a * b for a, b in zip(q, v)) / (qn * norm(v)) for v in vs]
        how = "embedding"
    else:
        aw = set(tokenize(answer))
        sims = [len(aw & set(tokenize(t))) / (len(set(tokenize(t))) or 1) for _, _, t, _ in cands]
        how = "overlap"
    best = sorted(range(len(cands)), key=lambda i: (-sims[i], i))[:n]
    return [{"chunk_id": cands[i][0], "rank": cands[i][1], "line": cands[i][2][:300],
             "origin": cands[i][3], "sim": round(sims[i], 3), "how": how}
            for i in best if sims[i] > 0]


def answer_payload(c: Config, question: str, pack: str,
                   compare: str | None = None) -> dict[str, Any]:
    """One question, answered exactly the way the sweep answers it.

    Single turn, no conversation history. Carrying the thread into the prompt
    would make the demo show behaviour the harness never measures -- the
    control is one question against one controlled context, and a screen that
    quietly does something else is worse than no screen.

    `compare` answers the same question through rag on a second pack. Same
    agent, same prompt, same budget, same retriever -- only the index differs,
    which is the pack axis read one question at a time. The bare baseline stays
    regardless: pack-vs-pack says which corpus is better, not whether either
    one helped at all.
    """
    from .types import Question
    probe = Question(qid="qa-live", intent="qa", query=question, answerable=True)
    agent = c.sweep.agent[0]

    rag_agent = build_agent_for(c, agent, "rag", pack)
    rag = rag_agent.answer(probe)
    base = build_agent_for(c, agent, "bare", None).answer(probe)

    out = {
        "answer": rag.text, "error": rag.error,
        "baseline": base.text, "baseline_error": base.error,
        "agent": agent, "provider": c.agents[agent].provider,
        "pack": pack, "pack_label": pack_label(c, pack),
        "injected_tokens": rag.injected_tokens,
        "budget": c.fixed.context_token_budget,
        "candidates": len(rag.retrieved),
        "skipped": len(rag.skipped_chunk_ids),
        "sources": injected_sources(c, rag_agent, rag, question),
        "retriever": c.retrieval.retriever,
        "retriever_note": RETRIEVER_NOTE.get(c.retrieval.retriever, ""),
        "compare": None,
    }
    out["attribution"] = [] if rag.error else attribute(c, rag.text, out["sources"])
    if compare and compare != pack and compare in c.packs:
        cmp_agent = build_agent_for(c, agent, "rag", compare)
        cmp = cmp_agent.answer(probe)
        out["compare"] = {
            "answer": cmp.text, "error": cmp.error,
            "pack": compare, "pack_label": pack_label(c, compare),
            "injected_tokens": cmp.injected_tokens,
            "skipped": len(cmp.skipped_chunk_ids),
            "sources": injected_sources(c, cmp_agent, cmp, question),
        }
        out["compare"]["attribution"] = ([] if cmp.error else
                                         attribute(c, cmp.text, out["compare"]["sources"]))
    for blk in (out, out["compare"] or {}):
        for src in blk.get("sources", []):
            src.pop("_raw", None)            # used for tagging; not worth shipping
    return out


@app.exception_handler(404)
async def not_found(request, exc) -> HTMLResponse:
    """A JSON stack trace is not a page. The app owns its own dead ends."""
    body = ("<h1>없는 페이지</h1>"
            '<p class="sub">주소를 다시 확인해 주세요.</p>'
            '<p class="sub"><a href="/">홈으로 →</a></p>')
    r = page("없는 페이지 · 기술지원 어시스턴트", "", body, None, [])
    return HTMLResponse(r.body, status_code=404)


@app.get("/api/ask")
def api_ask(q: str, pack: str = "", compare: str = "") -> dict[str, Any]:
    """The chat's only backend call. Reads packs, writes nothing."""
    c = cfg()
    pack = pack if pack in c.packs else next(iter(c.sweep.pack))
    if not q.strip():
        return {"answer": "", "error": "empty_question", "sources": []}
    return answer_payload(c, q.strip(), pack, compare or None)


CHAT_JS = """
const KEY = 'kbeval.thread';
const thread = document.getElementById('thread');
const form = document.getElementById('composer');
const input = document.getElementById('q');
const packSel = document.getElementById('pack');
const cmpSel = document.getElementById('compare');
const empty = document.getElementById('empty');
const send = document.getElementById('send');
let turns = [];
let pending = false;

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
function drawer(summaryText, build, cls) {
  const d = el('details', cls);
  const sum = el('summary', null, summaryText);
  d.appendChild(sum);
  const box = el('div', 'drawer');
  build(box);
  // A long drawer pushes its own summary off screen; a close at the bottom
  // keeps it closable from where the reader actually is.
  const shut = el('button', 'copy', '▴ 접기');
  shut.type = 'button';
  shut.addEventListener('click', () => {
    d.open = false;
    sum.scrollIntoView({ block: 'nearest' });
  });
  box.appendChild(shut);
  d.appendChild(box);
  return d;
}
function attribution(box, att, pack) {
  if (!att || !att.length) return;
  const wrap = el('div', 'attr');
  wrap.appendChild(el('div', 'meta',
    `${pack} 답변과 가장 가까운 근거 줄 · 사후 추정 (모델이 밝힌 인용이 아닙니다)`));
  att.forEach(x => {
    wrap.appendChild(el('div', 'hitline' + (x.origin === 'added' ? ' added' : ''), x.line));
    wrap.appendChild(el('div', 'meta',
      `${ORIGIN[x.origin] || ''} · #${x.rank ?? '-'} ${x.chunk_id} · 유사도 ${x.sim}`));
  });
  box.appendChild(wrap);
}
let NOTE = '';
const ORIGIN = { raw: '원문', added: '확장으로 추가된 줄' };
function cards(box, sources) {
  box.appendChild(el('div', 'meta',
    '검색 순위순입니다. ' + NOTE + ' 모델이 실제로 어느 문단을 썼는지는 모델이 말하지 않으므로, '
    + '답변 아래의 "가장 가까운 근거 줄"은 사후 추정입니다.'));
  sources.forEach((s, i) => {
    const h = el('div', 'head');
    h.appendChild(el('span', null, String(s.rank ?? i + 1)));
    h.appendChild(el('b', null, s.heading || s.chunk_id));
    h.appendChild(el('span', null, s.chunk_id));
    if (s.score !== null) h.appendChild(el('span', null, 'score ' + s.score));
    const card = el('div', 'src');
    card.appendChild(h);
    const m = s.matched || [];
    card.appendChild(el('div', 'meta', m.length ? ('겹친 단어 · ' + m.join(', ')) : '겹친 단어 없음 (예산이 남아 채워진 문단)'));
    if (s.n_added) card.appendChild(el('div', 'meta', `이 문단에 원문에 없는 줄 ${s.n_added}개 (확장으로 추가됨)`));
    (s.lines || []).forEach(ln => {
      const t = typeof ln === 'string' ? { text: ln, origin: 'raw' } : ln;   // turns saved before tagging
      card.appendChild(el('div', 'hitline' + (t.origin === 'added' ? ' added' : ''), t.text));
    });
    card.appendChild(drawer('문단 전체', b => b.appendChild(el('div', 'body', s.text)), 'full'));
    box.appendChild(card);
  });
}
function render(question, a) {
  const me = el('div', 'turn me');
  me.appendChild(el('div', 'bubble', question));
  thread.appendChild(me);

  const bot = el('div', 'turn bot');
  const body = a.error ? ('오류 · ' + a.error) : a.answer;
  bot.appendChild(el('div', 'bubble' + (a.error ? ' err' : ''), body));
  const meta = el('div', 'meta',
    `${a.agent} (${a.provider}) · ${a.pack_label} · 주입 ${a.sources.length}청크 `
    + `${a.injected_tokens}/${a.budget} tok`
    + (a.skipped ? ` · 예산 초과 ${a.skipped}건 제외` : '')
    + ` · 검색 후보 ${a.candidates}건 · `);
  const copy = el('button', 'copy', '답변 복사');
  copy.addEventListener('click', () => {
    navigator.clipboard.writeText(body).then(
      () => { copy.textContent = '복사됨'; setTimeout(() => copy.textContent = '답변 복사', 1400); },
      () => { copy.textContent = '복사할 수 없습니다'; });
  });
  meta.appendChild(copy);
  bot.appendChild(meta);

  // Order = reading order: the answer, then what it is compared against, and
  // the evidence last. Answers first because the comparison IS the point;
  // evidence last because it is what you open when the comparison surprises you.
  const c = a.compare;
  if (c) {
    bot.appendChild(drawer(`${c.pack} pack으로 같은 질문을 하면`, box => {
      box.appendChild(el('div', 'bubble' + (c.error ? ' err' : ''), c.error ? ('오류 · ' + c.error) : c.answer));
      box.appendChild(el('div', 'meta',
        `${c.pack_label} · 주입 ${c.sources.length}청크 ${c.injected_tokens}/${a.budget} tok`
        + (c.skipped ? ` · 예산 초과 ${c.skipped}건 제외` : '')));
      box.appendChild(el('div', 'meta',
        '같은 모델·프롬프트·예산·검색기에서 인덱스만 바꾼 답변입니다. 이 차이가 pack 효과입니다. '
        + '근거는 맨 아래에 있습니다.'));
    }));
  }

  bot.appendChild(drawer('지식 주입 없이 같은 질문을 하면', box => {
    box.appendChild(el('div', 'base',
      a.baseline_error ? ('오류 · ' + a.baseline_error) : a.baseline));
    box.appendChild(el('div', 'meta',
      '같은 모델, 같은 프롬프트, 자료만 뺀 baseline입니다. 이 차이가 측정 대상입니다.'));
  }));

  NOTE = a.retriever_note || '';
  attribution(bot, a.attribution, a.pack);
  // The main pack's passage list is not rendered (user choice, 2026-09-23):
  // its closest lines are shown above, and the payload still carries every
  // injected chunk in a.sources, so the evidence is one API call away.
  if (!a.sources.length) {
    bot.appendChild(el('div', 'meta', '검색된 문단이 없습니다 — 질문의 어휘가 코퍼스와 겹치지 않습니다.'));
  }
  if (c && c.sources.length) {
    bot.appendChild(drawer(`${c.pack} 근거 · 주입된 문단 ${c.sources.length}개`, box => {
      attribution(box, c.attribution, c.pack);
      cards(box, c.sources);
    }));
  }
  thread.appendChild(bot);
}
function busy(on) {
  pending = on;
  send.disabled = on;
  send.textContent = on ? '검색 중' : '전송';
}
async function ask(question) {
  if (pending) return;
  busy(true);
  empty.hidden = true;
  const me = el('div', 'turn me');
  me.appendChild(el('div', 'bubble', question));
  thread.appendChild(me);
  const wait = el('div', 'thinking', '문서를 검색하는 중…');
  thread.appendChild(wait);
  window.scrollTo(0, document.body.scrollHeight);

  try {
    const r = await fetch('/api/ask?q=' + encodeURIComponent(question)
                          + '&pack=' + encodeURIComponent(packSel.value)
                          + '&compare=' + encodeURIComponent(cmpSel.value));
    const a = await r.json();
    wait.remove();
    me.remove();
    render(question, a);
    turns.push({ q: question, a: a });
    store();
  } catch (err) {
    wait.textContent = '요청이 실패했습니다 · ' + err;
  }
  busy(false);
  window.scrollTo(0, document.body.scrollHeight);
}
function submit() {
  const v = input.value.trim();
  if (!v || pending) return;
  input.value = '';
  input.style.height = 'auto';
  ask(v);
}
form.addEventListener('submit', ev => { ev.preventDefault(); submit(); });
input.addEventListener('keydown', ev => {
  // Enter sends, Shift+Enter breaks the line -- what a chat box is expected
  // to do; a textarea that only submits on a button feels like a form.
  if (ev.key === 'Enter' && !ev.shiftKey && !ev.isComposing) {
    ev.preventDefault();
    submit();
  }
});
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 160) + 'px';
});
document.querySelectorAll('[data-sample]').forEach(b =>
  b.addEventListener('click', ev => { ev.preventDefault(); ask(b.dataset.sample); }));
document.getElementById('clear').addEventListener('click', () => {
  turns = []; store(); thread.replaceChildren(); empty.hidden = false;
});
restore();
"""


@app.get("/qa", response_class=HTMLResponse)
def qa(pack: str = "", compare: str = "", run: str | None = None) -> HTMLResponse:
    c = cfg()
    rid = pick_run(c, run)
    pack = pack if pack in c.packs else next(iter(c.sweep.pack))
    compare = compare if compare in c.packs else ""

    opts = "".join(
        f'<option value="{e(p)}"{" selected" if p == pack else ""}>{e(pack_label(c, p))}</option>'
        for p in c.sweep.pack)
    cmp_opts = '<option value="">비교 없음</option>' + "".join(
        f'<option value="{e(p)}"{" selected" if p == compare else ""}>비교: {e(p)}</option>'
        for p in c.sweep.pack)
    qs = load_questions(c.eval.questions, c.eval.intents)
    samples = "".join(
        f'<button type="button" class="chip" data-sample="{html.escape(x.query, quote=True)}">'
        f'{e(x.query[:30])}…</button>'
        for x in qs[:3])
    # What the chat answers from comes from the config, not from this file --
    # a hardcoded corpus name outlives the corpus and tells the reader the wrong
    # thing about what the model is looking at.
    docs = ", ".join(sorted({ch.doc_id for ch in pack_for(c, pack).chunks}))
    first = qs[0] if qs else None
    cli = (f"kbeval ask {first.qid} --mode rag --pack {pack}" if first
           else f"kbeval ask <qid> --mode rag --pack {pack}")

    twins = identical_packs(c)
    twin_note = ""
    if twins:
        pairs = " / ".join(" = ".join(g) for g in twins)
        twin_note = (f'<br><b>{e(pairs)}</b> 는 현재 문서에서 청크가 완전히 동일합니다. '
                     f'선택을 바꿔도 답이 같은 것은 고장이 아니라, 그 전처리 단계가 이 '
                     f'문서들에 대해 할 일이 없다는 뜻입니다 — 중복이 없어 <code>dedup</code>이, '
                     f'표기가 이미 일관돼 <code>normalize_terms</code>가 바꿀 것을 '
                     f'찾지 못합니다.')

    body = f"""<h1>기술지원 Q&amp;A</h1>
<p class="sub"><b>{e(docs)}</b> 를 근거로 답합니다. 답변마다 근거 문단을 펼쳐 볼 수 있습니다.</p>
{mock_chip(c)}

<noscript><div class="banner">이 화면은 자바스크립트가 필요합니다.
같은 일을 터미널에서 하려면 <code>{e(cli)}</code>.</div></noscript>

<div class="chat" id="thread"></div>
<div class="empty-chat" id="empty">무엇이든 물어보세요.<div class="chips"
  style="justify-content:center">{samples}</div></div>

<div class="composer"><form id="composer" autocomplete="off">
<textarea id="q" rows="1" autofocus
  placeholder="질문을 입력하세요"></textarea>
<select id="pack" title="지식베이스 준비 수준">{opts}</select>
<select id="compare" title="같은 질문을 다른 pack으로도 답해 나란히 봅니다">{cmp_opts}</select>
<button type="submit" id="send">전송</button>
<button type="button" class="ghost" id="clear">초기화</button>
</form></div>

<div class="footnote">
<b>이 대화는 저장되지 않습니다.</b> 질문은 <code>runs/</code>에 기록되지 않고 run id도
만들지 않습니다. 실험 기록은 <code>kbeval run</code> 한 경로뿐이고, 그 결과는
<a href="/lab">측정</a>에서 봅니다.<br>
<b>각 질문은 독립적으로 답합니다.</b> 앞의 대화를 컨텍스트에 넣지 않습니다 —
그래야 여기서 보이는 동작이 실제로 측정하는 동작과 같습니다.
{twin_note}</div>
<script>{CHAT_JS}</script>"""
    return page("Q&A · 기술지원 어시스턴트", "qa", body, rid, list_runs(c))
