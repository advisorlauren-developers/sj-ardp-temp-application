from __future__ import annotations

from pathlib import Path

import typer

from .config import load_config
from .eval import build_report, render, score_run
from .packs import build_chunker, build_pack, load_questions
from .retrieval import DEFAULT_KS, build_retriever, summarize
from .runner import run_sweep
from .sources import load_documents
from .sweep import expand_sweep
from .wiring import build as build_agent_for

app = typer.Typer(add_completion=False, help="AI-ready dataset downstream eval harness")

CFG = typer.Option(["configs/base.yaml"], "--config", "-c", help="layered config files")
SET = typer.Option(None, "--set", "-s", help="override, e.g. -s fixed.n_repeat=3")


@app.callback()
def main() -> None:
    """Forces Typer into subcommand mode."""


@app.command()
def plan(config: list[str] = CFG, set_: list[str] = SET) -> None:
    """Resolve the config and show the sweep. Spends no tokens.

    Run this before every experiment: it is the cheapest place to catch a typo,
    an undeclared model, or a sweep that is an order of magnitude larger than
    intended.
    """
    cfg = load_config(config, set_)
    specs = expand_sweep(cfg)
    typer.echo(f"run_id      : {cfg.run_id}")
    typer.echo(f"config_hash : {cfg.hash}")
    typer.echo(f"specs       : {len(specs)}"
               f"   (cross product would be "
               f"{len(cfg.sweep.agent) * len(cfg.sweep.mode) * len(cfg.sweep.pack)})")
    for s in specs:
        typer.echo(f"  {s.key}")


@app.command()
def docs(config: list[str] = CFG, set_: list[str] = SET, show: int = 0) -> None:
    """List the documents the source adapter produced.

    Worth running whenever the source changes: `chars` is the size of the
    coordinate space that every gold span will index into, so a document that
    silently arrives empty or half-parsed shows up here rather than three
    stages later as a mysterious retrieval failure.
    """
    cfg = load_config(config, set_)
    ds = load_documents(cfg.source)
    typer.echo(f"adapter={cfg.source.adapter}  path={cfg.source.path}  docs={len(ds)}")
    for d in ds:
        typer.echo(f"  {d.doc_id:<20} chars={len(d.text):>6}  title={d.title}")
        typer.echo(f"  {'':<20} meta={d.meta}")
    if show:
        typer.echo(f"\n--- {ds[0].doc_id} first {show} chars (offset 0 starts here) ---")
        typer.echo(repr(ds[0].text[:show]))


@app.command()
def chunks(pack: str, config: list[str] = CFG, set_: list[str] = SET,
           show: int = 0) -> None:
    """Chunk the corpus with one pack's chunker and check the offset invariant.

    The invariant -- doc.text[start:end].strip() == chunk.text -- is what makes
    a gold span labelled once valid for every pack. It cannot be eyeballed, so
    it is asserted here on every chunk before anything downstream trusts it.
    """
    cfg = load_config(config, set_)
    pc = cfg.packs[pack]
    ch = build_chunker(pc.chunker, pc.chunk_chars, pc.chunk_overlap)
    docs = {d.doc_id: d for d in load_documents(cfg.source)}

    out = [c for d in docs.values() for c in ch.split(d)]
    bad = [c.chunk_id for c in out
           if docs[c.doc_id].text[c.char_start:c.char_end].strip() != c.text]

    lens = [len(c.text) for c in out] or [0]
    typer.echo(f"pack={pack}  chunker={pc.chunker}  size={pc.chunk_chars} "
               f"overlap={pc.chunk_overlap}")
    typer.echo(f"chunks={len(out)}  avg_chars={sum(lens) / len(lens):.0f}  "
               f"max_chars={max(lens)}")
    typer.echo("offset invariant: " + ("OK" if not bad else f"BROKEN {bad}"))
    for c in out[:show]:
        head = c.meta.get("heading_path", "-")
        typer.echo(f"  {c.chunk_id:<20} [{c.char_start:>4}:{c.char_end:>4}] "
                   f"{len(c.text):>4}c  {head}")
        typer.echo(f"  {'':<20} {c.text[:60]!r}")
    if bad:
        raise typer.Exit(1)


@app.command()
def pack(name: str, config: list[str] = CFG, set_: list[str] = SET,
         show: int = 0) -> None:
    """Build one pack and verify that gold survived its transforms.

    Coverage is the number to watch. A transform that deletes the only chunk
    holding an answer lowers that pack's ceiling, and in the final report that
    is indistinguishable from the model getting worse -- unless it was caught
    here first.
    """
    cfg = load_config(config, set_)
    pk = build_pack(cfg, name)
    qs = load_questions(cfg.eval.questions, cfg.eval.intents)
    cov = pk.verify_gold_coverage(qs, cfg.eval.gold_overlap_ratio)

    m = pk.manifest
    typer.echo(f"pack={name}  chunker={m['chunker']}  transforms={m['transforms']}")
    typer.echo(f"  chunks={m['n_chunks']}  avg_chars={m['avg_chunk_chars']}  "
               f"max_chars={m['max_chunk_chars']}  total_chars={m['total_chars']}")
    typer.echo(f"  coverage={cov['coverage']}  "
               f"answerable={cov['n_answerable']}  "
               f"avg_gold_chunks={cov['avg_gold_chunks']}")
    typer.echo(f"  saved -> {pk.dir}/")

    for q in qs[:show]:
        ids = pk.gold_chunk_ids(q, cfg.eval.gold_overlap_ratio)
        typer.echo(f"  {q.qid}  gold={ids or '-'}")
    if show:
        c = pk.chunks[min(1, len(pk.chunks) - 1)]
        typer.echo(f"\n--- sample chunk {c.chunk_id} [{c.char_start}:{c.char_end}] ---")
        typer.echo(c.text[:280])

    if cov["missing_gold"]:
        typer.secho(f"  WARNING gold missing for {cov['missing_gold']}", fg="red")
        raise typer.Exit(1)


@app.command()
def search(query: str, pack: str = "mid", k: int = 0, qid: str = "",
           config: list[str] = CFG, set_: list[str] = SET) -> None:
    """Rank one pack's chunks against one query. Spends no tokens.

    Read this before blaming a model. If the gold chunk is not in the list, the
    answer was never reachable and no prompt engineering downstream will
    recover it -- the finding belongs to the data, not the generator.
    """
    cfg = load_config(config, set_)
    k = k or cfg.retrieval.candidate_k
    pk = build_pack(cfg, pack)
    r = build_retriever(cfg.retrieval)
    r.index(pk.chunks)

    qs = load_questions(cfg.eval.questions, cfg.eval.intents)
    q = next((x for x in qs if x.qid == qid or x.query == query), None)
    gold = set(pk.gold_chunk_ids(q, cfg.eval.gold_overlap_ratio)) if q else set()
    if q and qid:
        query = q.query

    typer.echo(f"pack={pack}  retriever={cfg.retrieval.retriever}  k={k}  "
               f"chunks={len(pk.chunks)}")
    typer.echo(f"query : {query}")
    typer.echo(f"gold  : {sorted(gold) or '- (no labelled question matched)'}")
    for hit in r.search(query, k):
        mark = "*" if hit.chunk_id in gold else " "
        body = pk.get(hit.chunk_id).text.replace("\n", " ")[:70]
        typer.echo(f" {mark}{hit.rank:>2}. {hit.score:>7.3f}  {hit.chunk_id:<18} {body}")


@app.command("retrieval-eval")
def retrieval_eval(config: list[str] = CFG, set_: list[str] = SET,
                   detail: bool = False) -> None:
    """hit@k and MRR for every pack. The whole stage costs zero tokens.

    The covariates are printed alongside on purpose: a pack with more, smaller
    chunks has an easier time being hit at all, so a hit@k difference is only
    readable next to n_chunks and avg_chunk_chars.

    Unanswerable questions are excluded -- they have no gold and are an
    abstention test, not a retrieval one.
    """
    cfg = load_config(config, set_)
    qs = [q for q in load_questions(cfg.eval.questions, cfg.eval.intents)
          if q.answerable and q.gold is not None]
    depth = max(DEFAULT_KS)

    head = "  ".join(f"hit@{k}" for k in DEFAULT_KS)
    typer.echo(f"retriever={cfg.retrieval.retriever}  bigrams={cfg.retrieval.tokenizer.bigrams}"
               f"  questions={len(qs)}  depth={depth}")
    typer.echo(f"{'pack':<6}{'chunks':>7}{'avg_chars':>11}   {head}   {'mrr':>5}")

    for name in cfg.sweep.pack:
        pk = build_pack(cfg, name)
        r = build_retriever(cfg.retrieval)
        r.index(pk.chunks)

        rows, ranks = [], {}
        for q in qs:
            ranked = [h.chunk_id for h in r.search(q.query, depth)]
            gold = pk.gold_chunk_ids(q, cfg.eval.gold_overlap_ratio)
            rows.append((ranked, gold))
            ranks[q.qid] = next((i for i, c in enumerate(ranked, 1) if c in set(gold)), None)

        m = summarize(rows)
        cells = "  ".join(f"{m[f'hit@{k}']:>5.3f}" for k in DEFAULT_KS)
        typer.echo(f"{name:<6}{pk.manifest['n_chunks']:>7}"
                   f"{pk.manifest['avg_chunk_chars']:>11}   {cells}   {m['mrr']:>5.3f}")
        if detail:
            for q in qs:
                pos = ranks[q.qid]
                typer.echo(f"       {q.qid}  gold_rank={pos if pos else 'MISS':<5} {q.query[:44]}")


@app.command()
def ask(qid: str, agent: str = "mock_a", mode: str = "rag", pack: str = "mid",
        show_prompt: bool = False, config: list[str] = CFG, set_: list[str] = SET) -> None:
    """Answer ONE question with one (agent, mode, pack) spec.

    The debugging surface for S6: it prints what went into the prompt, not just
    what came out. `injected` vs `skipped` is the token budget doing its job --
    if skipped is never empty the control variable is binding, and if injected
    is empty the pack never reached the model at all.
    """
    cfg = load_config(config, set_)
    qs = {q.qid: q for q in load_questions(cfg.eval.questions, cfg.eval.intents)}
    if qid not in qs:
        typer.secho(f"unknown qid '{qid}'. available: {sorted(qs)}", fg="red")
        raise typer.Exit(1)
    q = qs[qid]

    a = build_agent_for(cfg, agent, mode, None if mode in ("bare", "oracle_ref") else pack)
    user, _ = a.render(q)
    res = a.answer(q)

    typer.echo(f"spec    : {agent} | {mode} | {pack if mode == 'rag' else '-'}"
               f"   config_hash={cfg.hash}")
    typer.echo(f"question: [{q.qid}] answerable={q.answerable}  {q.query}")
    typer.echo(f"injected: {len(res.injected_chunk_ids)} chunks / "
               f"{res.injected_tokens} tok of {cfg.fixed.context_token_budget}"
               f"   skipped={len(res.skipped_chunk_ids)}")
    if res.injected_chunk_ids:
        typer.echo(f"          {res.injected_chunk_ids}")
    if show_prompt:
        typer.echo("\n--- system ---\n" + a.system.strip())
        typer.echo("\n--- user ---\n" + user.strip())

    if res.error:
        typer.secho(f"error   : {res.error}   (NOT counted as an answer)", fg="red")
        raise typer.Exit(1)
    typer.echo(f"thinking: {res.usage.get('had_thinking')}   latency={res.latency_ms:.0f}ms")
    typer.echo(f"answer  : {res.text}")


@app.command()
def run(limit: int = 0, resume: bool = True, config: list[str] = CFG,
        set_: list[str] = SET) -> None:
    """Execute the sweep. Append-only, resumable, and gated on gold coverage.

    Rerunning the same command after a failure is the supported recovery path:
    finished (spec, question, repeat) units are read back from traces.jsonl and
    skipped. Start with `--limit 1` against a new model -- one question per spec
    is enough to catch a wrong model id, and it costs almost nothing.
    """
    cfg = load_config(config, set_)

    def progress(done: int, total: int) -> None:
        if total and (done % 10 == 0 or done == total):
            typer.echo(f"  ... {done}/{total}", err=True)

    try:
        s = run_sweep(cfg, limit=limit or None, resume=resume, on_progress=progress)
    except RuntimeError as e:
        typer.secho(str(e), fg="red")
        raise typer.Exit(1) from e

    typer.echo(f"run_id  : {s.run_id}")
    typer.echo(f"specs   : {s.specs}   planned={s.planned}  "
               f"resumed={s.resumed}  written={s.written}")
    typer.echo(f"errors  : {s.errors}" + (f"   {s.error_samples}" if s.error_samples else ""))
    typer.echo(f"elapsed : {s.elapsed_s}s")
    typer.echo(f"traces  : {s.root}/traces.jsonl")


def _run_root(cfg, run_id: str) -> Path:
    root = Path(cfg.run.out_dir) / (run_id or cfg.run_id)
    if not root.exists():
        typer.secho(f"no such run: {root}", fg="red")
        raise typer.Exit(1)
    return root


@app.command()
def score(run_id: str = "", resume: bool = True, config: list[str] = CFG,
          set_: list[str] = SET) -> None:
    """Score a finished run. Re-runnable without paying for candidates again.

    Scores are appended to scores.jsonl rather than written back into
    traces.jsonl: traces are evidence, and evidence is not edited in place.
    """
    cfg = load_config(config, set_)
    s = score_run(cfg, _run_root(cfg, run_id), resume=resume)
    typer.echo(f"rows    : {s.rows}   scored={s.scored}  resumed={s.resumed}")
    typer.echo(f"skipped : {s.skipped_errors} (provider errors -- NOT counted as wrong answers)")
    typer.echo(f"format  : {s.format_fails} judge parse failure(s)")
    for n in s.notes:
        typer.secho(f"note    : {n}", fg="yellow")
    typer.echo(f"scores  : {_run_root(cfg, run_id)}/scores.jsonl")


@app.command()
def report(run_id: str = "", config: list[str] = CFG, set_: list[str] = SET) -> None:
    """Differences, covariates and health for one run."""
    cfg = load_config(config, set_)
    typer.echo(render(build_report(cfg, _run_root(cfg, run_id))))


@app.command()
def web(port: int = 8000, host: str = "127.0.0.1", reload: bool = False,
        config: list[str] = CFG) -> None:
    """Serve the read-only viewer.

    It reads finished runs and never writes to runs/. Measurement stays on one
    path -- this command starts a window onto the evidence, not a second way to
    produce it.
    """
    import os

    import uvicorn

    os.environ["KBEVAL_CONFIG"] = ",".join(config)
    typer.echo(f"http://{host}:{port}   (read-only; config={','.join(config)})")
    uvicorn.run("kbeval.web:app", host=host, port=port, reload=reload, log_level="warning")


if __name__ == "__main__":
    app()