"""The report: differences, not absolute scores.

An absolute correctness number mixes "this model already knew telecom" with
"this data was good", and no amount of averaging separates them afterwards.
Only the three-way difference does:

    headroom    = oracle_ref - bare      what knowledge was there to gain
    lift        = rag        - bare      what this pack actually delivered
    realization = lift / headroom        the primary observable

Four rules are enforced here rather than left to the reader:

* headroom below `scoring.min_headroom` -> realization is NOT computed. A tiny
  denominator turns noise into a spectacular-looking ratio.
* a NEGATIVE realization is kept and shown. Context that confuses a 30B model
  is common and is itself the finding; dropping it would make every pack look
  non-harmful.
* unanswerable questions never enter realization. They are scored on
  abstention, in their own section.
* covariates sit next to every number. A pack with more, smaller chunks is
  easier to hit and cheaper to inject, and without n_chunks and
  avg_chunk_chars beside it a hit@k difference is unreadable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from ..config import Config
from ..packs import build_pack
from ..runner import RunPaths, read_traces
from ..sweep import PACK_FREE_MODES
from .scoring import load_scores


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(fmean(vals), 4) if vals else None


@dataclass
class RealizationRow:
    agent: str
    pack: str
    bare: float | None
    rag: float | None
    oracle: float | None
    headroom: float | None
    lift: float | None
    realization: float | None
    n: int
    note: str = ""


@dataclass
class Report:
    run_id: str
    config_hash: str
    n_traces: int
    malformed: int
    errors: int
    format_fails: int
    realization: list[RealizationRow] = field(default_factory=list)
    retrieval: list[dict[str, Any]] = field(default_factory=list)
    abstention: list[dict[str, Any]] = field(default_factory=list)
    tau: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _correctness_by_qid(rows: list[dict], agent: str, mode: str,
                        pack: str | None, answerable: bool) -> dict[str, float]:
    out = {}
    for r in rows:
        if (r["agent"], r["mode"], r["pack"]) != (agent, mode, pack):
            continue
        if r["answerable"] is not answerable:
            continue
        verdict = r.get("judge") or {}
        if verdict.get("skipped") or verdict.get("format_fail") == 1.0:
            continue            # an error or a parse failure is not a wrong answer
        if verdict.get("correctness") is not None:
            out[r["qid"]] = float(verdict["correctness"])
    return out


def _paired_p(a: dict[str, float], b: dict[str, float]) -> float | None:
    """Wilcoxon on the questions both arms answered.

    Paired by question on purpose: between-model variance is larger than the
    pack effect, so an unpaired test on this design measures mostly which
    models were in the sweep.
    """
    shared = sorted(set(a) & set(b))
    diffs = [a[q] - b[q] for q in shared]
    if len(diffs) < 5 or not any(diffs):
        return None
    from scipy.stats import wilcoxon
    try:
        return round(float(wilcoxon(diffs).pvalue), 4)
    except ValueError:
        return None


def build_report(cfg: Config, root: Path) -> Report:
    traces, malformed = read_traces(RunPaths(root).traces)
    scores = load_scores(root)

    rep = Report(
        run_id=cfg.run_id, config_hash=cfg.hash, n_traces=len(traces),
        malformed=malformed,
        errors=sum(1 for t in traces if t.get("error")),
        format_fails=sum(1 for s in scores if (s.get("judge") or {}).get("format_fail") == 1.0),
    )
    if not scores:
        rep.notes.append("no scores found -- run `kbeval score` first")
        return rep

    # --- realization -----------------------------------------------------
    for agent in cfg.sweep.agent:
        bare = _correctness_by_qid(scores, agent, "bare", None, answerable=True)
        oracle = _correctness_by_qid(scores, agent, "oracle_ref", None, answerable=True)
        b, o = _mean(list(bare.values())), _mean(list(oracle.values()))
        headroom = None if (b is None or o is None) else round(o - b, 4)

        for pack in cfg.sweep.pack:
            rag = _correctness_by_qid(scores, agent, "rag", pack, answerable=True)
            r = _mean(list(rag.values()))
            lift = None if (r is None or b is None) else round(r - b, 4)

            note, realization = "", None
            if headroom is None or lift is None:
                note = "missing arm"
            elif headroom < cfg.scoring.min_headroom:
                note = f"headroom<{cfg.scoring.min_headroom} -- denominator is noise"
            else:
                realization = round(lift / headroom, 4)
                if realization < 0:
                    note = "negative: context made it worse (kept on purpose)"
                elif realization > 1:
                    # The pack beat the curated reference. Usually the reference
                    # is one paragraph while retrieval hands over the whole
                    # section -- a real result about the ceiling, not an error.
                    note = "above 1: rag beat oracle_ref — check the reference is complete"

            rep.realization.append(RealizationRow(
                agent=agent, pack=pack, bare=b, rag=r, oracle=o,
                headroom=headroom, lift=lift, realization=realization,
                n=len(rag), note=note))

    # --- retrieval + covariates ------------------------------------------
    for pack in cfg.sweep.pack:
        manifest = build_pack(cfg, pack).manifest
        rows = [s for s in scores if s["pack"] == pack and s.get("retrieval")]
        rag_traces = [t for t in traces if t["pack"] == pack and t["mode"] == "rag"]
        entry = {"pack": pack,
                 "n_chunks": manifest["n_chunks"],
                 "avg_chunk_chars": manifest["avg_chunk_chars"],
                 "injected_tokens": _mean([t["injected_tokens"] for t in rag_traces]),
                 "injected_chunks": _mean([len(t["injected_chunk_ids"]) for t in rag_traces]),
                 "n": len(rows)}
        for k in cfg.scoring.ks:
            entry[f"hit@{k}"] = _mean([s["retrieval"].get(f"hit@{k}") for s in rows])
        entry["mrr"] = _mean([s["retrieval"].get("mrr") for s in rows])
        entry["gold_injected"] = _mean([s["retrieval"].get("gold_injected") for s in rows])
        entry["gold_lost_to_budget"] = _mean(
            [s["retrieval"].get("gold_lost_to_budget") for s in rows])
        rep.retrieval.append(entry)

    # --- abstention (unanswerable only) ----------------------------------
    for agent in cfg.sweep.agent:
        for mode in ("bare", "rag", "oracle_ref"):
            packs = [None] if mode in PACK_FREE_MODES else cfg.sweep.pack
            for pack in packs:
                rows = [s for s in scores
                        if (s["agent"], s["mode"], s["pack"]) == (agent, mode, pack)
                        and not s["answerable"]]
                verdicts = [s.get("judge") or {} for s in rows]
                verdicts = [v for v in verdicts if not v.get("skipped")]
                if not verdicts:
                    continue
                rep.abstention.append({
                    "agent": agent, "mode": mode, "pack": pack, "n": len(verdicts),
                    "abstention": _mean([v.get("abstention") for v in verdicts]),
                    "harmful_action": _mean([v.get("harmful_action") for v in verdicts]),
                })

    # --- did the axis move at all? ---------------------------------------
    per_agent = {}
    for row in rep.realization:
        per_agent.setdefault(row.agent, set()).add(row.rag)
    if per_agent and all(len(v) == 1 for v in per_agent.values()) and len(cfg.sweep.pack) > 1:
        rep.notes.append(
            "every pack scored identically for every model — the pack axis did not "
            "discriminate. Read this as a property of the QUESTION SET, not of the "
            "packs: if the answers are reachable and easy for all of them, there is "
            "no headroom for preprocessing to win.")
    if any(r.rag == 1.0 and r.oracle == 1.0 for r in rep.realization):
        rep.notes.append(
            "at least one arm scored a perfect 1.000 — the set is saturated there and "
            "a better pack has nothing left to prove.")

    # --- cross-model agreement -------------------------------------------
    rep.tau = _kendall(rep, cfg)

    # --- statistics -------------------------------------------------------
    for agent in cfg.sweep.agent:
        bare = _correctness_by_qid(scores, agent, "bare", None, answerable=True)
        for pack in cfg.sweep.pack:
            rag = _correctness_by_qid(scores, agent, "rag", pack, answerable=True)
            p = _paired_p(rag, bare)
            for row in rep.realization:
                if (row.agent, row.pack) == (agent, pack):
                    row.note = (row.note + f" | wilcoxon p={p}").strip(" |") if p is not None \
                        else (row.note + " | n too small for a paired test").strip(" |")
    return rep


def _kendall(rep: Report, cfg: Config) -> list[dict[str, Any]]:
    """Do the models rank the packs the same way?

    tau >= 0.7 is the transfer claim: if a Bedrock stand-in orders the packs
    the same way another model does, the ordering is a property of the data and
    can be carried to the on-prem model the experiment is actually about.
    """
    ranks = {}
    for agent in cfg.sweep.agent:
        vals = [next((r.rag for r in rep.realization
                      if r.agent == agent and r.pack == p), None) for p in cfg.sweep.pack]
        if all(v is not None for v in vals):
            ranks[agent] = vals

    out = []
    agents = sorted(ranks)
    for i, a in enumerate(agents):
        for b in agents[i + 1:]:
            if len(set(ranks[a])) == 1 or len(set(ranks[b])) == 1:
                out.append({"pair": f"{a} vs {b}", "tau": None,
                            "note": "one model ranked every pack equally -- tau undefined"})
                continue
            from scipy.stats import kendalltau
            t = kendalltau(ranks[a], ranks[b])
            out.append({"pair": f"{a} vs {b}", "tau": round(float(t.statistic), 3),
                        "p": round(float(t.pvalue), 4),
                        "note": "" if t.statistic >= 0.7 else "below 0.7 -- ranking is model-dependent"})
    return out


def _fmt(v: Any, width: int = 7, nd: int = 3) -> str:
    if v is None:
        return "-".rjust(width)
    if isinstance(v, float):
        return f"{v:>{width}.{nd}f}"
    return f"{v:>{width}}"


def render(rep: Report) -> str:
    """Plain text. The report is an instrument reading, not a document."""
    out: list[str] = []
    out.append(f"run_id : {rep.run_id}    config_hash: {rep.config_hash}")
    out.append(f"traces : {rep.n_traces} rows   errors={rep.errors}   "
               f"judge_format_fail={rep.format_fails}   unreadable_lines={rep.malformed}")
    for n in rep.notes:
        out.append(f"note   : {n}")

    out.append("\n## realization  (answerable only; lift / headroom)")
    w = max([len(r.agent) for r in rep.realization] + [5]) + 2
    out.append(f"{'agent':<{w}}{'pack':<7}{'bare':>7}{'rag':>7}{'oracle':>8}"
               f"{'headroom':>10}{'lift':>7}{'realiz':>8}{'n':>4}  note")
    for r in rep.realization:
        out.append(f"{r.agent:<{w}}{r.pack:<7}{_fmt(r.bare)}{_fmt(r.rag)}{_fmt(r.oracle, 8)}"
                   f"{_fmt(r.headroom, 10)}{_fmt(r.lift)}{_fmt(r.realization, 8)}"
                   f"{r.n:>4}  {r.note}")

    out.append("\n## retrieval + covariates  (rag rows)")
    ks = [k for k in rep.retrieval[0] if k.startswith("hit@")] if rep.retrieval else []
    out.append(f"{'pack':<7}{'chunks':>7}{'avg_chars':>10}" + "".join(f"{k:>7}" for k in ks)
               + f"{'mrr':>7}{'gold_inj':>9}{'lost_budget':>12}{'inj_tok':>9}{'inj_chunks':>11}")
    for e in rep.retrieval:
        out.append(f"{e['pack']:<7}{e['n_chunks']:>7}{e['avg_chunk_chars']:>10}"
                   + "".join(_fmt(e[k]) for k in ks)
                   + _fmt(e["mrr"]) + _fmt(e["gold_injected"], 9)
                   + _fmt(e["gold_lost_to_budget"], 12)
                   + _fmt(e["injected_tokens"], 9, 1) + _fmt(e["injected_chunks"], 11, 2))

    out.append("\n## abstention  (unanswerable only -- excluded from realization)")
    out.append(f"{'agent':<9}{'mode':<12}{'pack':<7}{'n':>4}{'abstain':>9}{'harmful':>9}")
    for a in rep.abstention:
        out.append(f"{a['agent']:<9}{a['mode']:<12}{a['pack'] or '-'!s:<7}{a['n']:>4}"
                   f"{_fmt(a['abstention'], 9)}{_fmt(a['harmful_action'], 9)}")

    out.append("\n## kendall tau  (do the models rank the packs the same way?)")
    tw = max([len(t["pair"]) for t in rep.tau] + [10]) + 2
    for t in rep.tau:
        out.append(f"{t['pair']:<{tw}}tau={_fmt(t.get('tau'))}  {t.get('note', '')}")

    return "\n".join(out)
