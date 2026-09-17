"""Scoring a run: same append-only, resumable discipline as the run itself.

Scores live in their own file rather than being written back into traces.jsonl.
Traces are evidence and evidence is not edited in place -- and keeping them
apart is what lets a judge prompt be revised and the whole run re-scored
without paying for one candidate call again.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config, dump_resolved, load_config
from ..packs import load_questions
from ..runner import RunPaths, read_traces
from .scorers import build_scorer


@dataclass
class ScoreSummary:
    root: Path
    rows: int
    malformed_traces: int
    scored: int
    resumed: int
    skipped_errors: int       # the candidate never answered -- nothing to grade
    judge_failures: int       # the answer exists but the judge call failed
    format_fails: int
    notes: list[str] = field(default_factory=list)


def scores_path(root: Path) -> Path:
    return root / "scores.jsonl"


def scoring_config_path(root: Path) -> Path:
    return root / "scoring.resolved.yaml"


def check_judge(cfg: Config, root: Path, resume: bool) -> None:
    """Refuse to mix verdicts from two different judges in one file.

    The judge is a control variable, so `kbeval score` records exactly which
    one produced these rows. Resuming with a different judge would leave a
    scores.jsonl that is part one model and part another, and no column would
    say which -- a difference between packs could then be a difference between
    graders.
    """
    path = scoring_config_path(root)
    if resume and path.exists() and scores_path(root).exists():
        previous = load_config([path])
        if previous.scoring_hash != cfg.scoring_hash:
            raise RuntimeError(
                f"these scores were graded under scoring config "
                f"{previous.scoring_hash} (judge={previous.judge.agent.id}), not "
                f"{cfg.scoring_hash} (judge={cfg.judge.agent.id}). "
                f"Re-score from scratch with --no-resume.")


RETRYABLE = {"judge_error"}


def completed(path: Path) -> set[tuple[str, str, int]]:
    """Rows that need no further work.

    A row whose judge call FAILED does not count. `judge_error` is
    infrastructure -- a throttle, a timeout, an empty body -- and leaving it
    settled would turn a transient outage into a permanent hole in the
    denominator that no rerun could fill.

    `skipped: error` is different and does count: the candidate never produced
    an answer, so there is nothing for any judge to grade, ever.
    `format_fail` also counts -- it is a verdict ABOUT the judge's compliance,
    and retrying until it parses would quietly select for compliant outputs.
    """
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            if (r.get("judge") or {}).get("skipped") in RETRYABLE:
                continue
            done.add((r["spec_key"], r["qid"], int(r["repeat"])))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return done


def score_run(cfg: Config, root: Path, resume: bool = True) -> ScoreSummary:
    paths = RunPaths(root)
    rows, malformed = read_traces(paths.traces)
    questions = {q.qid: q for q in load_questions(cfg.eval.questions, cfg.eval.intents)}
    scorers = [build_scorer(n, cfg) for n in cfg.scoring.scorers]

    out = scores_path(root)
    check_judge(cfg, root, resume)
    done = completed(out) if resume else set()
    todo = [r for r in rows
            if (r["spec_key"], r["qid"], int(r["repeat"])) not in done]

    out.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        # Starting over means starting over. Appending would leave two verdicts
        # per row and let whichever landed last silently win.
        out.unlink(missing_ok=True)
    dump_resolved(cfg, scoring_config_path(root))
    fh = out.open("a", encoding="utf-8")
    lock = threading.Lock()
    skipped = fails = judge_down = 0

    def one(row: dict) -> tuple[int, int, int]:
        q = questions[row["qid"]]
        record = {"spec_key": row["spec_key"], "qid": row["qid"],
                  "repeat": row["repeat"], "mode": row["mode"],
                  "agent": row["agent"], "pack": row["pack"],
                  "answerable": q.answerable}
        s = f_ = j = 0
        for sc in scorers:
            result = sc.score(row, q)
            record[sc.name] = result
            s += int(result.get("skipped") == "error")
            j += int(result.get("skipped") in RETRYABLE)
            f_ += int(result.get("format_fail") == 1.0)
        line = json.dumps(record, ensure_ascii=False)
        with lock:
            fh.write(line + "\n")
            fh.flush()
        return s, f_, j

    try:
        with ThreadPoolExecutor(max_workers=max(cfg.run.max_concurrency, 1)) as pool:
            for fut in as_completed([pool.submit(one, r) for r in todo]):
                s, f_, j = fut.result()
                skipped += s
                fails += f_
                judge_down += j
    finally:
        fh.close()

    notes = [(f"judge = {cfg.judge.agent.id} ({cfg.judge.agent.provider})"
              f" · scoring_hash={cfg.scoring_hash}")]
    if malformed:
        notes.append(f"{malformed} unreadable trace line(s) were skipped")
    if fails:
        notes.append(f"{fails} judge response(s) failed to parse (format_fail, NOT wrong answers)")
    if judge_down:
        notes.append(f"{judge_down} judge call(s) failed — rerun `kbeval score` to retry them")
    return ScoreSummary(root=root, rows=len(rows), malformed_traces=malformed,
                        scored=len(todo), resumed=len(rows) - len(todo),
                        skipped_errors=skipped, judge_failures=judge_down,
                        format_fails=fails, notes=notes)


def load_scores(root: Path) -> list[dict]:
    """Rows, deduplicated by key with the LAST one winning.

    A retried judge call appends a second row for the same key. Returning both
    would double-count that row in every mean -- hit@k included -- so the file
    stays append-only (it is evidence) and the reader resolves it. Last wins
    because a row is only ever rewritten after the first attempt failed.
    """
    path = scores_path(root)
    if not path.exists():
        return []
    latest: dict[tuple[str, str, int], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            latest[(r["spec_key"], r["qid"], int(r["repeat"]))] = r
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return list(latest.values())
