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

from ..config import Config
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
    skipped_errors: int
    format_fails: int
    notes: list[str] = field(default_factory=list)


def scores_path(root: Path) -> Path:
    return root / "scores.jsonl"


def completed(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
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
    done = completed(out) if resume else set()
    todo = [r for r in rows
            if (r["spec_key"], r["qid"], int(r["repeat"])) not in done]

    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("a", encoding="utf-8")
    lock = threading.Lock()
    skipped = fails = 0

    def one(row: dict) -> tuple[int, int]:
        q = questions[row["qid"]]
        record = {"spec_key": row["spec_key"], "qid": row["qid"],
                  "repeat": row["repeat"], "mode": row["mode"],
                  "agent": row["agent"], "pack": row["pack"],
                  "answerable": q.answerable}
        s = f_ = 0
        for sc in scorers:
            result = sc.score(row, q)
            record[sc.name] = result
            s += int(bool(result.get("skipped")))
            f_ += int(result.get("format_fail") == 1.0)
        line = json.dumps(record, ensure_ascii=False)
        with lock:
            fh.write(line + "\n")
            fh.flush()
        return s, f_

    try:
        with ThreadPoolExecutor(max_workers=max(cfg.run.max_concurrency, 1)) as pool:
            for fut in as_completed([pool.submit(one, r) for r in todo]):
                s, f_ = fut.result()
                skipped += s
                fails += f_
    finally:
        fh.close()

    notes = []
    if malformed:
        notes.append(f"{malformed} unreadable trace line(s) were skipped")
    if fails:
        notes.append(f"{fails} judge response(s) failed to parse (format_fail, NOT wrong answers)")
    return ScoreSummary(root=root, rows=len(rows), malformed_traces=malformed,
                        scored=len(todo), resumed=len(rows) - len(todo),
                        skipped_errors=skipped, format_fails=fails, notes=notes)


def load_scores(root: Path) -> list[dict]:
    path = scores_path(root)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
