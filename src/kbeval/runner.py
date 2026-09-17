"""The sweep runner: append-only, resumable, and refusing to start on bad data.

At this scale a mid-run failure is not a risk, it is a certainty -- throttling,
a expired credential, a laptop lid. So the unit of progress is one
(spec_key, qid, repeat) row on disk, and a rerun picks up exactly what is
missing. Nothing is ever rewritten; a trace file is evidence, and evidence that
can be edited in place is not evidence.

Three things are written together at the start, on purpose:
  config.resolved.yaml  the exact settings, so the hash can be recomputed later
  gold_coverage.json    proof that every pack still contained its answers
  traces.jsonl          the rows themselves

The coverage file is a gate, not a report. Rule 9: if a transform deleted a
gold chunk, the sweep must not start -- a run built on a pack with missing
answers looks in the report like a model that got worse, and by then the money
is spent.
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config, dump_resolved
from .packs import build_pack, load_questions
from .sweep import RunSpec, expand_sweep
from .types import Question, Trace
from .wiring import build as build_agent_for


@dataclass(frozen=True)
class TaskKey:
    spec_key: str
    qid: str
    repeat: int

    def as_tuple(self) -> tuple[str, str, int]:
        return (self.spec_key, self.qid, self.repeat)


@dataclass
class RunPaths:
    root: Path

    @property
    def traces(self) -> Path:
        return self.root / "traces.jsonl"

    @property
    def resolved(self) -> Path:
        return self.root / "config.resolved.yaml"

    @property
    def coverage(self) -> Path:
        return self.root / "gold_coverage.json"


@dataclass
class RunSummary:
    run_id: str
    root: Path
    specs: int
    planned: int
    resumed: int
    written: int
    errors: int
    elapsed_s: float
    error_samples: list[str] = field(default_factory=list)


class TraceWriter:
    """One lock, one handle, flushed per row.

    Buffering would trade the only property that makes resume work -- a row on
    disk means that unit is done -- for throughput nobody needs at this scale.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _heal_torn_line(path)
        self._fh = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, trace: Trace) -> None:
        line = json.dumps(trace.to_json(), ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _heal_torn_line(path: Path) -> None:
    """Terminate a half-written final line before appending to it.

    A crash mid-write leaves a row with no trailing newline. Appending after it
    would splice the next row onto the wreckage and produce ONE corrupt line
    where there were two -- and unlike a torn tail, a spliced row is not
    obviously broken to a reader. One byte prevents it.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as fh:
        fh.seek(-1, 2)
        if fh.read(1) != b"\n":
            fh.write(b"\n")


def read_traces(path: Path) -> tuple[list[dict], int]:
    """Rows plus the number of unreadable lines.

    Malformed lines are skipped, never guessed at -- but they are COUNTED and
    surfaced, because "some evidence was lost" and "there was less evidence"
    are different claims and only one of them is honest.
    """
    if not path.exists():
        return [], 0
    rows, bad = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    return rows, bad


def completed_keys(path: Path) -> set[tuple[str, str, int]]:
    """Read what is already done, tolerating a half-written final line.

    A crash during a write leaves a truncated row. Treating that as corruption
    and refusing to resume would be worse than redoing one call, so the line is
    skipped and its task gets rerun.
    """
    if not path.exists():
        return set()
    done: set[tuple[str, str, int]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            done.add((row["spec_key"], row["qid"], int(row["repeat"])))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return done


def check_gold(cfg: Config, paths: RunPaths) -> dict:
    """Rule 9, as a gate. Raises before a single token is spent."""
    qs = load_questions(cfg.eval.questions, cfg.eval.intents)
    report = {}
    for name in cfg.sweep.pack:
        pk = build_pack(cfg, name)
        report[name] = pk.verify_gold_coverage(qs, cfg.eval.gold_overlap_ratio)

    paths.root.mkdir(parents=True, exist_ok=True)
    paths.coverage.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    broken = {n: r["missing_gold"] for n, r in report.items() if r["coverage"] < 1.0}
    if broken:
        raise RuntimeError(f"gold missing after transforms, refusing to run: {broken}")
    return report


def plan_tasks(cfg: Config, specs: list[RunSpec], questions: list[Question]) -> list[tuple[RunSpec, Question, int]]:
    return [(s, q, r)
            for s in specs
            for q in questions
            for r in range(cfg.fixed.n_repeat)]


def run_sweep(cfg: Config, limit: int | None = None, resume: bool = True,
              on_progress: Callable[[int, int], None] | None = None) -> RunSummary:
    t0 = time.perf_counter()
    paths = RunPaths(Path(cfg.run.out_dir) / cfg.run_id)
    check_gold(cfg, paths)
    dump_resolved(cfg, paths.resolved)

    specs = expand_sweep(cfg)
    questions = load_questions(cfg.eval.questions, cfg.eval.intents)
    if limit:
        questions = questions[:limit]

    tasks = plan_tasks(cfg, specs, questions)
    done = completed_keys(paths.traces) if resume else set()
    todo = [t for t in tasks if TaskKey(t[0].key, t[1].qid, t[2]).as_tuple() not in done]

    # One agent per spec, not per task: building a rag agent re-indexes the
    # retriever, and paying that per question would dominate the run.
    agents = {s.key: build_agent_for(cfg, s.agent, s.mode, s.pack) for s in specs}
    packs = {s.pack: build_pack(cfg, s.pack) for s in specs if s.pack}

    writer = TraceWriter(paths.traces)
    errors: list[str] = []
    written = 0
    lock = threading.Lock()

    def one(item: tuple[RunSpec, Question, int]) -> str | None:
        spec, q, repeat = item
        res = agents[spec.key].answer(q)
        gold = (packs[spec.pack].gold_chunk_ids(q, cfg.eval.gold_overlap_ratio)
                if spec.pack else [])
        writer.write(Trace(
            run_id=cfg.run_id, config_hash=cfg.hash, spec_key=spec.key, repeat=repeat,
            qid=q.qid, intent=q.intent, agent=spec.agent, mode=spec.mode, pack=spec.pack,
            query=q.query, answer=res.text,
            retrieved=[{"chunk_id": r.chunk_id, "score": r.score, "rank": r.rank}
                       for r in res.retrieved],
            injected_chunk_ids=res.injected_chunk_ids,
            injected_tokens=res.injected_tokens,
            skipped_chunk_ids=res.skipped_chunk_ids,
            gold_chunk_ids=gold,
            latency_ms=res.latency_ms, usage=res.usage, error=res.error,
        ))
        return res.error

    try:
        with ThreadPoolExecutor(max_workers=max(cfg.run.max_concurrency, 1)) as pool:
            futures = [pool.submit(one, t) for t in todo]
            for i, f in enumerate(as_completed(futures), start=1):
                err = f.result()          # a crash in a worker must not be swallowed
                with lock:
                    written += 1
                    if err:
                        errors.append(err)
                if on_progress:
                    on_progress(i, len(todo))
    finally:
        writer.close()

    return RunSummary(
        run_id=cfg.run_id, root=paths.root, specs=len(specs), planned=len(tasks),
        resumed=len(tasks) - len(todo), written=written, errors=len(errors),
        elapsed_s=round(time.perf_counter() - t0, 2),
        error_samples=sorted(set(errors))[:5],
    )


def load_traces(root: Path) -> Iterable[dict]:
    """Read a run back. Used by `kbeval score` and `kbeval report` at S8."""
    rows, _ = read_traces(RunPaths(root).traces)
    return rows
