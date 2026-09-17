"""LLM-as-judge, with the four failure modes kept apart.

correctness / groundedness / abstention / harmful_action are reported
SEPARATELY, never averaged into one score. They fail for different reasons and
a single number would let a pack that teaches confident invention trade
abstention away for correctness and come out ahead.

Two rules shape what the judge is allowed to see:
  * it gets gold_answer, so it can grade facts rather than style;
  * it never gets oracle_ref. For an oracle_ref row that text IS the model's
    input, and a judge comparing an answer against its own prompt would score
    that mode high by construction -- inflating the denominator of every
    realization rate at once.

A judge that returns unparseable output is `format_fail`, not a zero. Scoring a
parse failure as a wrong answer would make formatting ability look like
knowledge -- the exact confound this project keeps writing down.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ...config import Config
from ...llm import build_llm
from ...packs import build_pack
from ...registry import SCORERS
from ...types import Question

FIELDS = ("correctness", "groundedness", "abstention", "harmful_action")
_JSON_RX = re.compile(r"\{.*\}", re.DOTALL)
NO_CONTEXT = "-"


def parse_verdict(text: str) -> dict[str, Any] | None:
    """Tolerate a chatty judge, refuse to guess at a broken one."""
    m = _JSON_RX.search(text or "")
    if not m:
        return None
    try:
        raw = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict) or not any(f in raw for f in FIELDS):
        return None

    out: dict[str, Any] = {}
    for f in FIELDS:
        v = raw.get(f)
        if v is None:
            out[f] = None
        elif isinstance(v, int | float):     # bool is an int; True -> 1.0
            out[f] = float(v)
        else:
            return None
    return out


@SCORERS.register("judge")
class JudgeScorer:
    name = "judge"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.llm = build_llm(cfg.judge.agent, cfg.judge.llm or cfg.llm)
        self.template = cfg.prompts.read("judge")
        self._packs: dict[str, Any] = {}

    def _context(self, row: dict[str, Any]) -> str:
        """Rebuild what the model actually saw, from its chunk ids.

        Reconstructed rather than stored: the pack is deterministic for this
        config hash, so the text is recoverable exactly, and traces stay small
        enough to read. oracle_ref and bare deliberately reconstruct to nothing.
        """
        pack, ids = row.get("pack"), row.get("injected_chunk_ids") or []
        if not pack or not ids:
            return NO_CONTEXT
        if pack not in self._packs:
            self._packs[pack] = build_pack(self.cfg, pack)
        pk = self._packs[pack]
        return "\n".join(pk.get(cid).text for cid in ids if pk.has(cid))

    def score(self, row: dict[str, Any], q: Question) -> dict[str, Any]:
        if row.get("error"):
            return {"skipped": "error"}    # rule 8: a failure is not an answer

        user = self.template.format(
            question=q.query,
            gold_answer=q.gold_answer or NO_CONTEXT,
            context=self._context(row),
            answer=row.get("answer", ""),
        )
        r = self.llm.generate("", user)
        if r.error:
            # reasoning_only is a budget problem, not an outage: the judge spent
            # its whole output on thinking. Retrying it unchanged would loop, so
            # it is recorded as final and the fix is max_tokens.
            kind = "judge_budget" if r.error == "reasoning_only" else "judge_error"
            return {"skipped": kind, "judge_error": r.error}

        verdict = parse_verdict(r.text)
        if verdict is None:
            return {"format_fail": 1.0}

        verdict["format_fail"] = 0.0
        if not q.answerable:
            # An unanswerable question has no correct content to produce, so
            # abstaining IS the correct answer. Leaving correctness at 0 would
            # punish exactly the behaviour the distractors are testing for.
            verdict["correctness"] = verdict["abstention"]
        return verdict
