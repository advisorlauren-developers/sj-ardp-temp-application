"""Config = the coordinates of one reproducible experiment.

Three layers merge in order:  base.yaml -> profile.yaml -> CLI --set overrides.
The result is validated with extra="forbid" (a typo is an error, never a silent
no-op), dumped verbatim into the run directory, and hashed. That hash is
stamped onto every trace row, so any number can be traced back to its settings.

S1 defines only the blocks the sweep needs. Later stages add their own block
alongside their Protocol -- and because unknown keys are rejected, you cannot
use a setting you have not declared.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunCfg(Strict):
    """Operational settings, mostly OUTSIDE the hash.

    `seed` is a control variable and is hashed. `name`, `out_dir` and
    `max_concurrency` are not: renaming a run or raising the thread count does
    not change a single answer, and hashing them would mean every operational
    tweak silently forbade comparison with everything measured before it.
    """
    name: str = "poc"
    seed: int = 0
    out_dir: str = "runs"
    max_concurrency: int = 4


class AgentSpec(Strict):
    """A base model.

    `endpoint` matters: Bedrock model IDs differ between the native
    bedrock-runtime endpoint and the OpenAI-compatible bedrock-mantle endpoint,
    and several families additionally need a us./global. inference-profile
    prefix. A wrong ID reads like a missing model.
    """
    provider: Literal["bedrock", "mock", "mock_judge"] = "bedrock"
    endpoint: Literal["runtime", "mantle"] = "runtime"
    id: str
    family: Literal["dense", "moe"] = "dense"
    params_b: float | None = None
    thinking: bool = False
    notes: str | None = None


class PackCfg(Strict):
    name: str
    chunker: str = "recursive"
    chunk_chars: int = 900
    chunk_overlap: int = 120
    transforms: list[str] = Field(default_factory=list)


class SourceCfg(Strict):
    """Where documents come from. `adapter` names a SOURCES registry key, so
    switching to JIRA later is a config change, not a code change."""
    adapter: str = "markdown_docs"
    path: str = "data/raw/docs"
    params: dict[str, Any] = Field(default_factory=dict)


class EvalCfg(Strict):
    """The question set, and how a gold span is allowed to land on a chunk.

    `gold_overlap_ratio` is a measurement setting, not a convenience: raising it
    shrinks every pack's gold set at once and therefore moves hit@k for reasons
    that have nothing to do with the packs. It lives in the hash so that a run
    scored under a different threshold cannot be silently compared with this one.
    """
    questions: str = "data/eval/questions.jsonl"
    intents: list[str] = Field(default_factory=lambda: ["qa"])
    gold_overlap_ratio: float = Field(default=0.3, gt=0.0, le=1.0)


class BM25Cfg(Strict):
    k1: float = 1.2
    b: float = 0.75


class TokenizerCfg(Strict):
    bigrams: bool = True     # Hangul syllable bigrams; off = whitespace only


class RetrievalCfg(Strict):
    """Fixed across packs, by design.

    Only the INDEX may depend on the pack -- that is the thing being measured.
    Letting a pack choose its own retriever or its own tokenizer would confound
    the two axes, so these settings live here and not inside PackCfg.

    `candidate_k` is the retrieval depth, NOT the context control. How much of
    that candidate list actually reaches the model is decided by
    fixed.context_token_budget at S6.
    """
    retriever: str = "bm25"
    candidate_k: int = 10
    bm25: BM25Cfg = Field(default_factory=BM25Cfg)
    tokenizer: TokenizerCfg = Field(default_factory=TokenizerCfg)


class PromptsCfg(Strict):
    """Prompt TEXT is versioned on disk, never as a string literal in .py.

    A literal is invisible to the config hash, so two runs with different
    instructions would share a coordinate. Paths alone are not enough either --
    editing a file in place leaves the path unchanged -- so canonical() folds
    the file CONTENTS into the hash. Change a word in system_qa.md and the run
    id moves, which is the whole point.
    """
    system_qa: str = "prompts/system_qa.md"
    context: str = "prompts/context_v1.md"
    chunk: str = "prompts/chunk_v1.md"
    judge: str = "prompts/judge_v1.md"
    no_material: str = "(제공된 자료 없음)"

    def read(self, which: str) -> str:
        return Path(getattr(self, which)).read_text(encoding="utf-8")

    def digest(self) -> str:
        """The prompts the CANDIDATE sees. The judge prompt is not one of them."""
        h = hashlib.sha256()
        for which in ("system_qa", "context", "chunk"):
            h.update(self.read(which).encode())
        return h.hexdigest()[:12]

    def judge_digest(self) -> str:
        return hashlib.sha256(self.read("judge").encode()).hexdigest()[:12]


class LLMCfg(Strict):
    """Decoding settings, held constant across every model and every pack.

    temperature 0 is not about answer quality, it is about being able to
    attribute a difference to the pack rather than to sampling noise.
    """
    temperature: float | None = 0.0   # None -> omitted; some models reject it outright
    top_p: float | None = None        # None -> omitted from the payload entirely
    max_tokens: int = 512
    timeout_s: int = 60
    max_retries: int = 5
    region: str = "us-west-2"


class JudgeCfg(Strict):
    """The judge is a control variable, so it lives in the hash.

    It must be FIXED across the sweep and much stronger than every candidate.
    Grading a 30B with a 30B produces confidence intervals that mean nothing,
    and a judge that changed between packs would move scores on its own.
    """
    agent: AgentSpec = Field(default_factory=lambda: AgentSpec(provider="mock_judge",
                                                               id="mock-judge"))
    abstain_marker: str = "답변 불가"
    llm: LLMCfg | None = None
    """Decoding settings for the judge alone.

    The candidates want temperature 0 so a pack difference cannot be sampling
    noise. Some frontier models reject `temperature` outright -- Claude
    Sonnet 5 answers "temperature is deprecated for this model" -- so the judge
    needs its own block rather than a shared one bent to fit both.

    None means "use the global llm block".
    """


class ScoringCfg(Strict):
    scorers: list[str] = Field(default_factory=lambda: ["retrieval", "judge"])
    ks: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    min_headroom: float = 0.05     # below this the realization denominator is noise


class SweepCfg(Strict):
    agent: list[str]
    mode: list[str] = Field(default_factory=lambda: ["bare", "rag", "oracle_ref"])
    pack: list[str]


class FixedCfg(Strict):
    """Everything held constant. Changing any of these invalidates comparison
    with earlier runs -- which is exactly why they live inside the hash."""
    context_token_budget: int = 3000     # the control is TOKENS, not top_k
    max_injected_chunks: int = 8
    chars_per_token: float = 2.0
    prompt_version: str = "v1"
    n_repeat: int = 1


class Config(Strict):
    run: RunCfg = Field(default_factory=RunCfg)
    agents: dict[str, AgentSpec]
    source: SourceCfg = Field(default_factory=SourceCfg)
    packs: dict[str, PackCfg]
    eval: EvalCfg = Field(default_factory=EvalCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    prompts: PromptsCfg = Field(default_factory=PromptsCfg)
    llm: LLMCfg = Field(default_factory=LLMCfg)
    judge: JudgeCfg = Field(default_factory=JudgeCfg)
    scoring: ScoringCfg = Field(default_factory=ScoringCfg)
    sweep: SweepCfg
    fixed: FixedCfg = Field(default_factory=FixedCfg)

    @model_validator(mode="after")
    def _check_refs(self) -> Config:
        for a in self.sweep.agent:
            if a not in self.agents:
                raise ValueError(f"sweep.agent '{a}' not defined in agents:")
        for p in self.sweep.pack:
            if p not in self.packs:
                raise ValueError(f"sweep.pack '{p}' not defined in packs:")
        return self

    def canonical(self) -> str:
        """Everything that shapes a TRACE, and nothing else.

        Includes the candidate prompt text: without the digest, editing
        prompts/system_qa.md would leave the hash untouched and two different
        experiments would occupy one coordinate.

        Excludes the judge and the scoring settings. They do not change a
        single answer -- they change how answers are GRADED -- so folding them
        in would mean swapping the judge invalidated the coordinate of traces
        it never touched, and a rerun would have to buy those answers again.
        Grading provenance lives in scoring_hash and in scoring.resolved.yaml.
        """
        payload = self.model_dump(mode="json") | {"_prompts_digest": self.prompts.digest()}
        payload["run"] = {"seed": self.run.seed}     # operational fields excluded
        payload.pop("judge", None)
        payload.pop("scoring", None)
        # the judge prompt PATH belongs to grading too, not just its contents
        payload.get("prompts", {}).pop("judge", None)
        return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))

    def scoring_canonical(self) -> str:
        """The grading coordinate: which judge, how it decodes, what it reads,
        and the thresholds the report applies."""
        payload = {"judge": self.judge.model_dump(mode="json"),
                   "scoring": self.scoring.model_dump(mode="json"),
                   "judge_prompt": self.prompts.judge,
                   "_judge_prompt_digest": self.prompts.judge_digest()}
        return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:12]

    @property
    def scoring_hash(self) -> str:
        return hashlib.sha256(self.scoring_canonical().encode()).hexdigest()[:12]

    @property
    def run_id(self) -> str:
        return f"{self.run.name}-{self.hash}"


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def _coerce(s: str) -> Any:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def _apply_override(d: dict, dotted: str) -> None:
    key, _, val = dotted.partition("=")
    cur = d
    parts = key.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = _coerce(val)


def load_config(paths: list[str | Path], overrides: list[str] | None = None) -> Config:
    merged: dict[str, Any] = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            merged = _deep_merge(merged, yaml.safe_load(f) or {})
    for o in overrides or []:
        _apply_override(merged, o)
    return Config.model_validate(merged)


def dump_resolved(cfg: Config, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg.model_dump(mode="json"),
                                  allow_unicode=True, sort_keys=True), encoding="utf-8")