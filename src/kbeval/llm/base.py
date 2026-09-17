"""The LLMClient contract, plus the two response rules that protect the numbers.

Rule 8 -- a failed call is an ERROR, never an answer. Bedrock will happily
return HTTP 200 with an error document in the body; scoring that as a wrong
answer turns an outage into "this pack made the model worse".

Rule 10 -- reasoning tokens never reach the judge. A model that thinks out loud
inside tags gets stripped here, once, for every provider. Models that reason
inline cannot be separated at all; those are flagged on AgentSpec.thinking and
reported as their own layer instead of being silently mixed in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..config import AgentSpec, LLMCfg
from ..registry import LLMS

_THINK_RX = re.compile(
    r"<\s*(think|thinking|reasoning|scratchpad)\s*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL)
_UNCLOSED_RX = re.compile(r"<\s*(think|thinking|reasoning|scratchpad)\s*>.*\Z",
                          re.IGNORECASE | re.DOTALL)


@dataclass(slots=True)
class LLMResponse:
    text: str
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None
    had_thinking: bool = False


def strip_thinking(text: str) -> tuple[str, bool]:
    """Remove tag-wrapped reasoning. Returns (clean_text, had_thinking).

    The unclosed case matters: a model that hits max_tokens mid-thought emits
    an opening tag and never closes it. Left alone, the entire response would
    be reasoning scored as an answer.
    """
    clean, n = _THINK_RX.subn("", text)
    clean, n2 = _UNCLOSED_RX.subn("", clean)
    return clean.strip(), bool(n or n2)


def validate(text: str, latency_ms: float, usage: dict[str, Any],
             reasoned: bool = False) -> LLMResponse:
    """The single funnel every provider returns through.

    An empty body after stripping is a failure, not an abstention: abstention
    is a deliberate `답변 불가`, and conflating the two would let an infra
    problem read as a model correctly declining to answer.

    `reasoned` is for providers that separate reasoning natively (Bedrock
    returns it as its own content block). When that is ALL that came back, the
    error says so: "reasoning_only" is a budget problem with an obvious fix,
    while "empty_response" sends the reader looking for an outage.
    """
    clean, had = strip_thinking(text or "")
    had = had or reasoned
    if not clean:
        return LLMResponse(text="", usage=usage, latency_ms=latency_ms, had_thinking=had,
                           error="reasoning_only" if reasoned else "empty_response")
    return LLMResponse(text=clean, usage=usage, latency_ms=latency_ms, had_thinking=had)


@runtime_checkable
class LLMClient(Protocol):
    def generate(self, system: str, user: str) -> LLMResponse:
        ...


def build_llm(spec: AgentSpec, cfg: LLMCfg) -> LLMClient:
    """Dispatch on provider, so `mock` and `bedrock` are the same call site."""
    return LLMS.get(spec.provider)(spec, cfg)
