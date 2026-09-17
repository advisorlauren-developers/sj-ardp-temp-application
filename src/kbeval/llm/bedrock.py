"""Bedrock, in the two shapes it actually comes in.

`runtime`  -- the native Converse API through boto3.
`mantle`   -- the OpenAI-compatible chat/completions endpoint, plain HTTP with
              a bearer token. Same model, different id and different payload;
              a model id from one endpoint sent to the other reads exactly like
              a model that does not exist.

Everything here funnels through llm.base.validate(), because the failure this
layer exists to contain is Bedrock returning HTTP 200 with an error document.
Counting that as a wrong answer would show up in the report as a pack that made
the model worse.

NOT YET EXERCISED AGAINST REAL AWS -- that is S9, and S9 starts with
`aws bedrock list-foundation-models` for exactly this reason.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request

from ..config import AgentSpec, LLMCfg
from ..registry import LLMS
from .base import LLMResponse, validate

_THROTTLE = ("throttl", "toomanyrequests", "serviceunavailable", "timeout",
             "modelnotready", "internalserver")


def _is_transient(msg: str) -> bool:
    low = msg.lower()
    return any(t in low for t in _THROTTLE)


def _inference_config(cfg: LLMCfg) -> dict:
    """top_p is omitted when None.

    Several Bedrock families reject temperature and top_p together, and the
    rejection arrives as a generic validation error that looks like a bad model
    id. Sending only what was asked for avoids the whole class.
    """
    out: dict = {"maxTokens": cfg.max_tokens}
    if cfg.temperature is not None:
        out["temperature"] = cfg.temperature
    if cfg.top_p is not None:
        out["topP"] = cfg.top_p
    return out


@LLMS.register("bedrock")
class BedrockLLM:
    def __init__(self, spec: AgentSpec, cfg: LLMCfg) -> None:
        self.spec, self.cfg = spec, cfg
        self.model_id = spec.id
        self.endpoint = spec.endpoint
        self._client = None
        # The runner calls this from a ThreadPool. boto3 client CREATION is not
        # thread-safe, and the race surfaces as a sporadic import-time error
        # that reads like a credentials problem.
        self._client_lock = threading.Lock()

    # -- transports -------------------------------------------------------
    def _converse(self, system: str, user: str) -> tuple[str, dict, bool]:
        if self._client is None:
            import boto3  # lazy: no AWS, no import
            from botocore.config import Config as BotoCfg
            self._client = boto3.client(
                "bedrock-runtime", region_name=self.cfg.region,
                config=BotoCfg(read_timeout=self.cfg.timeout_s, retries={"max_attempts": 1}))

        # An EMPTY system block is rejected outright ("Invalid length for
        # parameter system[0].text"), and the judge sends none -- its whole
        # instruction is the user message. Omit the key rather than send "".
        kwargs = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "inferenceConfig": _inference_config(self.cfg),
        }
        if system.strip():
            kwargs["system"] = [{"text": system}]
        r = self._client.converse(**kwargs)
        blocks = r.get("output", {}).get("message", {}).get("content", [])
        text = "".join(b.get("text", "") for b in blocks)
        # Reasoning arrives as its own block type, which is the clean case: it
        # is dropped here rather than parsed out of the answer. It still has to
        # be REPORTED, because a model that spent its whole output budget
        # thinking produced no answer for a reason worth knowing.
        reasoned = any("reasoningContent" in b for b in blocks)
        return text, {"provider": "bedrock", "endpoint": "runtime",
                      "model": self.model_id, "stop_reason": r.get("stopReason"),
                      **r.get("usage", {})}, reasoned

    def _mantle(self, system: str, user: str) -> tuple[str, dict, bool]:
        token = (os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
                 or os.environ.get("BEDROCK_API_KEY"))
        if not token:
            raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK (or BEDROCK_API_KEY) "
                               "is required for endpoint=mantle")
        messages = ([{"role": "system", "content": system}] if system.strip() else [])
        messages.append({"role": "user", "content": user})
        body = {"model": self.model_id,
                "messages": messages,
                "max_completion_tokens": self.cfg.max_tokens}
        if self.cfg.temperature is not None:
            body["temperature"] = self.cfg.temperature
        if self.cfg.top_p is not None:
            body["top_p"] = self.cfg.top_p

        req = urllib.request.Request(
            f"https://bedrock-runtime.{self.cfg.region}.amazonaws.com/openai/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            payload = json.loads(resp.read().decode())

        choices = payload.get("choices") or []
        msg = choices[0].get("message", {}) if choices else {}
        text = msg.get("content") or ""
        reasoned = bool(msg.get("reasoning_content") or msg.get("reasoning"))
        return text, {"provider": "bedrock", "endpoint": "mantle",
                      "model": self.model_id, **payload.get("usage", {})}, reasoned

    # -- public -----------------------------------------------------------
    def generate(self, system: str, user: str) -> LLMResponse:
        send = self._converse if self.endpoint == "runtime" else self._mantle
        t0 = time.perf_counter()
        last = ""

        for attempt in range(self.cfg.max_retries):
            try:
                text, usage, reasoned = send(system, user)
                return validate(text, (time.perf_counter() - t0) * 1000, usage, reasoned)
            except urllib.error.HTTPError as e:
                last = f"http_{e.code}: {e.read()[:200]!r}"
            # deliberately blind: ANY provider failure must become an `error`
            # field. An exception escaping here would abort a sweep mid-run.
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"

            if not _is_transient(last) or attempt == self.cfg.max_retries - 1:
                break
            # full jitter: a synchronised retry storm across a ThreadPool is
            # itself a throttling source
            time.sleep(random.uniform(0, min(2 ** attempt, 30)))

        return LLMResponse(text="", latency_ms=(time.perf_counter() - t0) * 1000,
                           usage={"provider": "bedrock", "model": self.model_id},
                           error=last or "unknown_error")
