"""Bedrock embeddings: Cohere Embed v4 (InvokeModel), bearer-token auth.

Why v4 and not embed-multilingual-v3: v3 truncates input at 512 tokens, and
domain-agent puts its expansion at the END of a chunk (raw text first, then
## Context). Truncation would embed da_full without the very part that makes it
da_full -- a retriever-side artefact reported as "the expansion did nothing".

v4 is INFERENCE_PROFILE only, so the id carries the `us.` prefix; without it
the call fails exactly like a model that does not exist (the same trap as the
Anthropic judge, CLAUDE.md S9).

The response body is validated: Bedrock can return 200 with an error document,
and an empty vector list must not become a zero-similarity index.
"""
from __future__ import annotations

import json
import random
import threading
import time

from ..config import EmbedderCfg
from ..registry import EMBEDDERS
from .base import Kind

_INPUT_TYPE = {"query": "search_query", "document": "search_document"}
_THROTTLE = ("throttl", "toomanyrequests", "serviceunavailable", "timeout",
             "modelnotready", "internalserver")


@EMBEDDERS.register("bedrock")
class BedrockEmbedder:
    def __init__(self, cfg: EmbedderCfg) -> None:
        self.cfg = cfg
        self.model_id = cfg.id
        self._client = None
        self._lock = threading.Lock()

    def _c(self):
        with self._lock:
            if self._client is None:
                import boto3  # lazy: no AWS, no import
                from botocore.config import Config as BotoCfg
                self._client = boto3.client(
                    "bedrock-runtime", region_name=self.cfg.region,
                    config=BotoCfg(read_timeout=60, retries={"max_attempts": 1}))
        return self._client

    def _call(self, texts: list[str], kind: Kind) -> list[list[float]]:
        body = {"texts": texts, "input_type": _INPUT_TYPE[kind],
                "embedding_types": ["float"], "output_dimension": self.cfg.dim}
        r = self._c().invoke_model(modelId=self.model_id, body=json.dumps(body))
        doc = json.loads(r["body"].read())
        vecs = (doc.get("embeddings") or {}).get("float")
        if not isinstance(vecs, list) or len(vecs) != len(texts):
            raise RuntimeError(f"embedding response malformed: keys={sorted(doc)}")
        if any(len(v) != self.cfg.dim for v in vecs):
            raise RuntimeError(f"embedding dim mismatch: wanted {self.cfg.dim}")
        return vecs

    def embed(self, texts: list[str], kind: Kind) -> list[list[float]]:
        for attempt in range(6):
            try:
                return self._call(texts, kind)
            except Exception as e:  # classify, then re-raise
                if attempt == 5 or not any(t in str(e).lower() for t in _THROTTLE):
                    raise
                time.sleep(min(2 ** attempt, 20) + random.random())
        raise AssertionError("unreachable")
