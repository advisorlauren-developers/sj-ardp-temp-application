"""A deterministic stand-in embedder: hashed character trigrams.

It exists so dense and hybrid retrieval run end to end with no AWS, the same
reason llm/mock.py exists. It is lexical at the character level -- it will NOT
bridge a Korean question to an English manual, which is exactly the thing a
real multilingual embedder is for. Numbers from it are plumbing evidence only.
"""
from __future__ import annotations

import hashlib
import math

from ..config import EmbedderCfg
from ..registry import EMBEDDERS
from .base import Kind


@EMBEDDERS.register("mock")
class MockEmbedder:
    def __init__(self, cfg: EmbedderCfg) -> None:
        self.model_id = cfg.id
        self.dim = cfg.dim

    def _one(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        s = f"  {text.lower()}  "
        for i in range(len(s) - 2):
            h = hashlib.blake2b(s[i:i + 3].encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            v[idx] += 1.0 if h[4] & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed(self, texts: list[str], kind: Kind) -> list[list[float]]:
        return [self._one(t) for t in texts]
