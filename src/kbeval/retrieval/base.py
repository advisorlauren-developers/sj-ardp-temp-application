"""The Retriever contract, and the Korean tokenizer every retriever shares.

Retrieval is settled BEFORE any model is attached. A broken retriever plus a
generator is impossible to diagnose -- a wrong answer could be a missing chunk
or a confused model -- and it burns tokens to stay confused. Everything in this
package costs nothing to run.

The rule that keeps the pack axis honest: the ALGORITHM is fixed across packs,
only the INDEX is pack-dependent. That is why RetrievalCfg is a top-level block
and not a per-pack one -- a pack cannot bring its own retriever any more than
it can bring its own model.
"""
from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from ..config import RetrievalCfg
from ..registry import RETRIEVERS
from ..types import Chunk, Retrieved

# Latin/digit runs keep internal - _ . so "E-411" stays ONE token, and Hangul
# runs are split off from them so "autofind로" yields "autofind" + "로".
_WORD_RX = re.compile(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*|[가-힣]+")


def _is_hangul(word: str) -> bool:
    return "가" <= word[0] <= "힣"


def tokenize(text: str, bigrams: bool = True) -> list[str]:
    """Whitespace tokens plus Hangul syllable bigrams.

    Korean is agglutinative: "프로파일은", "프로파일을", "프로파일이" are three
    surface forms of one word, so whitespace tokens alone make recall collapse.
    Syllable bigrams give partial credit -- 프로/로파/파일 match across all of
    them -- at zero dependency cost. A morphological analyser would be better
    and can replace this later, but it must then be FIXED ACROSS PACKS, or the
    analyser's quality shows up in the results as data quality.

    What this deliberately does NOT do is normalise terms. Folding "E411" onto
    "E-411" here would apply to every pack at once and silently delete the
    normalize_terms factor from the experiment -- the `full` pack would have
    nothing left to contribute. Term normalisation is a pack transform, which
    is exactly where it can be measured.
    """
    out: list[str] = []
    for m in _WORD_RX.finditer(text.lower()):
        w = m.group()
        out.append(w)
        # len > 2 only: for a two-syllable word the bigram IS the word, and
        # emitting both would double its term frequency against longer ones.
        if bigrams and len(w) > 2 and _is_hangul(w):
            out.extend(w[i:i + 2] for i in range(len(w) - 1))
    return out


@runtime_checkable
class Retriever(Protocol):
    def index(self, chunks: list[Chunk]) -> None:
        ...

    def search(self, query: str, k: int) -> list[Retrieved]:
        ...


def build_retriever(cfg: RetrievalCfg) -> Retriever:
    """The only entry point. Callers never name a concrete retriever, so the
    dense and hybrid variants land as config changes, not call-site edits."""
    return RETRIEVERS.get(cfg.retriever)(cfg)
