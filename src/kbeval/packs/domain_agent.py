"""domain-agent의 chunk 경계와 의미 확장을 kb-eval의 두 층으로 나눠 붙인다.

**왜 chunker와 transform으로 나누는가.** 이 저장소의 계약이 그렇게 요구한다.

- chunker는 **자르기만** 한다: `doc.text[start:end].strip() == chunk.text`
- 텍스트 재작성은 **transform**의 일이고, 그때도 오프셋은 건드리지 않는다

domain-agent의 산출물은 정확히 그 두 겹이다. 경계는 원문 위에서 정해지고
(그쪽 `pipeline/chunk.py`: "경계는 raw text 위에서 정한다"), 의미 확장은 그
경계를 바꾸지 않고 텍스트 뒤에 덧붙는다(그쪽 불변식:
`text.startswith(text_raw)`).

그래서 두 pack이 **같은 경계**를 갖는다.

    domain-agent-raw    chunker=domain_agent, transforms=[]
    domain-agent-full   chunker=domain_agent, transforms=[domain_agent_enrich]

기존 `raw` pack(고정 크기 flat)과 비교하면 "경계 차이"와 "확장 차이"가 섞인다.
위 둘을 비교하면 **확장 효과만** 남는다.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..registry import CHUNKERS, TRANSFORMS
from ..types import Chunk, Document

# export 디렉터리. 소스 설정과 같은 값을 가리켜야 한다.
EXPORT_ROOT = Path(os.environ.get("KBEVAL_DOMAIN_AGENT_EXPORT", "data/domain-agent-export"))


def _rows(variant: str) -> list[dict]:
    path = EXPORT_ROOT / variant / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없다. domain-agent 쪽에서 "
            "`uv run python tools/export_kb_eval_pack.py --out <여기>` 를 먼저 돌린다."
        )
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln]


@CHUNKERS.register("domain_agent")
class DomainAgentChunker:
    """domain-agent가 정한 경계를 그대로 쓴다. **자르기만 한다.**

    텍스트는 확장본이 아니라 **문서 슬라이스**를 쓴다 — chunker의 불변식이
    그것을 요구하기 때문이다. 확장본은 transform이 얹는다.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        self.rows = _rows("domain-agent-rawtext")

    def split(self, doc: Document) -> list[Chunk]:
        out: list[Chunk] = []
        for r in self.rows:
            if r["doc_id"] != doc.doc_id:
                continue
            start, end = r["char_start"], r["char_end"]
            out.append(
                Chunk(r["chunk_id"], doc.doc_id, start, end, doc.text[start:end], dict(r["meta"]))
            )
        return out


@TRANSFORMS.register("domain_agent_enrich")
def domain_agent_enrich(chunks: list[Chunk]) -> list[Chunk]:
    """chunk 텍스트를 domain-agent의 확장본으로 바꾼다. **오프셋은 그대로다.**

    확장본은 `원문 + ## Section + ## Context(모델 해설) + ## Observations`이고,
    원문이 접두로 보존된다. 그래서 오프셋이 가리키는 자리와 텍스트의 앞부분이
    계속 대응한다.
    """
    enriched = {r["chunk_id"]: r["text"] for r in _rows("domain-agent")}
    return [
        Chunk(c.chunk_id, c.doc_id, c.char_start, c.char_end,
              enriched.get(c.chunk_id, c.text), c.meta)
        for c in chunks
    ]
