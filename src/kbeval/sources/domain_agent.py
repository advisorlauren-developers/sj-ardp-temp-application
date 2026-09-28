"""domain-agent 산출물을 Document로 읽는다 (2026-09-21).

domain-agent(`../domain-agent`)는 Extractor 블록을 section 단위로 묶고, 용어를
Domain Pack에 매핑하고, 원문 문장을 용어 뜻이 풀린 형태로 다시 쓴다. 그 결과를
`tools/export_kb_eval_pack.py`가 여기서 읽을 모양으로 내보낸다.

**이 어댑터가 만드는 `text`가 좌표계다.** domain-agent 쪽에서 유지 블록을 읽기
순서로 `\\n`으로 이어붙인 문자열이며, 그쪽 `baseline_chunks.jsonl`의
`doc_char_start/end`와 같은 공간이다. 두 저장소가 같은 좌표를 쓴다는 사실이
gold span을 양쪽에서 쓸 수 있게 만든다.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import SourceCfg
from ..registry import SOURCES
from ..types import Document


@SOURCES.register("domain_agent")
class DomainAgentExport:
    """`params.path`는 export 디렉터리. 그 안에 `document.json`이 있다."""

    def __init__(self, cfg: SourceCfg) -> None:
        self.root = Path(cfg.path)

    def load(self) -> list[Document]:
        payload = json.loads((self.root / "document.json").read_text(encoding="utf-8"))
        return [
            Document(
                doc_id=payload["doc_id"],
                title=payload.get("title") or payload["doc_id"],
                text=payload["text"],
                meta=payload.get("meta", {}),
            )
        ]
