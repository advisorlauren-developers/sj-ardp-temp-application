"""JIRA adapter -- a stub during the Q&A-first PoC, but the two decisions that
decide whether BUG/feature intents can work at all are already baked in.

1. The answer usually lives in the COMMENT THREAD, not the description. The
   ticket body is a symptom report; the cause and the fix are in comment #7.
   An adapter that flattens only the description removes the gold from the
   corpus entirely -- a data-preparation failure that will look exactly like a
   retrieval failure unless you know to check here.

2. Tickets go STALE. status / resolution / fixVersion / updated must end up in
   the chunk text (see the inline_meta transform), not only in a vector-store
   field: the model cannot read fields it is never shown. In telecom support a
   confident answer about a bug fixed six months ago is an operational
   incident, not a wrong answer.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..registry import SOURCES
from ..types import Document


@SOURCES.register("jira_export")
class JiraExport:
    def __init__(self, cfg) -> None:
        self.path = Path(cfg.path)
        self.include_comments = cfg.params.get("include_comments", True)

    def load(self) -> list[Document]:
        docs: list[Document] = []
        for p in sorted(self.path.glob("*.json")):
            for issue in json.loads(p.read_text(encoding="utf-8")):
                docs.append(self._to_doc(issue))
        return docs

    def _to_doc(self, issue: dict) -> Document:
        f = issue.get("fields", {})
        parts = [f.get("summary", ""), "", f.get("description") or ""]
        if self.include_comments:
            for c in (f.get("comment", {}) or {}).get("comments", []):
                parts += ["", f"[comment @ {c.get('created', '')}]", c.get("body", "")]
        return Document(
            doc_id=issue["key"],
            title=f.get("summary", issue["key"]),
            text="\n".join(parts).strip(),
            meta={
                "status": (f.get("status") or {}).get("name"),
                "resolution": (f.get("resolution") or {}).get("name"),
                "fix_version": [v.get("name") for v in f.get("fixVersions", [])],
                "component": [c.get("name") for c in f.get("components", [])],
                "updated": f.get("updated"),
                "issue_type": (f.get("issuetype") or {}).get("name"),
            },
        )