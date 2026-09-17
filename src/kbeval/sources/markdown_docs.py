"""Internal wiki / manual pages as .md files."""
from __future__ import annotations

from pathlib import Path

from ..registry import SOURCES
from ..types import Document


@SOURCES.register("markdown_docs")
class MarkdownDocs:
    def __init__(self, cfg) -> None:
        self.root = Path(cfg.path)
        self.glob = cfg.params.get("glob", "**/*.md")

    def load(self) -> list[Document]:
        if not self.root.exists():
            raise FileNotFoundError(f"source path not found: {self.root}")
        docs: list[Document] = []
        for p in sorted(self.root.glob(self.glob)):
            meta, body = self._split_front_matter(p.read_text(encoding="utf-8"))
            docs.append(Document(
                doc_id=p.relative_to(self.root).as_posix(),
                title=meta.pop("title", p.stem),
                text=body,
                meta=meta,
            ))
        if not docs:
            raise ValueError(f"no documents matched {self.root}/{self.glob}")
        return docs

    @staticmethod
    def _split_front_matter(raw: str) -> tuple[dict, str]:
        """Front matter is lifted into meta AND removed from text.

        Offsets therefore index the BODY only. If someone later edits the
        `updated:` line, every gold span in the file stays valid -- which is the
        difference between metadata being editable and gold labels being
        disposable.
        """
        lines = raw.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}, raw
        try:
            end = lines.index("---", 1)
        except ValueError:
            return {}, raw
        meta = {}
        for ln in lines[1:end]:
            if ":" in ln:
                k, _, v = ln.partition(":")
                meta[k.strip()] = v.strip()
        return meta, "\n".join(lines[end + 1:]).lstrip("\n")