"""Load help-centre articles and their front-matter metadata."""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from .models import Document

FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)

# Front-matter keys an article must declare for its chunks to be indexable.
REQUIRED_FRONT_MATTER = ("article_id",)


class IngestError(RuntimeError):
    """Raised when a document cannot be ingested (usually missing metadata)."""


def parse_front_matter(raw: str) -> tuple[Dict[str, Any], str]:
    """Split a markdown file into (front matter dict, body)."""
    match = FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta = yaml.safe_load(match.group(1)) or {}
    if not isinstance(meta, dict):
        raise IngestError("front matter must be a YAML mapping")
    return meta, raw[match.end():]


def load_document(path: Path) -> Document:
    """Load one article, validating that its required metadata is present."""
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)

    missing = [key for key in REQUIRED_FRONT_MATTER if not str(meta.get(key, "")).strip()]
    if missing:
        raise IngestError(f"{path.name}: missing required front matter {missing}")

    title_match = H1_RE.search(body)
    title = str(meta.get("title") or (title_match.group(1) if title_match else path.stem)).strip()

    metadata = {str(k): _jsonable(v) for k, v in meta.items()}
    metadata["source_file"] = path.name
    metadata["title"] = title

    return Document(
        article_id=str(meta["article_id"]).strip(),
        source_file=path.name,
        path=str(path),
        title=title,
        text=body,
        metadata=metadata,
    )


def _jsonable(value: Any) -> Any:
    """Make front-matter values JSON-safe (PyYAML turns dates into date objects)."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def load_documents(paths: Iterable[Path]) -> List[Document]:
    """Load every markdown file under the given files/directories, sorted by name."""
    files: List[Path] = []
    for entry in paths:
        entry = Path(entry)
        if entry.is_dir():
            files.extend(sorted(entry.glob("*.md")))
        elif entry.is_file():
            files.append(entry)
        else:
            raise IngestError(f"path not found: {entry}")

    if not files:
        raise IngestError(f"no markdown documents found in {[str(p) for p in paths]}")

    documents = [load_document(path) for path in files]

    seen: Dict[str, str] = {}
    for doc in documents:
        if doc.article_id in seen:
            raise IngestError(
                f"duplicate article_id {doc.article_id} in {doc.source_file} and {seen[doc.article_id]}"
            )
        seen[doc.article_id] = doc.source_file
    return documents
