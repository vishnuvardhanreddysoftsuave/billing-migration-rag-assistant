"""Core data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Metadata every chunk must carry. A chunk missing any of these is a failed
# ingest and the indexer raises rather than storing it.
REQUIRED_METADATA = ("source_file", "article_id", "product_area", "last_updated")

# Metadata keys retrieval may be filtered on.
FILTERABLE_METADATA = ("product_area", "article_id", "source_file", "last_updated")


@dataclass(frozen=True)
class Document:
    """One help-centre article, front matter parsed out."""

    article_id: str
    source_file: str
    path: str
    title: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """An indexed unit of text plus the metadata that makes it citable."""

    chunk_id: str
    text: str
    strategy: str
    position: int
    char_start: int
    char_end: int
    section: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def article_id(self) -> str:
        return str(self.metadata.get("article_id", ""))

    @property
    def source_file(self) -> str:
        return str(self.metadata.get("source_file", ""))

    @property
    def product_area(self) -> str:
        return str(self.metadata.get("product_area", ""))

    @property
    def last_updated(self) -> str:
        return str(self.metadata.get("last_updated", ""))

    @property
    def n_chars(self) -> int:
        return len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Chunk":
        return cls(**payload)

    def citation_label(self) -> str:
        return f"{self.article_id} · {self.section}"


@dataclass(frozen=True)
class SearchHit:
    """A retrieved chunk with its rank and score."""

    rank: int
    score: float
    chunk: Chunk

    def to_dict(self) -> Dict[str, Any]:
        return {"rank": self.rank, "score": round(self.score, 6), "chunk": self.chunk.to_dict()}


@dataclass(frozen=True)
class Citation:
    """A claim-level citation that must resolve to a real indexed chunk."""

    chunk_id: str
    article_id: str
    source_file: str
    section: str
    resolved: bool
    quote: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Answer:
    """The end-to-end result of asking a question."""

    question: str
    refused: bool
    text: str
    citations: List[Citation] = field(default_factory=list)
    hits: List[SearchHit] = field(default_factory=list)
    backend: str = ""
    refusal_reason: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "refused": self.refused,
            "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "hits": [h.to_dict() for h in self.hits],
            "backend": self.backend,
            "refusal_reason": self.refusal_reason,
            "diagnostics": self.diagnostics,
        }
