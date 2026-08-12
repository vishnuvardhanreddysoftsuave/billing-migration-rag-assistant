"""Chunker interface and the section bookkeeping every chunker shares."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple

from ..models import Chunk, Document

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def section_index(text: str) -> List[Tuple[int, int, str]]:
    """Return (offset, level, title) for every markdown heading, in order."""
    return [(m.start(), len(m.group(1)), m.group(2).strip()) for m in HEADING_RE.finditer(text)]


def section_for(offset: int, index: Sequence[Tuple[int, int, str]]) -> str:
    """The most recent heading at or before ``offset`` ("" when there is none)."""
    title = ""
    for pos, _level, heading in index:
        if pos <= offset:
            title = heading
        else:
            break
    return title


def heading_path_for(offset: int, index: Sequence[Tuple[int, int, str]]) -> str:
    """Breadcrumb of the enclosing headings, e.g. ``Title > Section``."""
    stack: List[Tuple[int, str]] = []
    for pos, level, heading in index:
        if pos > offset:
            break
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
    return " > ".join(title for _level, title in stack)


class Chunker(ABC):
    """Splits a :class:`Document` into indexable :class:`Chunk` objects."""

    name: str = "base"

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split(self, document: Document) -> List[Chunk]:
        """Split one document. Implementations must set every required metadata key."""

    # -- shared helpers -------------------------------------------------

    def build_chunk(
        self,
        document: Document,
        text: str,
        position: int,
        char_start: int,
        char_end: int,
        index: Sequence[Tuple[int, int, str]],
        extra_metadata: Dict[str, Any] | None = None,
    ) -> Chunk:
        metadata: Dict[str, Any] = dict(document.metadata)
        metadata.update(
            {
                "article_id": document.article_id,
                "source_file": document.source_file,
                "chunking_strategy": self.name,
                "heading_path": heading_path_for(char_start, index),
            }
        )
        if extra_metadata:
            metadata.update(extra_metadata)

        return Chunk(
            chunk_id=f"{document.article_id}#{self.name}-{position:04d}",
            text=text,
            strategy=self.name,
            position=position,
            char_start=char_start,
            char_end=char_end,
            section=section_for(char_start, index),
            metadata=metadata,
        )
