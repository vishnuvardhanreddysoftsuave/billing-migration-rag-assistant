"""Baseline chunker: recursive character splitting, structure-agnostic.

This is the chunker the Week 3 app already shipped with. It walks a separator
hierarchy (blank line, newline, space, character) and greedily packs the pieces
into fixed-size windows with an overlap. It knows nothing about markdown tables,
so a long troubleshooting table gets cut wherever the character budget runs out.
"""

from __future__ import annotations

from typing import List, Tuple

from ..models import Chunk, Document
from .base import Chunker, section_index

SEPARATORS: Tuple[str, ...] = ("\n\n", "\n", " ", "")

Span = Tuple[int, int]


class BaselineChunker(Chunker):
    name = "baseline"

    def split(self, document: Document) -> List[Chunk]:
        text = document.text
        index = section_index(text)
        pieces = _atomic_spans(text, 0, self.chunk_size, SEPARATORS)
        spans = _merge_spans(pieces, self.chunk_size, self.chunk_overlap)

        chunks: List[Chunk] = []
        for position, (start, end) in enumerate(spans):
            body = text[start:end].strip()
            if not body:
                continue
            chunks.append(
                self.build_chunk(
                    document=document,
                    text=body,
                    position=len(chunks),
                    char_start=start,
                    char_end=end,
                    index=index,
                )
            )
        return chunks


def _atomic_spans(text: str, base: int, size: int, separators: Tuple[str, ...]) -> List[Span]:
    """Break text into pieces that are each <= size where the separators allow."""
    if len(text) <= size:
        return [(base, base + len(text))] if text.strip() else []

    if not separators:
        return [(base + i, base + min(i + size, len(text))) for i in range(0, len(text), size)]

    separator, rest = separators[0], separators[1:]
    if separator == "":
        return [(base + i, base + min(i + size, len(text))) for i in range(0, len(text), size)]
    if separator not in text:
        return _atomic_spans(text, base, size, rest)

    spans: List[Span] = []
    cursor = 0
    for part in text.split(separator):
        start, end = cursor, cursor + len(part)
        cursor = end + len(separator)
        if not part.strip():
            continue
        if len(part) > size:
            spans.extend(_atomic_spans(part, base + start, size, rest))
        else:
            spans.append((base + start, base + end))
    return spans


def _merge_spans(pieces: List[Span], size: int, overlap: int) -> List[Span]:
    """Greedily pack pieces into windows of at most ``size`` chars with overlap."""
    chunks: List[Span] = []
    current: List[Span] = []
    current_len = 0

    for piece in pieces:
        piece_len = piece[1] - piece[0]
        if current and current_len + piece_len > size:
            chunks.append((current[0][0], current[-1][1]))
            tail: List[Span] = []
            tail_len = 0
            for span in reversed(current):
                if tail_len >= overlap:
                    break
                tail.insert(0, span)
                tail_len += span[1] - span[0]
            current, current_len = tail, tail_len
        current.append(piece)
        current_len += piece_len

    if current:
        chunks.append((current[0][0], current[-1][1]))
    return chunks
