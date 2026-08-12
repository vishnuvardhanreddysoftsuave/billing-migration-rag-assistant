"""Structure-aware chunker: markdown blocks in, table rows never orphaned.

The baseline chunker packs characters and cannot see that a troubleshooting table
is a table. When a table runs past the character budget it gets cut mid-table, and
every row after the cut is stored without the header that says which column is the
error code, which is the cause and which is the fix.

This chunker parses the document into blocks first and then guarantees:

1. **A table row is never separated from its header row.** A table that does not
   fit in one chunk is split between rows, and every resulting chunk repeats the
   header row and its separator.
2. **A row is never split down the middle.** If a single row plus its header
   exceeds the chunk size the row is emitted whole and oversized, because half a
   row is worse than a large chunk.
3. **Chunks do not span section boundaries**, so unrelated sections never share a
   chunk, and each chunk is prefixed with its heading breadcrumb so a bare table
   row still carries the context of the section it came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ..models import Chunk, Document
from .base import Chunker, section_index

HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\|[\s:|-]+\|$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Block:
    """A markdown block: heading, paragraph or table."""

    kind: str
    start: int
    end: int
    text: str
    level: int = 0
    title: str = ""
    header: List[str] = field(default_factory=list)
    rows: List[Tuple[int, int, str]] = field(default_factory=list)


@dataclass
class Unit:
    """An atomic piece of text that a chunk may contain."""

    text: str
    start: int
    end: int
    kind: str
    atomic: bool = False
    n_rows: int = 0
    header_repeated: bool = False


class StructureAwareChunker(Chunker):
    name = "structure-aware"

    def split(self, document: Document) -> List[Chunk]:
        text = document.text
        index = section_index(text)
        blocks = parse_blocks(text)

        chunks: List[Chunk] = []
        for heading_path, group in _group_by_section(blocks):
            prefix = f"{heading_path}\n\n" if heading_path else ""
            budget = max(self.chunk_size - len(prefix), 200)

            units: List[Unit] = []
            for block in group:
                if block.kind == "table":
                    units.extend(_table_units(block, budget))
                else:
                    units.append(Unit(block.text, block.start, block.end, block.kind))

            for unit_group in _pack(units, budget, self.chunk_overlap):
                body = "\n\n".join(u.text for u in unit_group).strip()
                if not body:
                    continue
                table_units = [u for u in unit_group if u.kind == "table"]
                chunks.append(
                    self.build_chunk(
                        document=document,
                        text=prefix + body,
                        position=len(chunks),
                        char_start=min(u.start for u in unit_group),
                        char_end=max(u.end for u in unit_group),
                        index=index,
                        extra_metadata={
                            "contains_table": bool(table_units),
                            "table_rows": sum(u.n_rows for u in table_units),
                            "table_header_repeated": any(u.header_repeated for u in table_units),
                            "heading_prefixed": bool(prefix),
                        },
                    )
                )
        return chunks


# --------------------------------------------------------------------------
# Block parsing
# --------------------------------------------------------------------------


def parse_blocks(text: str) -> List[Block]:
    """Split markdown into heading / table / paragraph blocks, keeping offsets."""
    lines = text.split("\n")
    offsets: List[int] = []
    position = 0
    for line in lines:
        offsets.append(position)
        position += len(line) + 1

    blocks: List[Block] = []
    i, n = 0, len(lines)
    while i < n:
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue

        heading = HEADING_LINE_RE.match(stripped)
        if heading:
            blocks.append(
                Block(
                    kind="heading",
                    start=offsets[i],
                    end=offsets[i] + len(lines[i]),
                    text=lines[i].strip(),
                    level=len(heading.group(1)),
                    title=heading.group(2).strip(),
                )
            )
            i += 1
            continue

        if stripped.startswith("|"):
            j = i
            while j < n and lines[j].strip().startswith("|"):
                j += 1
            blocks.append(_table_block(lines, offsets, i, j))
            i = j
            continue

        j = i
        while j < n and lines[j].strip() and not lines[j].strip().startswith(("|", "#")):
            j += 1
        blocks.append(
            Block(
                kind="paragraph",
                start=offsets[i],
                end=offsets[j - 1] + len(lines[j - 1]),
                text="\n".join(line.strip() for line in lines[i:j]),
            )
        )
        i = j
    return blocks


def _table_block(lines: Sequence[str], offsets: Sequence[int], start: int, stop: int) -> Block:
    header: List[str] = []
    rows: List[Tuple[int, int, str]] = []
    body_start = start

    if stop - start >= 2 and TABLE_SEPARATOR_RE.match(lines[start + 1].strip()):
        header = [lines[start].strip(), lines[start + 1].strip()]
        body_start = start + 2

    for k in range(body_start, stop):
        row = lines[k].strip()
        if not row or TABLE_SEPARATOR_RE.match(row):
            continue
        rows.append((offsets[k], offsets[k] + len(lines[k]), row))

    return Block(
        kind="table",
        start=offsets[start],
        end=offsets[stop - 1] + len(lines[stop - 1]),
        text="\n".join(line.strip() for line in lines[start:stop]),
        header=header,
        rows=rows,
    )


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def _group_by_section(blocks: Sequence[Block]) -> List[Tuple[str, List[Block]]]:
    """Group content blocks under their heading breadcrumb."""
    groups: List[Tuple[str, List[Block]]] = []
    stack: List[Tuple[int, str]] = []
    current: List[Block] = []
    breadcrumb = ""

    for block in blocks:
        if block.kind == "heading":
            if current:
                groups.append((breadcrumb, current))
                current = []
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.title))
            breadcrumb = " > ".join(title for _level, title in stack)
            continue
        current.append(block)

    if current:
        groups.append((breadcrumb, current))
    return groups


def _table_units(block: Block, budget: int) -> List[Unit]:
    """Turn a table into units, repeating the header whenever it must be split."""
    if not block.rows:
        return [Unit(block.text, block.start, block.end, "table", atomic=True)]

    header_text = "\n".join(block.header)
    header_len = len(header_text) + 1 if header_text else 0

    if len(block.text) <= budget:
        return [
            Unit(
                text=block.text,
                start=block.start,
                end=block.end,
                kind="table",
                atomic=True,
                n_rows=len(block.rows),
                header_repeated=False,
            )
        ]

    units: List[Unit] = []
    batch: List[Tuple[int, int, str]] = []
    batch_len = header_len

    def flush() -> None:
        nonlocal batch, batch_len
        if not batch:
            return
        body = "\n".join(row for _s, _e, row in batch)
        text = f"{header_text}\n{body}" if header_text else body
        units.append(
            Unit(
                text=text,
                start=batch[0][0],
                end=batch[-1][1],
                kind="table",
                atomic=True,
                n_rows=len(batch),
                header_repeated=bool(header_text) and bool(units),
            )
        )
        batch, batch_len = [], header_len

    for row in block.rows:
        row_len = len(row[2]) + 1
        # A row is never split: if it does not fit it is emitted whole, oversized.
        if batch and batch_len + row_len > budget:
            flush()
        batch.append(row)
        batch_len += row_len
    flush()
    return units


def _pack(units: Sequence[Unit], budget: int, overlap: int) -> List[List[Unit]]:
    """Pack units into chunks; atomic units (tables) always stand alone."""
    packed: List[List[Unit]] = []
    current: List[Unit] = []
    current_len = 0

    for unit in units:
        if unit.atomic:
            if current:
                packed.append(current)
                current, current_len = [], 0
            packed.append([unit])
            continue

        pieces = _split_prose(unit, budget)
        for piece in pieces:
            piece_len = len(piece.text) + 2
            if current and current_len + piece_len > budget:
                packed.append(current)
                current, current_len = _overlap_tail(current, overlap)
            current.append(piece)
            current_len += piece_len

    if current:
        packed.append(current)
    return packed


def _split_prose(unit: Unit, budget: int) -> List[Unit]:
    """Split an oversized paragraph on sentence boundaries."""
    if len(unit.text) <= budget:
        return [unit]

    pieces: List[Unit] = []
    cursor = unit.start
    buffer: List[str] = []
    buffer_len = 0

    for sentence in SENTENCE_RE.split(unit.text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if buffer and buffer_len + len(sentence) + 1 > budget:
            body = " ".join(buffer)
            pieces.append(Unit(body, cursor, min(cursor + len(body), unit.end), unit.kind))
            cursor = min(cursor + len(body) + 1, unit.end)
            buffer, buffer_len = [], 0
        buffer.append(sentence)
        buffer_len += len(sentence) + 1

    if buffer:
        body = " ".join(buffer)
        pieces.append(Unit(body, cursor, unit.end, unit.kind))
    return pieces


def _overlap_tail(current: Sequence[Unit], overlap: int) -> Tuple[List[Unit], int]:
    """Carry trailing prose into the next chunk, up to the overlap budget."""
    if overlap <= 0:
        return [], 0
    tail: List[Unit] = []
    tail_len = 0
    for unit in reversed(current):
        if tail_len >= overlap:
            break
        tail.insert(0, unit)
        tail_len += len(unit.text) + 2
    return tail, tail_len
