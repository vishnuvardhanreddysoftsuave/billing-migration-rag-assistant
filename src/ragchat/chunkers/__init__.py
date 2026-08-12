"""Chunking strategies, addressable by name."""

from __future__ import annotations

from typing import Dict, Type

from .base import Chunker
from .baseline import BaselineChunker
from .structure_aware import StructureAwareChunker

_REGISTRY: Dict[str, Type[Chunker]] = {
    BaselineChunker.name: BaselineChunker,
    StructureAwareChunker.name: StructureAwareChunker,
}


def available_strategies() -> list[str]:
    return sorted(_REGISTRY)


def get_chunker(name: str, chunk_size: int, chunk_overlap: int) -> Chunker:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown chunking strategy {name!r}; available: {available_strategies()}") from None
    return cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


__all__ = [
    "Chunker",
    "BaselineChunker",
    "StructureAwareChunker",
    "available_strategies",
    "get_chunker",
]
