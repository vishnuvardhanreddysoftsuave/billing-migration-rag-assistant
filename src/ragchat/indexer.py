"""Ingest pipeline: documents -> chunks -> vectors -> store.

Ingest is **append-only**. Adding a new drop of articles never recomputes the
vectors already in the index, because the embedding function is stateless.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .chunkers import get_chunker
from .config import Config
from .embeddings import build_embedder
from .loader import load_documents
from .models import Chunk, Document
from .store import IngestEvent, VectorStore, namespace_for


@dataclass
class IngestReport:
    namespace: str
    label: str
    source_files: List[str]
    n_documents: int
    n_chunks_added: int
    n_chunks_total: int
    seconds: float
    index_path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "namespace": self.namespace,
            "label": self.label,
            "source_files": self.source_files,
            "n_documents": self.n_documents,
            "n_chunks_added": self.n_chunks_added,
            "n_chunks_total": self.n_chunks_total,
            "seconds": round(self.seconds, 3),
            "index_path": self.index_path,
        }


def chunk_documents(documents: Sequence[Document], strategy: str, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    chunker = get_chunker(strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: List[Chunk] = []
    for document in documents:
        chunks.extend(chunker.split(document))
    return chunks


def ingest(
    config: Config,
    paths: Sequence[Path],
    strategy: str | None = None,
    label: str = "ingest",
    index_dir: Path | None = None,
) -> IngestReport:
    """Chunk, embed and append the given documents to the index."""
    strategy = strategy or config.chunking.default_strategy
    index_dir = Path(index_dir) if index_dir else config.paths.index_dir
    namespace = namespace_for(strategy, config.chunking.chunk_size, config.chunking.chunk_overlap)

    started = time.perf_counter()
    documents = load_documents(paths)
    chunks = chunk_documents(
        documents,
        strategy=strategy,
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
    )

    embedder = build_embedder(config.embedding)
    texts = [chunk.text for chunk in chunks]
    vectors = embedder.embed_documents(texts)
    df_delta = embedder.document_frequencies(texts)

    store = VectorStore.load_or_create(index_dir, namespace, embedder.spec)
    elapsed = time.perf_counter() - started
    event = IngestEvent(
        at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        label=label,
        source_files=[doc.source_file for doc in documents],
        n_documents=len(documents),
        n_chunks=len(chunks),
        seconds=elapsed,
    )
    store.add(chunks=chunks, vectors=vectors, df_delta=df_delta, event=event)
    index_path = store.save(index_dir)

    return IngestReport(
        namespace=namespace,
        label=label,
        source_files=event.source_files,
        n_documents=len(documents),
        n_chunks_added=len(chunks),
        n_chunks_total=len(store.chunks),
        seconds=elapsed,
        index_path=str(index_path),
    )
