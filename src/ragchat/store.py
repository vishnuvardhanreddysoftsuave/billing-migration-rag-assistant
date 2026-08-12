"""On-disk vector store: sparse vectors plus chunk metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .embeddings import EmbeddingSpec
from .models import Chunk, REQUIRED_METADATA

VECTORS_FILE = "vectors.npz"
CHUNKS_FILE = "chunks.jsonl"
STATS_FILE = "stats.npz"
META_FILE = "meta.json"


class StoreError(RuntimeError):
    """Raised on a corrupt or incompatible index."""


@dataclass
class IngestEvent:
    """One append to the index — the audit trail for what was (re)indexed."""

    at: str
    label: str
    source_files: List[str]
    n_documents: int
    n_chunks: int
    seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "at": self.at,
            "label": self.label,
            "source_files": self.source_files,
            "n_documents": self.n_documents,
            "n_chunks": self.n_chunks,
            "seconds": round(self.seconds, 3),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "IngestEvent":
        return cls(
            at=payload["at"],
            label=payload["label"],
            source_files=list(payload["source_files"]),
            n_documents=int(payload["n_documents"]),
            n_chunks=int(payload["n_chunks"]),
            seconds=float(payload["seconds"]),
        )


class VectorStore:
    """Append-only store of chunk vectors and their metadata."""

    def __init__(self, namespace: str, spec: EmbeddingSpec) -> None:
        self.namespace = namespace
        self.spec = spec
        self.chunks: List[Chunk] = []
        self.matrix: Optional[sp.csr_matrix] = None
        self.df = np.zeros(spec.n_features, dtype=np.float32)
        # Number of indexed chunks: the collection size N used by the query-side idf.
        self.n_units = 0
        self.history: List[IngestEvent] = []
        self._by_id: Dict[str, int] = {}

    # -- construction ---------------------------------------------------

    def add(
        self,
        chunks: Sequence[Chunk],
        vectors: sp.csr_matrix,
        df_delta: np.ndarray,
        event: IngestEvent,
    ) -> None:
        if len(chunks) != vectors.shape[0]:
            raise StoreError("chunk count and vector count disagree")
        if vectors.shape[1] != self.spec.n_features:
            raise StoreError("vector width does not match the embedding spec")

        incoming: set[str] = set()
        for chunk in chunks:
            missing = [key for key in REQUIRED_METADATA if not str(chunk.metadata.get(key, "")).strip()]
            if missing:
                raise StoreError(f"chunk {chunk.chunk_id} is missing required metadata {missing}")
            # Duplicates must be caught both against the stored index and within this batch.
            if chunk.chunk_id in self._by_id or chunk.chunk_id in incoming:
                raise StoreError(f"duplicate chunk_id {chunk.chunk_id}")
            incoming.add(chunk.chunk_id)

        start = len(self.chunks)
        for offset, chunk in enumerate(chunks):
            self._by_id[chunk.chunk_id] = start + offset
        self.chunks.extend(chunks)
        self.matrix = vectors.tocsr() if self.matrix is None else sp.vstack([self.matrix, vectors]).tocsr()
        self.df = self.df + df_delta.astype(np.float32)
        self.n_units += len(chunks)
        self.history.append(event)

    # -- query ----------------------------------------------------------

    def search(
        self,
        query_vector: sp.csr_matrix,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[int, float]]:
        """Return (row index, score) for the ``top_k`` best chunks.

        ``filters`` maps a metadata key to an accepted value (or list of values).
        Filtering is applied to the candidate set before ranking, so a filtered
        search can return a different top-1, not merely a shorter list.
        """
        if self.matrix is None or not self.chunks:
            return []
        scores = np.asarray((self.matrix @ query_vector.T).todense()).ravel()
        if filters:
            keep = np.array([_matches(chunk, filters) for chunk in self.chunks], dtype=bool)
            scores = np.where(keep, scores, 0.0)
        return self._rank(scores, top_k)

    def distinct_values(self, key: str) -> List[str]:
        """Sorted distinct values of a metadata key across the index."""
        return sorted({str(chunk.metadata.get(key, "")) for chunk in self.chunks if chunk.metadata.get(key)})

    def _rank(self, scores: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        eligible = np.flatnonzero(scores > 0.0)
        if eligible.size == 0:
            return []
        k = min(top_k, eligible.size)
        top = eligible[np.argpartition(-scores[eligible], k - 1)[:k]]
        # Stable, deterministic ordering: score desc, then row index asc.
        order = sorted(top.tolist(), key=lambda i: (-float(scores[i]), i))
        return [(int(i), float(scores[i])) for i in order]

    def get(self, chunk_id: str) -> Optional[Chunk]:
        idx = self._by_id.get(chunk_id)
        return self.chunks[idx] if idx is not None else None

    def stats(self) -> Dict[str, Any]:
        sizes = [chunk.n_chars for chunk in self.chunks]
        return {
            "namespace": self.namespace,
            "n_chunks": len(self.chunks),
            "n_articles": len({chunk.article_id for chunk in self.chunks}),
            "mean_chunk_chars": round(float(np.mean(sizes)), 1) if sizes else 0.0,
            "max_chunk_chars": max(sizes) if sizes else 0,
            "embedding": self.spec.to_dict(),
            "history": [event.to_dict() for event in self.history],
        }

    # -- persistence ----------------------------------------------------

    def save(self, index_dir: Path) -> Path:
        target = Path(index_dir) / self.namespace
        target.mkdir(parents=True, exist_ok=True)

        matrix = self.matrix if self.matrix is not None else sp.csr_matrix((0, self.spec.n_features), dtype=np.float32)
        sp.save_npz(target / VECTORS_FILE, matrix)
        np.savez_compressed(target / STATS_FILE, df=self.df)
        with (target / CHUNKS_FILE).open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        meta = {
            "namespace": self.namespace,
            "embedding": self.spec.to_dict(),
            "n_units": self.n_units,
            "n_chunks": len(self.chunks),
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "history": [event.to_dict() for event in self.history],
        }
        (target / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, index_dir: Path, namespace: str) -> "VectorStore":
        target = Path(index_dir) / namespace
        meta_path = target / META_FILE
        if not meta_path.is_file():
            raise StoreError(f"no index at {target}; run `ingest` first")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        store = cls(namespace=namespace, spec=EmbeddingSpec.from_dict(meta["embedding"]))
        store.matrix = sp.load_npz(target / VECTORS_FILE).tocsr()
        store.df = np.load(target / STATS_FILE)["df"].astype(np.float32)
        store.n_units = int(meta["n_units"])
        store.history = [IngestEvent.from_dict(item) for item in meta.get("history", [])]

        with (target / CHUNKS_FILE).open("r", encoding="utf-8") as fh:
            store.chunks = [Chunk.from_dict(json.loads(line)) for line in fh if line.strip()]
        store._by_id = {chunk.chunk_id: i for i, chunk in enumerate(store.chunks)}

        if store.matrix.shape[0] != len(store.chunks):
            raise StoreError(f"index at {target} is corrupt: vector/chunk count mismatch")
        return store

    @classmethod
    def load_or_create(cls, index_dir: Path, namespace: str, spec: EmbeddingSpec) -> "VectorStore":
        try:
            store = cls.load(index_dir, namespace)
        except StoreError:
            return cls(namespace=namespace, spec=spec)
        if store.spec != spec:
            raise StoreError(
                f"index {namespace} was built with embedding {store.spec.to_dict()} "
                f"but the current config asks for {spec.to_dict()}; delete the index or restore the config"
            )
        return store


def _matches(chunk: Chunk, filters: Dict[str, Any]) -> bool:
    """True when a chunk satisfies every filter (case-insensitive, value or list)."""
    for key, wanted in filters.items():
        if wanted is None or wanted == "":
            continue
        actual = str(chunk.metadata.get(key, "")).strip().lower()
        accepted = wanted if isinstance(wanted, (list, tuple, set)) else [wanted]
        if actual not in {str(value).strip().lower() for value in accepted}:
            return False
    return True


def namespace_for(strategy: str, chunk_size: int, chunk_overlap: int) -> str:
    """One index per (strategy, size) pair so sweeps never overwrite each other."""
    return f"{strategy}__cs{chunk_size}_ov{chunk_overlap}"
