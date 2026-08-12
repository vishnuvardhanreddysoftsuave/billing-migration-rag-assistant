"""Search over an indexed namespace."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config
from .embeddings import HashingTfidfEmbedder, build_embedder
from .models import Chunk, SearchHit
from .store import VectorStore, namespace_for


class Retriever:
    """Embeds a question and returns the best chunks from one index namespace."""

    def __init__(self, store: VectorStore, embedder: HashingTfidfEmbedder) -> None:
        self.store = store
        self.embedder = embedder

    @classmethod
    def open(
        cls,
        config: Config,
        strategy: str | None = None,
        index_dir: Path | None = None,
    ) -> "Retriever":
        strategy = strategy or config.chunking.default_strategy
        index_dir = Path(index_dir) if index_dir else config.paths.index_dir
        namespace = namespace_for(strategy, config.chunking.chunk_size, config.chunking.chunk_overlap)
        embedder = build_embedder(config.embedding)
        return cls(store=VectorStore.load(index_dir, namespace), embedder=embedder)

    @property
    def namespace(self) -> str:
        return self.store.namespace

    def search(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        query_vector = self.embedder.embed_query(question, self.store.df, self.store.n_units)
        ranked = self.store.search(query_vector, top_k=top_k, filters=filters)
        return [
            SearchHit(rank=rank, score=score, chunk=self.store.chunks[row])
            for rank, (row, score) in enumerate(ranked, start=1)
        ]

    def distinct_values(self, key: str) -> List[str]:
        return self.store.distinct_values(key)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self.store.get(chunk_id)

    def idf(self, term: str) -> float:
        """Inverse document frequency of a term over the indexed chunks.

        Terms the corpus has never seen get the maximum weight, which is what
        makes the evidence gate notice that a question is about something the
        help centre simply does not cover.
        """
        index = self.embedder.feature_index(term)
        df = float(self.store.df[index]) if index is not None else 0.0
        return math.log((1.0 + self.store.n_units) / (1.0 + df)) + 1.0
