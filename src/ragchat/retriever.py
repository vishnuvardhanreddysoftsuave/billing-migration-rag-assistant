"""Search over an indexed namespace."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from .bm25 import BM25Index, reciprocal_rank_fusion
from .config import Config
from .embeddings import HashingTfidfEmbedder, build_embedder
from .models import Chunk, SearchHit
from .store import VectorStore, namespace_for

MODES = ("semantic", "hybrid")


class Retriever:
    """Embeds a question and returns the best chunks from one index namespace."""

    def __init__(
        self,
        store: VectorStore,
        embedder: HashingTfidfEmbedder,
        rrf_k: float = 10.0,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self._bm25: Optional[BM25Index] = None
        self._bm25_signature: Optional[int] = None

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
        return cls(
            store=VectorStore.load(index_dir, namespace),
            embedder=embedder,
            rrf_k=config.retrieval.rrf_k,
            bm25_k1=config.retrieval.bm25_k1,
            bm25_b=config.retrieval.bm25_b,
        )

    @property
    def namespace(self) -> str:
        return self.store.namespace

    def search(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        mode: str = "semantic",
    ) -> List[SearchHit]:
        """Retrieve the best chunks for a question.

        ``mode="semantic"`` (the default, and the only mode any Week 3 caller
        uses) is the cosine search over the hashed ``lnc.ltc`` vectors,
        unchanged. ``mode="hybrid"`` additionally ranks by BM25 keyword score
        and fuses the two rankings with reciprocal rank fusion — see
        ``eval/week4_writeup.md`` for why that recovers questions built around
        an exact error code that the semantic ranking alone pushes out of the
        top-3.
        """
        if mode not in MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}; use one of {MODES}")
        if mode == "hybrid":
            return self._search_hybrid(question, top_k=top_k, filters=filters)
        return self._search_semantic(question, top_k=top_k, filters=filters)

    def _search_semantic(
        self, question: str, top_k: int, filters: Optional[Dict[str, Any]]
    ) -> List[SearchHit]:
        query_vector = self.embedder.embed_query(question, self.store.df, self.store.n_units)
        ranked = self.store.search(query_vector, top_k=top_k, filters=filters)
        return [
            SearchHit(rank=rank, score=score, chunk=self.store.chunks[row])
            for rank, (row, score) in enumerate(ranked, start=1)
        ]

    def _search_hybrid(
        self, question: str, top_k: int, filters: Optional[Dict[str, Any]]
    ) -> List[SearchHit]:
        n_chunks = len(self.store.chunks)
        query_vector = self.embedder.embed_query(question, self.store.df, self.store.n_units)
        semantic_ranking = [row for row, _ in self.store.search(query_vector, top_k=n_chunks, filters=filters)]

        candidate_rows = self.store.filtered_rows(filters)
        keyword_ranking = [
            row for row, _ in self._bm25_index().rank(self.embedder.tokenize(question), rows=candidate_rows)
        ]

        fused = reciprocal_rank_fusion([semantic_ranking, keyword_ranking], k=self.rrf_k)[:top_k]
        return [
            SearchHit(rank=rank, score=score, chunk=self.store.chunks[row])
            for rank, (row, score) in enumerate(fused, start=1)
        ]

    def _bm25_index(self) -> BM25Index:
        """Build once and cache; the corpus is small enough to tokenize eagerly."""
        signature = len(self.store.chunks)
        if self._bm25 is None or self._bm25_signature != signature:
            tokenized = [self.embedder.tokenize(chunk.text) for chunk in self.store.chunks]
            self._bm25 = BM25Index.build(tokenized, k1=self.bm25_k1, b=self.bm25_b)
            self._bm25_signature = signature
        return self._bm25

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
