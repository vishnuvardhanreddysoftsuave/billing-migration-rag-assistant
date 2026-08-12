"""Deterministic, stateless text embeddings.

Documents are embedded with a hashing vectoriser using classic ``lnc`` weighting
(log term frequency, no idf, L2 normalised); queries use ``ltc`` (log term
frequency times idf, L2 normalised). Similarity is the dot product.

Two properties matter for this project:

* **Stateless.** No vocabulary is fitted, so appending a new drop to an existing
  index cannot shift the vectors already stored and never needs a re-index.
* **Strategy-independent.** The same embedding function is used for every
  chunking strategy and every chunk size, so a hit-rate difference between two
  runs is attributable to chunking alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

# Kept explicit (rather than sklearn's list) so every token in it survives the
# token pattern below; that avoids a preprocessing/stop-word mismatch warning.
STOP_WORDS = frozenset(
    """
    about above after again against all also am an and any are as at be because been
    before being below between both but by can cannot could did do does doing down
    during each few for from further had has have having he her here hers herself him
    himself his how if in into is it its itself me more most my myself no nor not of
    off on once only or other ought our ours ourselves out over own same she should so
    some such than that the their theirs them themselves then there these they this
    those through to too under until up very was we were what when where which while
    who whom why will with would you your yours yourself yourselves
    """.split()
)

# Keeps identifiers such as err-4032, 2026-06-01 and hmac-sha256 as single tokens.
TOKEN_PATTERN = r"(?u)\b\w[\w\-\.]+\b"


@dataclass(frozen=True)
class EmbeddingSpec:
    """Fingerprint of the embedding function, persisted alongside an index."""

    backend: str
    n_features: int
    ngram_min: int
    ngram_max: int
    lowercase: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "n_features": self.n_features,
            "ngram_min": self.ngram_min,
            "ngram_max": self.ngram_max,
            "lowercase": self.lowercase,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EmbeddingSpec":
        return cls(
            backend=str(payload["backend"]),
            n_features=int(payload["n_features"]),
            ngram_min=int(payload["ngram_min"]),
            ngram_max=int(payload["ngram_max"]),
            lowercase=bool(payload["lowercase"]),
        )


class HashingTfidfEmbedder:
    """Hashing vectoriser with lnc.ltc weighting."""

    backend = "hashing-tfidf"

    def __init__(
        self,
        n_features: int = 2**18,
        ngram_range: Sequence[int] = (1, 2),
        lowercase: bool = True,
    ) -> None:
        self.n_features = int(n_features)
        self.ngram_range = (int(ngram_range[0]), int(ngram_range[1]))
        self.lowercase = bool(lowercase)
        self._vectorizer = HashingVectorizer(
            n_features=self.n_features,
            ngram_range=self.ngram_range,
            lowercase=self.lowercase,
            token_pattern=TOKEN_PATTERN,
            stop_words=list(STOP_WORDS),
            alternate_sign=False,
            norm=None,
            dtype=np.float32,
        )

    @property
    def spec(self) -> EmbeddingSpec:
        return EmbeddingSpec(
            backend=self.backend,
            n_features=self.n_features,
            ngram_min=self.ngram_range[0],
            ngram_max=self.ngram_range[1],
            lowercase=self.lowercase,
        )

    def counts(self, texts: Iterable[str]) -> sp.csr_matrix:
        return self._vectorizer.transform(list(texts)).tocsr()

    def embed_documents(self, texts: Iterable[str]) -> sp.csr_matrix:
        """lnc: 1 + log(tf), L2 normalised. Returns one row per text."""
        matrix = self.counts(texts)
        return _l2_normalize(_log_tf(matrix))

    def document_frequencies(self, texts: Iterable[str]) -> np.ndarray:
        """Per-feature document counts, for the query-side idf."""
        matrix = self.counts(texts)
        binary = matrix.copy()
        binary.data = np.ones_like(binary.data)
        return np.asarray(binary.sum(axis=0)).ravel().astype(np.float32)

    def embed_query(self, text: str, df: np.ndarray, n_documents: int) -> sp.csr_matrix:
        """ltc: 1 + log(tf), times idf, L2 normalised."""
        vector = _log_tf(self.counts([text]))
        if vector.nnz:
            idf = np.log((1.0 + n_documents) / (1.0 + df[vector.indices])) + 1.0
            vector.data = vector.data * idf.astype(np.float32)
        return _l2_normalize(vector)

    def tokenize(self, text: str) -> List[str]:
        """Content tokens of a string, using the same analyzer as the vectors."""
        analyzer = self._vectorizer.build_analyzer()
        return [token for token in analyzer(text) if " " not in token]

    def feature_index(self, token: str) -> int | None:
        """Hashed column for a single token, or None if it is not a content token."""
        vector = self.counts([token])
        if vector.nnz != 1:
            return None
        return int(vector.indices[0])


def _log_tf(matrix: sp.csr_matrix) -> sp.csr_matrix:
    out = matrix.copy().astype(np.float32)
    if out.nnz:
        out.data = (1.0 + np.log(out.data)).astype(np.float32)
    return out


def _l2_normalize(matrix: sp.csr_matrix) -> sp.csr_matrix:
    out = matrix.tocsr(copy=True)
    norms = np.sqrt(np.asarray(out.multiply(out).sum(axis=1)).ravel())
    norms[norms == 0.0] = 1.0
    scale = sp.diags((1.0 / norms).astype(np.float32))
    return (scale @ out).tocsr()


def build_embedder(config: Any) -> HashingTfidfEmbedder:
    """Construct the embedder named by an :class:`~ragchat.config.EmbeddingConfig`."""
    if config.backend != HashingTfidfEmbedder.backend:
        raise ValueError(
            f"unknown embedding backend {config.backend!r}; available: {HashingTfidfEmbedder.backend!r}"
        )
    return HashingTfidfEmbedder(
        n_features=config.n_features,
        ngram_range=config.ngram_range,
        lowercase=config.lowercase,
    )
