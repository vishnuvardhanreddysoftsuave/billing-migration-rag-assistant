"""BM25 keyword scoring over the indexed chunks.

Computed at query time from the same token stream the embedder already
produces (:meth:`~ragchat.embeddings.HashingTfidfEmbedder.tokenize`), so an
identifier like ``err-4032`` survives as one token here exactly as it does in
the semantic vectors. That shared tokenizer is what makes the two rankers
comparable: the difference between them is the *weighting*, not the
vocabulary.

The weighting difference is the point. The semantic vectors are L2-normalised
(``lnc.ltc``), so a chunk's every term is divided by the norm of *all* the
chunk's terms — a long troubleshooting-table chunk with many distinct error
codes dilutes each individual code's contribution. BM25's length penalty
(the ``b`` parameter) is far gentler, and its term score is a plain sum of
idf-weighted, frequency-saturating contributions, so one rare, highly
distinctive query term (an error code that appears nowhere else in the
corpus) can dominate the score regardless of how many other terms share the
chunk. That is exactly the property a hybrid search needs for queries built
around an exact code, name or id.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class BM25Index:
    """Okapi BM25 over a fixed list of tokenized documents (one per chunk row)."""

    term_counts: List[Counter]
    doc_lengths: List[int]
    doc_freq: Dict[str, int]
    n_docs: int
    avg_doc_length: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, tokenized_docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        term_counts = [Counter(tokens) for tokens in tokenized_docs]
        doc_lengths = [len(tokens) for tokens in tokenized_docs]
        n_docs = len(tokenized_docs)
        doc_freq: Dict[str, int] = {}
        for counts in term_counts:
            for term in counts:
                doc_freq[term] = doc_freq.get(term, 0) + 1
        avg_length = (sum(doc_lengths) / n_docs) if n_docs else 0.0
        return cls(
            term_counts=term_counts,
            doc_lengths=doc_lengths,
            doc_freq=doc_freq,
            n_docs=n_docs,
            avg_doc_length=avg_length,
            k1=k1,
            b=b,
        )

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        # Robertson-Sparck Jones idf, floored so a term in every document never
        # goes negative and silently subtracts from the score.
        return max(math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0), 1e-9)

    def rank(self, query_tokens: Sequence[str], rows: Optional[Sequence[int]] = None) -> List[Tuple[int, float]]:
        """Rows with a positive BM25 score, best first, restricted to ``rows`` if given.

        Ties break on row index so results are deterministic — the same
        contract :meth:`ragchat.store.VectorStore._rank` makes for cosine scores.
        """
        if not query_tokens or self.avg_doc_length == 0:
            return []
        candidates = range(self.n_docs) if rows is None else rows
        query_counts = Counter(query_tokens)
        scored: List[Tuple[int, float]] = []
        for row in candidates:
            counts = self.term_counts[row]
            length = self.doc_lengths[row]
            total = 0.0
            for term, qtf in query_counts.items():
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf(term)
                denom = tf + self.k1 * (1 - self.b + self.b * length / self.avg_doc_length)
                total += idf * (tf * (self.k1 + 1)) / denom
            if total > 0:
                scored.append((row, total))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: float = 10.0) -> List[Tuple[int, float]]:
    """Fuse several rank-ordered row lists into one score per row (RRF).

    RRF only uses each ranker's *order*, never its raw score, which is what
    makes it safe to combine cosine similarity with a BM25 score even though
    the two live on unrelated scales. ``k`` dampens the influence of rank:
    the standard web-search default is 60, tuned for corpora of thousands of
    documents; a collection of a few dozen chunks needs a much smaller
    constant or the top ranks of every source end up scored almost
    identically (see ``config.yaml`` for the value this project ships).
    """
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for position, row in enumerate(ranking, start=1):
            fused[row] = fused.get(row, 0.0) + 1.0 / (k + position)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))
