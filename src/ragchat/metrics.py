"""Retrieval-quality metrics: hit-rate@k, recall@k, MRR.

All three take the same shape of input — one :class:`RetrievalRecord` per
question, holding the ranked ids actually retrieved and the set of ids that
would make the answer complete — so the difference between the metrics is
only in how they read that record, not in how it is built.

For a question with exactly one relevant chunk, hit-rate@k and recall@k are
numerically identical: either the chunk is in the top-k or it is not. They
diverge only on a question whose complete answer needs more than one chunk
(see ``also_needs`` in ``eval/week4_questions.yaml``), where hit-rate asks
"did retrieval succeed at all" and recall asks "how much of the answer did it
surface".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Set


@dataclass(frozen=True)
class RetrievalRecord:
    """One question's retrieval outcome, reduced to what the metrics need."""

    retrieved_ids: List[str]
    relevant_ids: Set[str] = field(default_factory=set)

    def rank_of_first_relevant(self, k: int | None = None) -> int | None:
        window = self.retrieved_ids[:k] if k else self.retrieved_ids
        for rank, chunk_id in enumerate(window, start=1):
            if chunk_id in self.relevant_ids:
                return rank
        return None


def hit_rate_at_k(records: Sequence[RetrievalRecord], k: int) -> float:
    """Share of questions with at least one relevant chunk in the top-k."""
    if not records:
        return 0.0
    hits = sum(1 for r in records if r.rank_of_first_relevant(k) is not None)
    return hits / len(records)


def recall_at_k(records: Sequence[RetrievalRecord], k: int) -> float:
    """Mean share of each question's relevant chunks found in the top-k."""
    if not records:
        return 0.0
    per_question = []
    for r in records:
        if not r.relevant_ids:
            continue
        found = len(set(r.retrieved_ids[:k]) & r.relevant_ids)
        per_question.append(found / len(r.relevant_ids))
    return sum(per_question) / len(per_question) if per_question else 0.0


def mrr(records: Sequence[RetrievalRecord], k: int | None = None) -> float:
    """Mean reciprocal rank of the first relevant chunk (0 if none within k)."""
    if not records:
        return 0.0
    total = 0.0
    for r in records:
        rank = r.rank_of_first_relevant(k)
        total += (1.0 / rank) if rank else 0.0
    return total / len(records)
