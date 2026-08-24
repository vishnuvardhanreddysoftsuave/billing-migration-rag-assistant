"""Week 4 Task Set A: label failures, then measure the one retrieval fix.

Where :mod:`ragchat.evaluation` asks "did chunking preserve the answer", this
module asks a different question of the same kind of index: for a question
that is already known to fail, is the failure *retrieval* (the wrong chunk
was fetched) or *generation* (the right chunk was fetched and the pipeline
still got it wrong)? Those need different fixes — a better retriever cannot
repair a generation failure, and a better generator cannot repair a retrieval
failure — which is why every question here runs through both a semantic-only
and a hybrid retrieval pass, and through the actual pipeline, before a label
is assigned. See ``eval/week4_questions.yaml`` for the question set and its
provenance, and ``eval/week4_writeup.md`` for the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from .config import Config
from .evaluation import _chunk_supports, ensure_index
from .metrics import RetrievalRecord, hit_rate_at_k, mrr, recall_at_k
from .models import Answer, SearchHit
from .pipeline import RAGPipeline

K = 3  # the hit-rate@k this week's task is scored on
WINDOW = 10  # how far to look before calling a chunk "not retrieved at all"


def load_week4_questions(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    if not payload or "questions" not in payload:
        raise ValueError(f"{path} does not look like a week4 question set")
    return list(payload["questions"])


def _facts(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The primary gold fact plus any ``also_needs`` facts, primary first."""
    primary = {"article_id": item["gold"]["article_id"], "must_contain": list(item.get("must_contain", []))}
    extra = [
        {"article_id": f["article_id"], "must_contain": list(f["must_contain"])}
        for f in item.get("also_needs", [])
    ]
    return [primary] + extra


def _rank_of_fact(hits: Sequence[SearchHit], fact: Dict[str, Any]) -> Optional[int]:
    for hit in hits:
        if hit.chunk.article_id == fact["article_id"] and _chunk_supports(hit.chunk.text, fact["must_contain"]):
            return hit.rank
    return None


def _relevant_ids(all_chunks: Sequence, facts: Sequence[Dict[str, Any]]) -> set:
    """The true relevant chunk ids for a fact set, found over the *whole* index.

    Deliberately independent of what any particular search returned: if the
    relevant chunk exists in the corpus but fell outside a ranker's top-k, it
    must still count as a miss for that ranker, not be silently excluded from
    the average the way it would be if "relevant" only meant "was retrieved".
    """
    return {
        chunk.chunk_id
        for fact in facts
        for chunk in all_chunks
        if chunk.article_id == fact["article_id"] and _chunk_supports(chunk.text, fact["must_contain"])
    }


def _retrieval_record(hits: Sequence[SearchHit], all_chunks: Sequence, facts: Sequence[Dict[str, Any]]) -> RetrievalRecord:
    retrieved_ids = [h.chunk.chunk_id for h in hits]
    return RetrievalRecord(retrieved_ids=retrieved_ids, relevant_ids=_relevant_ids(all_chunks, facts))


def _answer_supports_primary(answer: Answer, facts: Sequence[Dict[str, Any]]) -> bool:
    haystack = " ".join(answer.text.split()).lower()
    return all(" ".join(str(n).split()).lower() in haystack for n in facts[0]["must_contain"])


@dataclass
class FailureRecord:
    qid: str
    category: str
    question: str
    facts: List[Dict[str, Any]]
    known_answer: str
    semantic_hits: List[SearchHit]
    hybrid_hits: List[SearchHit]
    semantic_rank: Optional[int]
    hybrid_rank: Optional[int]
    answer: Answer  # pipeline.ask under semantic-only retrieval: the "before" behaviour

    @property
    def semantic_hit3(self) -> bool:
        return self.semantic_rank is not None and self.semantic_rank <= K

    @property
    def hybrid_hit3(self) -> bool:
        return self.hybrid_rank is not None and self.hybrid_rank <= K

    @property
    def fixed_by_hybrid(self) -> bool:
        return (not self.semantic_hit3) and self.hybrid_hit3

    @property
    def broken_by_hybrid(self) -> bool:
        return self.semantic_hit3 and not self.hybrid_hit3

    @property
    def label(self) -> str:
        """The two-way split the task asks for, computed from evidence, not assumed."""
        if not self.semantic_hit3:
            if self.semantic_rank is None:
                return f"retrieval failure: wrong document (not in the top-{WINDOW} at all)"
            return f"retrieval failure: right article exists but ranked {self.semantic_rank}, outside the top-{K}"
        if self.answer.refused:
            return "generation failure: right document retrieved, but the pipeline refused"
        if not _answer_supports_primary(self.answer, self.facts):
            return "generation failure: right document retrieved, but the answer omits the required fact"
        return "no failure"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "category": self.category,
            "question": self.question,
            "known_answer": self.known_answer,
            "label": self.label,
            "semantic_rank": self.semantic_rank,
            "hybrid_rank": self.hybrid_rank,
            "semantic_hit_at_3": self.semantic_hit3,
            "hybrid_hit_at_3": self.hybrid_hit3,
            "fixed_by_hybrid": self.fixed_by_hybrid,
            "broken_by_hybrid": self.broken_by_hybrid,
            "before_answer": {
                "refused": self.answer.refused,
                "text": self.answer.text,
                "refusal_reason": self.answer.refusal_reason,
            },
            "semantic_top5": [h.to_dict() for h in self.semantic_hits[:5]],
            "hybrid_top5": [h.to_dict() for h in self.hybrid_hits[:5]],
        }


@dataclass
class Week4Report:
    records: List[FailureRecord]
    semantic_hit_rate_at_3: float
    hybrid_hit_rate_at_3: float
    semantic_recall_at_3: float
    hybrid_recall_at_3: float
    semantic_mrr: float
    hybrid_mrr: float
    config_summary: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""

    @property
    def n_questions(self) -> int:
        return len(self.records)

    @property
    def n_semantic_hits(self) -> int:
        return sum(1 for r in self.records if r.semantic_hit3)

    @property
    def n_hybrid_hits(self) -> int:
        return sum(1 for r in self.records if r.hybrid_hit3)

    @property
    def n_fixed(self) -> int:
        return sum(1 for r in self.records if r.fixed_by_hybrid)

    @property
    def n_broken(self) -> int:
        return sum(1 for r in self.records if r.broken_by_hybrid)

    def by_category(self) -> Dict[str, List[FailureRecord]]:
        groups: Dict[str, List[FailureRecord]] = {}
        for r in self.records:
            groups.setdefault(r.category, []).append(r)
        return groups

    def summary_text(self) -> str:
        lines = [
            f"hit-rate@3   semantic {self.n_semantic_hits}/{self.n_questions}"
            f" ({self.semantic_hit_rate_at_3:.0%})  ->  hybrid {self.n_hybrid_hits}/{self.n_questions}"
            f" ({self.hybrid_hit_rate_at_3:.0%})",
            f"recall@3     semantic {self.semantic_recall_at_3:.2f}  ->  hybrid {self.hybrid_recall_at_3:.2f}",
            f"MRR@{WINDOW}       semantic {self.semantic_mrr:.3f}  ->  hybrid {self.hybrid_mrr:.3f}",
            f"fixed by the change: {self.n_fixed}   broken by the change: {self.n_broken}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "config": self.config_summary,
            "n_questions": self.n_questions,
            "hit_rate_at_3": {
                "semantic": round(self.semantic_hit_rate_at_3, 4),
                "hybrid": round(self.hybrid_hit_rate_at_3, 4),
            },
            "recall_at_3": {
                "semantic": round(self.semantic_recall_at_3, 4),
                "hybrid": round(self.hybrid_recall_at_3, 4),
            },
            "mrr": {"semantic": round(self.semantic_mrr, 4), "hybrid": round(self.hybrid_mrr, 4)},
            "n_semantic_hits": self.n_semantic_hits,
            "n_hybrid_hits": self.n_hybrid_hits,
            "n_fixed_by_hybrid": self.n_fixed,
            "n_broken_by_hybrid": self.n_broken,
            "records": [r.to_dict() for r in self.records],
        }


def evaluate_failures(
    config: Config,
    questions: Sequence[Dict[str, Any]],
    strategy: Optional[str] = None,
    index_dir: Optional[Path] = None,
) -> Week4Report:
    strategy = strategy or config.chunking.default_strategy
    ensure_index(config, strategy, index_dir)

    # The "before" pipeline is pinned to semantic-only retrieval regardless of
    # what config.yaml ships today, because failure labelling must reflect the
    # state these questions were caught failing in, not whatever the current
    # default happens to be.
    before_config = config.with_retrieval(mode="semantic")
    pipeline = RAGPipeline.open(before_config, strategy=strategy, index_dir=index_dir)
    retriever = pipeline.retriever

    records: List[FailureRecord] = []
    for item in questions:
        facts = _facts(item)
        semantic_hits = retriever.search(item["question"], top_k=WINDOW, mode="semantic")
        hybrid_hits = retriever.search(item["question"], top_k=WINDOW, mode="hybrid")
        answer = pipeline.ask(item["question"], mode="semantic")

        records.append(
            FailureRecord(
                qid=item["id"],
                category=item.get("category", ""),
                question=item["question"],
                facts=facts,
                known_answer=item.get("known_answer", ""),
                semantic_hits=semantic_hits,
                hybrid_hits=hybrid_hits,
                semantic_rank=_rank_of_fact(semantic_hits, facts[0]),
                hybrid_rank=_rank_of_fact(hybrid_hits, facts[0]),
                answer=answer,
            )
        )

    # hit-rate@k and MRR are about the PRIMARY fact only ("did the right
    # document show up") — a secondary also_needs fact must not let a
    # question count as a hit when the primary answer itself was missed.
    # recall@k is the one metric meant to give partial credit across the
    # full fact set, so it alone uses every fact.
    all_chunks = retriever.store.chunks
    semantic_primary = [_retrieval_record(r.semantic_hits, all_chunks, r.facts[:1]) for r in records]
    hybrid_primary = [_retrieval_record(r.hybrid_hits, all_chunks, r.facts[:1]) for r in records]
    semantic_all = [_retrieval_record(r.semantic_hits, all_chunks, r.facts) for r in records]
    hybrid_all = [_retrieval_record(r.hybrid_hits, all_chunks, r.facts) for r in records]

    return Week4Report(
        records=records,
        semantic_hit_rate_at_3=hit_rate_at_k(semantic_primary, K),
        hybrid_hit_rate_at_3=hit_rate_at_k(hybrid_primary, K),
        semantic_recall_at_3=recall_at_k(semantic_all, K),
        hybrid_recall_at_3=recall_at_k(hybrid_all, K),
        semantic_mrr=mrr(semantic_primary, WINDOW),
        hybrid_mrr=mrr(hybrid_primary, WINDOW),
    )


def run_week4_evaluation(
    config: Config,
    questions_path: Optional[Path] = None,
    strategy: Optional[str] = None,
    index_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> Week4Report:
    repo_root = config.repo_root
    questions_path = Path(questions_path) if questions_path else repo_root / "eval" / "week4_questions.yaml"
    questions = load_week4_questions(questions_path)
    strategy = strategy or config.chunking.default_strategy
    out_dir = Path(out_dir) if out_dir else config.paths.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = evaluate_failures(config, questions, strategy=strategy, index_dir=index_dir)
    report.generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report.config_summary = {
        "strategy": strategy,
        "k": K,
        "window": WINDOW,
        "rrf_k": config.retrieval.rrf_k,
        "bm25_k1": config.retrieval.bm25_k1,
        "bm25_b": config.retrieval.bm25_b,
        "shipping_retrieval_mode": config.retrieval.mode,
        "questions_file": str(questions_path),
    }

    from .reporting import write_week4_artifacts

    write_week4_artifacts(report, out_dir=out_dir, repo_root=repo_root)
    return report
