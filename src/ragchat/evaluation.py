"""Evaluation harness.

Produces every number in ``results.md``:

* hit-in-top-5 over the same eight known-answer questions for each chunking
  strategy, with the per-question record rather than a summary claim
* the unfiltered vs filtered result lists for one ``product_area`` query
* cited answers whose citations are checked against the chunk they name
* refusal transcripts for the out-of-corpus questions
* a chunk-size sweep across both strategies
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from .config import Config
from .indexer import ingest
from .models import Answer, SearchHit
from .pipeline import RAGPipeline
from .retriever import Retriever
from .store import StoreError, VectorStore, namespace_for

DEFAULT_STRATEGIES = ("baseline", "structure-aware")
SWEEP_SIZES = (400, 800, 1200, 1600)
# The bonus question is colloquially phrased and trips the shipping gate; the bonus
# comparison is about answer completeness, so it runs with the gate relaxed to here.
BONUS_COVERAGE = 0.30


# --------------------------------------------------------------------------
# Loading and indexing
# --------------------------------------------------------------------------


def load_questions(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    if not payload or "answerable" not in payload:
        raise ValueError(f"{path} does not look like a question set")
    return payload


def ensure_index(config: Config, strategy: str, index_dir: Path | None = None) -> str:
    """Build the index for a strategy if it does not exist yet.

    Both strategies are given the identical corpus — the pre-existing legacy
    articles plus the new drop — so a hit-rate difference cannot come from one
    index having seen more documents than the other.
    """
    index_dir = Path(index_dir) if index_dir else config.paths.index_dir
    namespace = namespace_for(strategy, config.chunking.chunk_size, config.chunking.chunk_overlap)
    try:
        VectorStore.load(index_dir, namespace)
        return namespace
    except StoreError:
        pass

    ingest(config, [config.paths.legacy_articles_dir], strategy=strategy,
           label="legacy-corpus (pre-existing index)", index_dir=index_dir)
    ingest(config, [config.paths.articles_dir], strategy=strategy,
           label="week3-new-drop", index_dir=index_dir)
    return namespace


# --------------------------------------------------------------------------
# Retrieval scoring
# --------------------------------------------------------------------------


@dataclass
class QuestionResult:
    qid: str
    question: str
    gold_article: str
    gold_section: str
    must_contain: List[str]
    depends_on_table_row: bool
    hit: bool
    hit_rank: Optional[int]
    hit_chunk_id: Optional[str]
    best_article_rank: Optional[int]
    diagnosis: str
    self_contained_hit: bool = False
    self_contained_rank: Optional[int] = None
    self_contained_chunk_id: Optional[str] = None
    hits: List[SearchHit] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "gold_article": self.gold_article,
            "gold_section": self.gold_section,
            "must_contain": self.must_contain,
            "depends_on_table_row": self.depends_on_table_row,
            "hit": self.hit,
            "hit_rank": self.hit_rank,
            "hit_chunk_id": self.hit_chunk_id,
            "self_contained_hit": self.self_contained_hit,
            "self_contained_rank": self.self_contained_rank,
            "self_contained_chunk_id": self.self_contained_chunk_id,
            "best_article_rank": self.best_article_rank,
            "diagnosis": self.diagnosis,
            "hits": [h.to_dict() for h in self.hits],
        }


@dataclass
class StrategyReport:
    strategy: str
    namespace: str
    n_chunks: int
    mean_chunk_chars: float
    results: List[QuestionResult]

    @property
    def n_hits(self) -> int:
        return sum(1 for r in self.results if r.hit)

    @property
    def n_self_contained(self) -> int:
        return sum(1 for r in self.results if r.self_contained_hit)

    @property
    def n_questions(self) -> int:
        return len(self.results)

    @property
    def table_self_contained(self) -> str:
        rows = [r for r in self.results if r.depends_on_table_row]
        return f"{sum(1 for r in rows if r.self_contained_hit)}/{len(rows)}"

    @property
    def table_hits(self) -> str:
        rows = [r for r in self.results if r.depends_on_table_row]
        return f"{sum(1 for r in rows if r.hit)}/{len(rows)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "namespace": self.namespace,
            "n_chunks": self.n_chunks,
            "mean_chunk_chars": self.mean_chunk_chars,
            "hit_at_5": f"{self.n_hits}/{self.n_questions}",
            "self_contained_hit_at_5": f"{self.n_self_contained}/{self.n_questions}",
            "results": [r.to_dict() for r in self.results],
        }


def _chunk_supports(chunk_text: str, must_contain: Sequence[str]) -> bool:
    haystack = " ".join(chunk_text.split()).lower()
    return all(" ".join(str(needle).split()).lower() in haystack for needle in must_contain)


def _is_separator_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and set(stripped) <= set("|-: ")


def contains_table_rows(text: str) -> bool:
    return any(line.strip().startswith("|") and not _is_separator_row(line) for line in text.splitlines())


def has_table_header(text: str) -> bool:
    """True when a header row (a row followed by its separator) is present."""
    lines = text.splitlines()
    return any(
        lines[i].strip().startswith("|")
        and not _is_separator_row(lines[i])
        and _is_separator_row(lines[i + 1])
        for i in range(len(lines) - 1)
    )


def is_self_contained(chunk_text: str) -> bool:
    """A chunk is self-contained when its table rows still have their header.

    This is the secondary metric. A bare ``| ERR-4032 | ... | ... |`` row is
    retrievable and it satisfies the committed hit criterion, but without the
    header an agent cannot tell which cell is the cause and which is the fix.
    """
    if not contains_table_rows(chunk_text):
        return True
    return has_table_header(chunk_text)


def evaluate_strategy(
    config: Config,
    strategy: str,
    questions: Dict[str, Any],
    index_dir: Path | None = None,
    top_k: int = 5,
) -> StrategyReport:
    namespace = ensure_index(config, strategy, index_dir)
    retriever = Retriever.open(config, strategy=strategy, index_dir=index_dir)

    results: List[QuestionResult] = []
    for item in questions["answerable"]:
        gold = item["gold"]
        must_contain = list(item.get("must_contain", []))
        hits = retriever.search(item["question"], top_k=top_k)

        hit_rank: Optional[int] = None
        hit_chunk_id: Optional[str] = None
        best_article_rank: Optional[int] = None
        sc_rank: Optional[int] = None
        sc_chunk_id: Optional[str] = None
        for hit in hits:
            if hit.chunk.article_id == gold["article_id"]:
                if best_article_rank is None:
                    best_article_rank = hit.rank
                if _chunk_supports(hit.chunk.text, must_contain):
                    if hit_rank is None:
                        hit_rank, hit_chunk_id = hit.rank, hit.chunk.chunk_id
                    if sc_rank is None and is_self_contained(hit.chunk.text):
                        sc_rank, sc_chunk_id = hit.rank, hit.chunk.chunk_id

        results.append(
            QuestionResult(
                qid=item["id"],
                question=item["question"],
                gold_article=gold["article_id"],
                gold_section=gold.get("section", ""),
                must_contain=must_contain,
                depends_on_table_row=bool(item.get("depends_on_table_row")),
                hit=hit_rank is not None,
                hit_rank=hit_rank,
                hit_chunk_id=hit_chunk_id,
                best_article_rank=best_article_rank,
                diagnosis=_diagnose(hits, gold["article_id"], must_contain, best_article_rank, hit_rank),
                self_contained_hit=sc_rank is not None,
                self_contained_rank=sc_rank,
                self_contained_chunk_id=sc_chunk_id,
                hits=hits,
            )
        )

    stats = retriever.store.stats()
    return StrategyReport(
        strategy=strategy,
        namespace=namespace,
        n_chunks=stats["n_chunks"],
        mean_chunk_chars=stats["mean_chunk_chars"],
        results=results,
    )


def _diagnose(
    hits: Sequence[SearchHit],
    gold_article: str,
    must_contain: Sequence[str],
    best_article_rank: Optional[int],
    hit_rank: Optional[int],
) -> str:
    if hit_rank is not None:
        return f"answer-bearing chunk retrieved at rank {hit_rank}"
    if best_article_rank is None:
        return f"no chunk from {gold_article} in the top {len(hits)}"
    partial = [
        needle
        for needle in must_contain
        if any(
            " ".join(str(needle).split()).lower() in " ".join(h.chunk.text.split()).lower()
            for h in hits
            if h.chunk.article_id == gold_article
        )
    ]
    missing = [n for n in must_contain if n not in partial]
    return (
        f"right article at rank {best_article_rank} but the retrieved chunks are missing "
        f"{missing!r} — the answer was split across chunk boundaries"
    )


# --------------------------------------------------------------------------
# Metadata filter demonstration
# --------------------------------------------------------------------------


@dataclass
class FilterDemo:
    question: str
    filters: Dict[str, Any]
    unfiltered: List[SearchHit]
    filtered: List[SearchHit]

    @property
    def changed_top1(self) -> bool:
        if not self.unfiltered or not self.filtered:
            return False
        return self.unfiltered[0].chunk.chunk_id != self.filtered[0].chunk.chunk_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "filters": self.filters,
            "changed_top1": self.changed_top1,
            "unfiltered": [h.to_dict() for h in self.unfiltered],
            "filtered": [h.to_dict() for h in self.filtered],
        }


def run_filter_demo(
    config: Config,
    strategy: str,
    demo: Dict[str, Any],
    index_dir: Path | None = None,
    top_k: int = 5,
) -> FilterDemo:
    retriever = Retriever.open(config, strategy=strategy, index_dir=index_dir)
    question = demo["question"]
    filters = dict(demo["filter"])
    return FilterDemo(
        question=question,
        filters=filters,
        unfiltered=retriever.search(question, top_k=top_k),
        filtered=retriever.search(question, top_k=top_k, filters=filters),
    )


# --------------------------------------------------------------------------
# Generation: cited answers and refusals
# --------------------------------------------------------------------------


@dataclass
class AnswerCheck:
    qid: str
    question: str
    answer: Answer
    citations_resolve: bool
    citation_supports_claim: bool
    supporting_chunk_id: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "citations_resolve": self.citations_resolve,
            "citation_supports_claim": self.citation_supports_claim,
            "supporting_chunk_id": self.supporting_chunk_id,
            "answer": self.answer.to_dict(),
        }


def run_answer_checks(
    pipeline: RAGPipeline,
    items: Sequence[Dict[str, Any]],
) -> List[AnswerCheck]:
    checks: List[AnswerCheck] = []
    for item in items:
        answer = pipeline.ask(item["question"])
        must_contain = list(item.get("must_contain", []))

        resolves = bool(answer.citations) and all(c.resolved for c in answer.citations)
        supporting_id = None
        for citation in answer.citations:
            chunk = pipeline.retriever.get_chunk(citation.chunk_id)
            if chunk and _chunk_supports(chunk.text, must_contain):
                supporting_id = chunk.chunk_id
                break

        checks.append(
            AnswerCheck(
                qid=item["id"],
                question=item["question"],
                answer=answer,
                citations_resolve=resolves,
                citation_supports_claim=supporting_id is not None,
                supporting_chunk_id=supporting_id,
            )
        )
    return checks


def run_refusals(pipeline: RAGPipeline, items: Sequence[Dict[str, Any]]) -> List[Answer]:
    return [pipeline.ask(item["question"]) for item in items]


# --------------------------------------------------------------------------
# Chunk-size sweep
# --------------------------------------------------------------------------


def run_sweep(
    config: Config,
    questions: Dict[str, Any],
    sizes: Sequence[int],
    strategies: Sequence[str],
    index_dir: Path | None = None,
) -> List[Dict[str, Any]]:
    """hit@5 for every (strategy, chunk size) pair. One variable changes at a time."""
    base_dir = Path(index_dir) if index_dir else config.paths.index_dir
    sweep_dir = base_dir / "sweep"
    rows: List[Dict[str, Any]] = []

    for strategy in strategies:
        for size in sizes:
            overlap = min(config.chunking.chunk_overlap, max(size // 8, 0))
            sized = config.with_chunking(chunk_size=size, chunk_overlap=overlap)
            report = evaluate_strategy(sized, strategy, questions, index_dir=sweep_dir)
            table = [r for r in report.results if r.depends_on_table_row]
            prose = [r for r in report.results if not r.depends_on_table_row]
            rows.append(
                {
                    "strategy": strategy,
                    "chunk_size": size,
                    "chunk_overlap": overlap,
                    "n_chunks": report.n_chunks,
                    "mean_chunk_chars": report.mean_chunk_chars,
                    "hit_at_5": f"{report.n_hits}/{report.n_questions}",
                    "hit_rate": round(report.n_hits / report.n_questions, 3),
                    "self_contained_hit_at_5": f"{report.n_self_contained}/{report.n_questions}",
                    "table_questions": f"{sum(1 for r in table if r.hit)}/{len(table)}",
                    "table_questions_self_contained": f"{sum(1 for r in table if r.self_contained_hit)}/{len(table)}",
                    "prose_questions": f"{sum(1 for r in prose if r.hit)}/{len(prose)}",
                }
            )
    return rows


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


@dataclass
class EvaluationReport:
    generated_at: str
    config_summary: Dict[str, Any]
    strategy_reports: List[StrategyReport]
    filter_demo: FilterDemo
    answer_checks: List[AnswerCheck]
    refusals: List[Answer]
    bonus: List[Dict[str, Any]]
    sweep: List[Dict[str, Any]]
    index_history: Dict[str, Any]

    def summary_text(self) -> str:
        lines = ["hit-in-top-5 over the same 8 known-answer questions:"]
        for report in self.strategy_reports:
            lines.append(
                f"  {report.strategy:16} hit@5 {report.n_hits}/{report.n_questions}"
                f"   self-contained {report.n_self_contained}/{report.n_questions}"
                f"   (table-row questions {report.table_hits} / self-contained {report.table_self_contained})"
                f"   chunks={report.n_chunks}"
            )
        lines.append(f"filter changed top-1: {self.filter_demo.changed_top1}")
        answered = sum(1 for c in self.answer_checks if not c.answer.refused)
        supported = sum(1 for c in self.answer_checks if c.citation_supports_claim)
        lines.append(f"cited answers: {answered}/{len(self.answer_checks)} answered, {supported} citation-verified")
        refused = sum(1 for a in self.refusals if a.refused)
        lines.append(f"out-of-corpus questions refused: {refused}/{len(self.refusals)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "config": self.config_summary,
            "strategies": [r.to_dict() for r in self.strategy_reports],
            "filter_demo": self.filter_demo.to_dict(),
            "answer_checks": [c.to_dict() for c in self.answer_checks],
            "refusals": [a.to_dict() for a in self.refusals],
            "bonus": self.bonus,
            "sweep": self.sweep,
            "index_history": self.index_history,
        }


def run_evaluation(
    config: Config,
    questions_path: Path | None = None,
    strategies: Sequence[str] | None = None,
    out_dir: Path | None = None,
    index_dir: Path | None = None,
    with_sweep: bool = True,
) -> EvaluationReport:
    repo_root = config.repo_root
    questions_path = Path(questions_path) if questions_path else repo_root / "eval" / "questions.yaml"
    questions = load_questions(questions_path)
    strategies = list(strategies) if strategies else list(DEFAULT_STRATEGIES)
    out_dir = Path(out_dir) if out_dir else config.paths.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    strategy_reports = [
        evaluate_strategy(config, strategy, questions, index_dir=index_dir) for strategy in strategies
    ]

    # Generation-side checks run against the strategy we intend to ship.
    shipping = strategies[-1]
    pipeline = RAGPipeline.open(config, strategy=shipping, index_dir=index_dir)
    answer_checks = run_answer_checks(pipeline, questions["answerable"][:3])
    refusals = run_refusals(pipeline, questions["unanswerable"])
    filter_demo = run_filter_demo(config, shipping, questions["filter_demo"], index_dir=index_dir)

    bonus = _run_bonus(config, questions, strategies, index_dir)

    sweep = (
        run_sweep(config, questions, sizes=SWEEP_SIZES, strategies=strategies, index_dir=index_dir)
        if with_sweep
        else []
    )

    history = {}
    for report in strategy_reports:
        store = VectorStore.load(Path(index_dir) if index_dir else config.paths.index_dir, report.namespace)
        history[report.strategy] = store.stats()["history"]

    evaluation = EvaluationReport(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config_summary={
            "chunk_size": config.chunking.chunk_size,
            "chunk_overlap": config.chunking.chunk_overlap,
            "embedding": config.embedding.backend,
            "n_features": config.embedding.n_features,
            "top_k": config.retrieval.top_k,
            "generation_backend": pipeline.generator.name,
            "generation_model": config.generation.model if pipeline.generator.name == "anthropic" else "n/a",
            "min_top_score": config.grounding.min_top_score,
            "min_evidence_coverage": config.grounding.min_evidence_coverage,
            "questions_file": str(questions_path),
        },
        strategy_reports=strategy_reports,
        filter_demo=filter_demo,
        answer_checks=answer_checks,
        refusals=refusals,
        bonus=bonus,
        sweep=sweep,
        index_history=history,
    )

    from .reporting import write_artifacts

    write_artifacts(evaluation, questions, out_dir=out_dir, repo_root=repo_root)
    return evaluation


def _run_bonus(
    config: Config,
    questions: Dict[str, Any],
    strategies: Sequence[str],
    index_dir: Path | None,
) -> List[Dict[str, Any]]:
    """Answer each bonus probe under every strategy for a side-by-side read.

    Both probes run with the evidence gate relaxed. The committed probe is phrased
    colloquially ("what will they need to hand?"), and "hit" and "hand" are ordinary
    English words this small corpus happens not to use, so the shipping gate refuses
    it — a false refusal discussed in the write-up. The bonus is about answer
    completeness rather than the gate, so the relaxed value is recorded and reported.
    """
    probes = [questions.get("bonus"), questions.get("bonus_followup")]
    relaxed = config.with_grounding(min_evidence_coverage=BONUS_COVERAGE)

    results: List[Dict[str, Any]] = []
    for probe in probes:
        if not probe:
            continue
        top_k = int(probe.get("context_chunks", config.retrieval.top_k))
        answers = {}
        for strategy in strategies:
            pipeline = RAGPipeline.open(relaxed, strategy=strategy, index_dir=index_dir)
            answers[strategy] = pipeline.ask(probe["question"], top_k=top_k).to_dict()
        results.append(
            {
                "id": probe.get("id", ""),
                "question": probe["question"],
                "known_answer": probe.get("known_answer", ""),
                "context_chunks": top_k,
                "added_after_first_run": bool(probe.get("added_after_first_run")),
                "answers": answers,
                "gate_relaxed_to": BONUS_COVERAGE,
                "shipping_gate": config.grounding.min_evidence_coverage,
            }
        )
    return results
