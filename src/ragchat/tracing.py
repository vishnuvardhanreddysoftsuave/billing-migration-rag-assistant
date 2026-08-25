"""Week 5 Task Set A: collect complete traces from a fair, random sample.

Where :mod:`ragchat.evaluation` and :mod:`ragchat.failure_analysis` each score a
question set someone already suspects is interesting, this module does the
opposite: it draws a sample from a *broad* pool built to stand in for real
traffic (see ``eval/week5_questions.yaml``), runs every sampled question
through the pipeline exactly as it ships today, and writes down everything
that happened -- not just whether it was a hit. Reading those traces by hand
is Week 5's actual task; this module only produces the material to read.

A "trace" here is deliberately the same shape as :class:`~ragchat.models.Answer`
plus the request-time settings that produced it: question, retrieval mode,
every retrieved chunk with its rank and score, the evidence-gate verdict, the
generation backend and its diagnostics, the final text or refusal reason, and
citations. That is enough to replay the request later with
``python rag.py inspect "<question>"`` against the same config and index.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from .config import Config
from .evaluation import ensure_index
from .models import Answer
from .pipeline import RAGPipeline

DEFAULT_SAMPLE_SIZE = 20
DEFAULT_SEED = 5  # fixed before the sampler was ever run; see eval/week5_questions.yaml

# Named problem groups the open-coding notes in eval/week5_open_coding.yaml assign
# traces to. Kept here, not duplicated in the YAML, so the report generator and the
# coding file cannot describe the same group two different ways.
PROBLEM_GROUPS: Dict[str, Dict[str, str]] = {
    "false_refusal_informal_phrasing": {
        "name": "False refusal on answerable questions (informal phrasing)",
        "description": (
            "The evidence gate's idf-weighted coverage check treats ordinary support-ticket "
            "filler words (\"getting\", \"trying\", \"throwing\", \"hey\", \"idea\", ...) as "
            "missing corpus content, because they are absent from this small corpus and are "
            "not on the QUESTION_WORDS stoplist. A short, informally-phrased ticket can lose "
            "enough idf-weighted coverage to this filler alone that it is refused even when "
            "the exact answer-bearing chunk was retrieved at rank 1."
        ),
    },
    "offtopic_confident_answer": {
        "name": "Off-topic confident answer after retrieval drift",
        "description": (
            "When retrieval drifts to a chunk that shares surface vocabulary with the question "
            "but not its actual topic, the extractive generator can still find enough term "
            "overlap to select and cite sentences, producing a fully grounded, citation-valid "
            "answer that is nonetheless about the wrong thing, with no signal to the reader "
            "that it might be off-topic."
        ),
    },
    "ambiguous_no_clarifying_question": {
        "name": "Ambiguous question refused instead of clarified",
        "description": (
            "A question that is genuinely ambiguous across more than one part of the corpus is "
            "refused outright with the standard refusal message, rather than being met with a "
            "clarifying question that names the specific possibilities the retrieved chunks "
            "already surfaced."
        ),
    },
    "tangential_citation_padding": {
        "name": "Correct answer padded with a tangential citation",
        "description": (
            "The extractive generator's relative-cutoff unit selection can admit a sentence "
            "that scores just above the cutoff but is not substantively related to the "
            "question, appending it to an otherwise correct and sufficient answer."
        ),
    },
}


def load_pool(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    if not payload or "pool" not in payload:
        raise ValueError(f"{path} does not look like a week5 question pool")
    return list(payload["pool"])


def sample_questions(
    pool: Sequence[Dict[str, Any]],
    n: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> List[Dict[str, Any]]:
    """A fair, random draw from the pool -- not a curated pick.

    Deterministic given ``(n, seed)`` so the same 20 traces can be regenerated
    later; two different seeds draw two different samples, which is what makes
    this a documented random sample rather than a hand-picked one. Order is the
    draw order, independent of the pool's own ordering.
    """
    if n > len(pool):
        raise ValueError(f"cannot sample {n} items from a pool of {len(pool)}")
    return random.Random(seed).sample(list(pool), n)


@dataclass
class TraceRecord:
    """One complete, replayable record of a single request."""

    trace_id: str
    category: str
    question: str
    pool_gold: Optional[Dict[str, Any]]
    pool_must_contain: List[str]
    pool_note: str
    config_snapshot: Dict[str, Any]
    answer: Answer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "category": self.category,
            "question": self.question,
            "pool_gold": self.pool_gold,
            "pool_must_contain": self.pool_must_contain,
            "pool_note": self.pool_note,
            "config": self.config_snapshot,
            "answer": self.answer.to_dict(),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# {self.trace_id} ({self.category})",
            "",
            f"**Question:** {self.question}",
        ]
        if self.pool_gold:
            lines.append(
                f"**Pool gold:** {self.pool_gold.get('article_id')} / {self.pool_gold.get('section', '')}"
            )
        if self.pool_must_contain:
            lines.append(f"**Pool must_contain:** {self.pool_must_contain}")
        if self.pool_note:
            lines.append(f"**Pool note:** {self.pool_note}")
        lines += [
            "",
            f"retrieval mode: `{self.config_snapshot['retrieval_mode']}`  ·  "
            f"strategy: `{self.config_snapshot['strategy']}`  ·  "
            f"generation backend: `{self.answer.backend}`",
            "",
            "## Retrieved (top-5)",
            "",
        ]
        hits = self.answer.hits[:5]
        if not hits:
            lines.append("(no matches)")
        for hit in hits:
            chunk = hit.chunk
            preview = " ".join(chunk.text.split())
            lines.append(
                f"- #{hit.rank} score={hit.score:.4f} `{chunk.chunk_id}` "
                f"[{chunk.article_id}] {chunk.section or '(no section)'}\n"
                f"  {preview[:200]}{'…' if len(preview) > 200 else ''}"
            )
        evidence = self.answer.diagnostics.get("evidence", {})
        lines += [
            "",
            "## Evidence gate",
            "",
            f"- sufficient: {evidence.get('sufficient')}",
            f"- idf-weighted coverage: {evidence.get('idf_weighted_coverage')}",
            f"- missing terms: {evidence.get('missing_terms')}",
            "",
            "## Final answer",
            "",
        ]
        if self.answer.refused:
            lines.append(f"REFUSED -- {self.answer.refusal_reason}")
        else:
            lines.append(self.answer.text.strip())
            lines.append("")
            lines.append("Citations:")
            for citation in self.answer.citations:
                lines.append(
                    f"  - [{citation.chunk_id}] resolved={citation.resolved} "
                    f"{citation.article_id} · {citation.section}"
                )
        return "\n".join(lines) + "\n"


@dataclass
class TraceSet:
    generated_at: str
    seed: int
    pool_size: int
    config_summary: Dict[str, Any]
    records: List[TraceRecord] = field(default_factory=list)

    @property
    def n_traces(self) -> int:
        return len(self.records)

    @property
    def n_refused(self) -> int:
        return sum(1 for r in self.records if r.answer.refused)

    @property
    def n_answered(self) -> int:
        return self.n_traces - self.n_refused

    def by_category(self) -> Dict[str, List[TraceRecord]]:
        groups: Dict[str, List[TraceRecord]] = {}
        for r in self.records:
            groups.setdefault(r.category, []).append(r)
        return groups

    def summary_text(self) -> str:
        lines = [
            f"{self.n_traces} traces sampled from a pool of {self.pool_size} (seed={self.seed})",
            f"answered: {self.n_answered}   refused: {self.n_refused}",
        ]
        for category, records in self.by_category().items():
            lines.append(f"  {category:22} {len(records)} traces")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "seed": self.seed,
            "pool_size": self.pool_size,
            "config": self.config_summary,
            "n_traces": self.n_traces,
            "n_answered": self.n_answered,
            "n_refused": self.n_refused,
            "records": [r.to_dict() for r in self.records],
        }


def run_trace_sample(
    config: Config,
    pool_path: Optional[Path] = None,
    strategy: Optional[str] = None,
    index_dir: Optional[Path] = None,
    n: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    out_dir: Optional[Path] = None,
) -> TraceSet:
    """Sample real questions and run every one through the pipeline as shipped.

    Uses ``config.yaml`` unchanged (whatever strategy and retrieval mode ship
    today), which is the point: these traces show the app a real user would
    actually hit, not a mode pinned for a particular week's demonstration.
    """
    repo_root = config.repo_root
    pool_path = Path(pool_path) if pool_path else repo_root / "eval" / "week5_questions.yaml"
    pool = load_pool(pool_path)

    strategy = strategy or config.chunking.default_strategy
    ensure_index(config, strategy, index_dir)
    pipeline = RAGPipeline.open(config, strategy=strategy, index_dir=index_dir)

    sampled = sample_questions(pool, n=n, seed=seed)
    config_snapshot = {
        "strategy": strategy,
        "retrieval_mode": config.retrieval.mode,
        "chunk_size": config.chunking.chunk_size,
        "chunk_overlap": config.chunking.chunk_overlap,
        "generation_backend": pipeline.generator.name,
        "namespace": pipeline.namespace,
    }

    records: List[TraceRecord] = []
    for item in sampled:
        answer = pipeline.ask(item["question"])
        records.append(
            TraceRecord(
                trace_id=item["id"],
                category=item.get("category", ""),
                question=item["question"],
                pool_gold=item.get("gold"),
                pool_must_contain=list(item.get("must_contain") or []),
                pool_note=item.get("note", ""),
                config_snapshot=dict(config_snapshot),
                answer=answer,
            )
        )

    trace_set = TraceSet(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        seed=seed,
        pool_size=len(pool),
        config_summary=config_snapshot,
        records=records,
    )

    out_dir = Path(out_dir) if out_dir else config.paths.results_dir
    write_trace_artifacts(trace_set, out_dir=out_dir)
    return trace_set


def write_trace_artifacts(trace_set: TraceSet, out_dir: Path) -> None:
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "week5_traces.json").write_text(
        json.dumps(trace_set.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    traces_dir = out_dir / "week5_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    for old in traces_dir.glob("*.md"):
        old.unlink()
    for record in trace_set.records:
        (traces_dir / f"{record.trace_id}.md").write_text(record.to_markdown(), encoding="utf-8")


# --------------------------------------------------------------------------
# Open coding and the ranked taxonomy
# --------------------------------------------------------------------------


def load_open_coding(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    if not payload or "coding" not in payload:
        raise ValueError(f"{path} does not look like a week5 open-coding file")
    rows = list(payload["coding"])
    for row in rows:
        group = row.get("problem_group")
        if group is not None and group not in PROBLEM_GROUPS:
            raise ValueError(f"{row['trace_id']}: unknown problem_group {group!r}")
        if group is None and row.get("severity", 0) != 0:
            raise ValueError(f"{row['trace_id']}: severity {row.get('severity')} but no problem_group")
    return rows


def validate_open_coding(trace_set: TraceSet, coding: Sequence[Dict[str, Any]]) -> None:
    """Every sampled trace must have exactly one note, and vice versa."""
    traced_ids = {r.trace_id for r in trace_set.records}
    coded_ids = [row["trace_id"] for row in coding]
    missing = traced_ids - set(coded_ids)
    if missing:
        raise ValueError(f"traces with no open-coding note: {sorted(missing)}")
    extra = set(coded_ids) - traced_ids
    if extra:
        raise ValueError(f"open-coding notes for traces that were not sampled: {sorted(extra)}")
    duplicates = [tid for tid in set(coded_ids) if coded_ids.count(tid) > 1]
    if duplicates:
        raise ValueError(f"duplicate open-coding notes: {sorted(duplicates)}")


@dataclass
class TaxonomyRow:
    slug: str
    name: str
    description: str
    count: int
    mean_severity: float
    trace_ids: List[str]

    @property
    def score(self) -> float:
        """Frequency x severity -- the ranking signal, not a claim of precision."""
        return round(self.count * self.mean_severity, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slug": self.slug,
            "name": self.name,
            "description": self.description,
            "count": self.count,
            "mean_severity": round(self.mean_severity, 2),
            "score": self.score,
            "trace_ids": self.trace_ids,
        }


def build_taxonomy(coding: Sequence[Dict[str, Any]]) -> List[TaxonomyRow]:
    """Group the open-coding notes into named problems, ranked by frequency x severity."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in coding:
        slug = row.get("problem_group")
        if not slug:
            continue
        groups.setdefault(slug, []).append(row)

    rows = []
    for slug, entries in groups.items():
        meta = PROBLEM_GROUPS[slug]
        severities = [entry["severity"] for entry in entries]
        rows.append(
            TaxonomyRow(
                slug=slug,
                name=meta["name"],
                description=meta["description"],
                count=len(entries),
                mean_severity=sum(severities) / len(severities),
                trace_ids=[entry["trace_id"] for entry in entries],
            )
        )
    rows.sort(key=lambda r: r.score, reverse=True)
    return rows


def pool_category_counts(pool: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in pool:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    return counts


def run_week5_report(
    config: Config,
    pool_path: Optional[Path] = None,
    coding_path: Optional[Path] = None,
    strategy: Optional[str] = None,
    index_dir: Optional[Path] = None,
    n: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Regenerate the traces, load the hand-written notes, rank the taxonomy, report."""
    repo_root = config.repo_root
    pool_path = Path(pool_path) if pool_path else repo_root / "eval" / "week5_questions.yaml"
    coding_path = Path(coding_path) if coding_path else repo_root / "eval" / "week5_open_coding.yaml"
    out_dir = Path(out_dir) if out_dir else config.paths.results_dir

    trace_set = run_trace_sample(
        config, pool_path=pool_path, strategy=strategy, index_dir=index_dir, n=n, seed=seed, out_dir=out_dir
    )
    coding = load_open_coding(coding_path)
    validate_open_coding(trace_set, coding)
    taxonomy = build_taxonomy(coding)
    pool = load_pool(pool_path)

    from .reporting import write_week5_artifacts

    write_week5_artifacts(
        trace_set,
        coding,
        taxonomy,
        pool_category_counts=pool_category_counts(pool),
        out_dir=out_dir,
        repo_root=repo_root,
    )
    return {"trace_set": trace_set, "coding": coding, "taxonomy": taxonomy}
