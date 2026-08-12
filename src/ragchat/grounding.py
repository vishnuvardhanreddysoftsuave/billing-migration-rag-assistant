"""The refusal gate and citation validation.

Refusal here is *forced*, not suggested. Two independent mechanisms enforce it:

1. An evidence gate that runs **before** generation. If retrieval did not clear
   the score and term-coverage thresholds, the question is refused and no
   generator is invoked at all.
2. Citation validation that runs **after** generation. Every factual sentence
   must carry a ``[chunk_id]`` that resolves to a chunk actually retrieved for
   this question; anything else fails closed into a refusal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from .models import Chunk, Citation, SearchHit

CITATION_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_\-.#]*)\]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
# "... card". [ID]" and "... card [ID]." are the same claim; normalise the first
# form into the second before splitting so a trailing citation is not orphaned.
TRAILING_CITATION_RE = re.compile(r"([.!?])\s*((?:\[[A-Za-z0-9][A-Za-z0-9_\-.#]*\]\s*)+)")
INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"

# Interrogative scaffolding. These carry no information the corpus is expected to
# contain: "how *long* ... how *many* ... are *made*" says nothing about the topic,
# so counting them as missing evidence made well-supported questions look unanswerable.
QUESTION_WORDS = frozenset(
    """
    what why how when where which who whom whose mean means meaning long many much
    tell need know happen happens do does did done get got give gives make makes made
    use used using way ways thing things say says tell tells
    """.split()
)


@dataclass
class EvidenceVerdict:
    """Outcome of the pre-generation gate."""

    sufficient: bool
    reason: str
    top_score: float
    coverage: float
    matched_terms: List[str] = field(default_factory=list)
    missing_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "reason": self.reason,
            "top_score": round(self.top_score, 4),
            "idf_weighted_coverage": round(self.coverage, 4),
            "matched_terms": self.matched_terms,
            "missing_terms": self.missing_terms,
        }


class EvidenceGate:
    """Decides whether the retrieved context is strong enough to answer at all.

    The decisive signal is **idf-weighted term coverage**: what share of the
    question's information content, measured in idf, is actually present in the
    retrieved chunks. Raw term counts are useless here because a question like
    "what is the refund SLA during the billing migration" shares its common
    words ("billing", "migration") with the whole corpus while every term that
    makes it specific ("refund", "SLA") is absent. Weighting by idf makes the
    missing distinctive terms dominate, so the question is refused.

    ``min_top_score`` is only a floor for "nothing matched at all" — the
    similarity scale depends on document length, so it is not a tuning knob.
    """

    def __init__(self, min_top_score: float, min_evidence_coverage: float, tokenizer, idf) -> None:
        self.min_top_score = float(min_top_score)
        self.min_evidence_coverage = float(min_evidence_coverage)
        self._tokenize = tokenizer
        self._idf = idf

    def evaluate(self, question: str, hits: Sequence[SearchHit]) -> EvidenceVerdict:
        if not hits:
            return EvidenceVerdict(False, "no chunk matched the question", 0.0, 0.0)

        top_score = hits[0].score
        question_terms = [t for t in _unique(self._tokenize(question)) if t not in QUESTION_WORDS]
        context_forms: Set[str] = set()
        for hit in hits:
            for token in self._tokenize(hit.chunk.text):
                context_forms.update(surface_forms(token))

        matched = [term for term in question_terms if surface_forms(term) & context_forms]
        missing = [term for term in question_terms if not (surface_forms(term) & context_forms)]
        total_weight = sum(self._idf(term) for term in question_terms)
        matched_weight = sum(self._idf(term) for term in matched)
        coverage = (matched_weight / total_weight) if total_weight else 0.0

        if top_score < self.min_top_score:
            reason = f"best retrieval score {top_score:.4f} is below the floor {self.min_top_score:.4f}"
            return EvidenceVerdict(False, reason, top_score, coverage, matched, missing)

        if coverage < self.min_evidence_coverage:
            reason = (
                f"only {coverage:.0%} of the question's information content (idf-weighted) "
                f"appears in the retrieved context, below the {self.min_evidence_coverage:.0%} "
                f"minimum; absent from the corpus: {', '.join(missing[:6]) or 'n/a'}"
            )
            return EvidenceVerdict(False, reason, top_score, coverage, matched, missing)

        return EvidenceVerdict(
            True, "retrieved context clears the evidence thresholds", top_score, coverage, matched, missing
        )


@dataclass
class CitationReport:
    """Result of checking an answer's citations against the retrieved chunks."""

    citations: List[Citation]
    unresolved_ids: List[str]
    uncited_claims: List[str]

    @property
    def valid(self) -> bool:
        return not self.unresolved_ids and not self.uncited_claims

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citations": [c.to_dict() for c in self.citations],
            "unresolved_ids": self.unresolved_ids,
            "uncited_claims": self.uncited_claims,
            "valid": self.valid,
        }


def validate_citations(text: str, allowed: Iterable[Chunk]) -> CitationReport:
    """Check every ``[chunk_id]`` resolves and every claim carries one."""
    by_id = {chunk.chunk_id: chunk for chunk in allowed}
    citations: List[Citation] = []
    unresolved: List[str] = []
    seen: Set[str] = set()

    for chunk_id in CITATION_RE.findall(text):
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        chunk = by_id.get(chunk_id)
        if chunk is None:
            unresolved.append(chunk_id)
            citations.append(
                Citation(chunk_id=chunk_id, article_id="", source_file="", section="", resolved=False)
            )
        else:
            citations.append(
                Citation(
                    chunk_id=chunk_id,
                    article_id=chunk.article_id,
                    source_file=chunk.source_file,
                    section=chunk.section,
                    resolved=True,
                    quote=_first_line(chunk.text),
                )
            )

    uncited = [claim for claim in _claim_sentences(text) if not CITATION_RE.search(claim)]
    return CitationReport(citations=citations, unresolved_ids=unresolved, uncited_claims=uncited)


def _claim_sentences(text: str) -> List[str]:
    """Sentences that assert something and therefore need a citation."""
    claims: List[str] = []
    for line in _normalize_citation_placement(text).splitlines():
        stripped = line.strip().lstrip("-*0123456789. ").strip()
        if len(stripped) < 25:
            # Headers, list bullets and short connectors are not standalone claims.
            continue
        for sentence in SENTENCE_RE.split(stripped):
            sentence = sentence.strip()
            if len(sentence) >= 25:
                claims.append(sentence)
    return claims


def surface_forms(token: str) -> Set[str]:
    """A token plus its common inflections, so ``deliveries`` matches ``delivery``.

    Deliberately a candidate set rather than a stemmer: matching succeeds when two
    tokens share any form, which avoids the classic stemmer failure where
    ``causes`` -> ``caus`` no longer matches ``cause``.
    """
    forms = {token}
    if len(token) > 4:
        if token.endswith("ies"):
            forms.add(token[:-3] + "y")
        if token.endswith("ied"):
            forms.add(token[:-3] + "y")
        if token.endswith("ing"):
            forms.add(token[:-3])
            forms.add(token[:-3] + "e")
        if token.endswith("ed"):
            forms.add(token[:-2])
            forms.add(token[:-1])
        if token.endswith("es"):
            forms.add(token[:-2])
            forms.add(token[:-1])
        elif token.endswith("s"):
            forms.add(token[:-1])
    return forms


def _normalize_citation_placement(text: str) -> str:
    """Move a citation that trails a full stop back inside the sentence."""
    return TRAILING_CITATION_RE.sub(lambda m: " " + m.group(2).strip() + m.group(1), text)


def _first_line(text: str, limit: int = 220) -> str:
    line = " ".join(text.split())
    return line[:limit] + ("…" if len(line) > limit else "")


def _unique(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
