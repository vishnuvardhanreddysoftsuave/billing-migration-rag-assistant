"""Answer generation backends.

Two backends, both grounded and both able to refuse:

* :class:`AnthropicGenerator` — Claude, used when credentials are available.
* :class:`ExtractiveGenerator` — deterministic, offline, copies evidence
  verbatim from the retrieved chunks. It cannot invent text by construction.

Neither backend is trusted: whatever comes back is citation-validated by
:mod:`ragchat.grounding` before it reaches the caller.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

from .config import Config, anthropic_credentials_present
from .grounding import INSUFFICIENT_CONTEXT, QUESTION_WORDS, SENTENCE_RE
from .models import SearchHit

IDENTIFIER_RE = re.compile(r"^[a-z]{2,}-[0-9]{2,}$")
TABLE_SEPARATOR_RE = re.compile(r"^\|[\s:|-]+\|$")

SYSTEM_PROMPT = """You are a support-desk assistant answering from an internal help-centre index.

Rules, in priority order:
1. Use ONLY the numbered context chunks in the user message. You have no other knowledge.
2. Every sentence that states a fact must end with a citation in square brackets naming the
   chunk it came from, exactly as given, for example [HC-4002#structure-aware-0007].
3. Only cite chunk_ids that appear in the context. Never invent, abbreviate or merge ids.
4. If the context does not contain the answer, reply with exactly INSUFFICIENT_CONTEXT and
   nothing else. Do not approximate, do not reason from adjacent facts, and do not offer a
   partial answer with a caveat.
5. Prefer quoting the exact figure, error code, date or step from the context over
   paraphrasing it.

Answer in at most six sentences."""


@dataclass
class GenerationResult:
    text: str
    refused: bool
    reason: str
    backend: str
    diagnostics: Dict[str, Any]


class Generator(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, question: str, hits: Sequence[SearchHit]) -> GenerationResult:
        ...


# --------------------------------------------------------------------------
# Deterministic offline backend
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceUnit:
    text: str
    chunk_id: str
    score: float
    kind: str


class ExtractiveGenerator(Generator):
    """Selects the best-matching sentences and table rows, verbatim, with citations."""

    name = "extractive"

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        max_units: int = 3,
        min_score: float = 0.2,
        relative_cutoff: float = 0.6,
    ) -> None:
        self._tokenize = tokenizer
        self.max_units = max_units
        self.min_score = min_score
        # Units scoring far below the best one are noise, not supporting evidence.
        self.relative_cutoff = relative_cutoff

    def generate(self, question: str, hits: Sequence[SearchHit]) -> GenerationResult:
        # Same interrogative filter the evidence gate uses, so a table header cell
        # reading "What happens" cannot out-score the row that actually answers.
        question_terms = {t for t in self._tokenize(question) if t not in QUESTION_WORDS}
        identifiers = {term for term in question_terms if IDENTIFIER_RE.match(term)}

        units: List[EvidenceUnit] = []
        for hit in hits:
            for text, kind in _evidence_units(hit.chunk.text):
                score = self._score(text, question_terms, identifiers)
                if score >= self.min_score:
                    units.append(EvidenceUnit(text=text, chunk_id=hit.chunk.chunk_id, score=score, kind=kind))

        units.sort(key=lambda u: (-u.score, len(u.text)))
        selected: List[EvidenceUnit] = []
        seen: set[str] = set()
        floor = units[0].score * self.relative_cutoff if units else 0.0
        for unit in units:
            if unit.score < floor:
                break
            key = " ".join(unit.text.lower().split())
            if key in seen:
                continue
            seen.add(key)
            selected.append(unit)
            if len(selected) >= self.max_units:
                break

        if not selected:
            return GenerationResult(
                text=INSUFFICIENT_CONTEXT,
                refused=True,
                reason="no retrieved sentence or table row matched the question closely enough",
                backend=self.name,
                diagnostics={"candidate_units": len(units)},
            )

        lines = [_attach_citation(unit.text, unit.chunk_id) for unit in selected]
        return GenerationResult(
            text="\n".join(lines),
            refused=False,
            reason="",
            backend=self.name,
            diagnostics={"units_used": len(selected), "top_unit_score": round(selected[0].score, 3)},
        )

    def _score(self, text: str, question_terms: set[str], identifiers: set[str]) -> float:
        if not question_terms:
            return 0.0
        unit_terms = set(self._tokenize(text))
        overlap = len(question_terms & unit_terms) / len(question_terms)
        if identifiers and identifiers & unit_terms:
            overlap += 0.5
        return overlap


def _evidence_units(chunk_text: str) -> List[Tuple[str, str]]:
    """Split a chunk into citable units: rendered table rows and prose sentences."""
    units: List[Tuple[str, str]] = []
    prose: List[str] = []
    header: List[str] | None = None
    lines = chunk_text.splitlines()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|"):
            if TABLE_SEPARATOR_RE.match(stripped):
                continue
            cells = _table_cells(stripped)
            is_header = i + 1 < len(lines) and TABLE_SEPARATOR_RE.match(lines[i + 1].strip()) is not None
            if is_header:
                header = cells
                continue
            units.append((_render_row(cells, header), "table_row"))
            continue

        if stripped.startswith("#") or not stripped:
            if prose:
                units.extend((s, "prose") for s in _sentences(" ".join(prose)))
                prose = []
            continue
        prose.append(stripped)

    if prose:
        units.extend((s, "prose") for s in _sentences(" ".join(prose)))
    return [(text, kind) for text, kind in units if len(text.strip()) > 2]


def _table_cells(line: str) -> List[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _render_row(cells: Sequence[str], header: Sequence[str] | None) -> str:
    """Render a table row as a sentence, using the header row for the labels.

    When the header is missing — which is exactly what happens when a chunker
    splits a table away from its header — the row is rendered without labels and
    reads as a bare list of cells.
    """
    if header and len(header) == len(cells):
        return "; ".join(f"{label}: {value}" for label, value in zip(header, cells) if value)
    return " | ".join(cell for cell in cells if cell)


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in SENTENCE_RE.split(text) if len(s.strip()) > 2]


def _attach_citation(text: str, chunk_id: str) -> str:
    """Cite every sentence, placing the citation inside the terminal punctuation."""
    parts = [p.strip() for p in SENTENCE_RE.split(text) if p.strip()]
    cited = []
    for part in parts:
        if part.endswith(f"[{chunk_id}]"):
            cited.append(part)
        elif part[-1:] in ".!?":
            cited.append(f"{part[:-1].rstrip()} [{chunk_id}]{part[-1]}")
        else:
            cited.append(f"{part} [{chunk_id}]")
    return "- " + " ".join(cited)


# --------------------------------------------------------------------------
# Claude backend
# --------------------------------------------------------------------------


class AnthropicGenerator(Generator):
    """Grounded generation with Claude."""

    name = "anthropic"

    def __init__(self, model: str, max_tokens: int, effort: str) -> None:
        try:
            import anthropic  # imported lazily so the offline path needs no SDK
        except ImportError as exc:  # pragma: no cover - exercised only without the SDK
            raise RuntimeError("the anthropic package is required for the anthropic backend") from exc
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort

    def generate(self, question: str, hits: Sequence[SearchHit]) -> GenerationResult:
        anthropic = self._anthropic
        prompt = _build_user_prompt(question, hits)

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": prompt}],
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
            )
        except anthropic.RateLimitError as exc:
            return self._error("rate limited by the Claude API", exc)
        except anthropic.APIStatusError as exc:
            return self._error(f"Claude API error {exc.status_code}", exc)
        except anthropic.APIConnectionError as exc:
            return self._error("could not reach the Claude API", exc)

        # Check the stop reason before touching content: a refusal or a truncated
        # response must never be presented as an answer.
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            return GenerationResult(
                text="",
                refused=True,
                reason=f"the model declined this request (category: {category})",
                backend=self.name,
                diagnostics={"stop_reason": response.stop_reason},
            )
        if response.stop_reason == "max_tokens":
            return GenerationResult(
                text="",
                refused=True,
                reason="the model response was truncated at max_tokens, so it cannot be trusted",
                backend=self.name,
                diagnostics={"stop_reason": response.stop_reason},
            )

        text = "\n".join(block.text for block in response.content if block.type == "text").strip()
        diagnostics = {
            "stop_reason": response.stop_reason,
            "model": response.model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        if not text or text.strip().upper().startswith(INSUFFICIENT_CONTEXT):
            return GenerationResult(
                text=INSUFFICIENT_CONTEXT,
                refused=True,
                reason="the model reported that the retrieved context does not contain the answer",
                backend=self.name,
                diagnostics=diagnostics,
            )
        return GenerationResult(text=text, refused=False, reason="", backend=self.name, diagnostics=diagnostics)

    def _error(self, reason: str, exc: Exception) -> GenerationResult:
        return GenerationResult(
            text="",
            refused=True,
            reason=f"{reason}: {exc}",
            backend=self.name,
            diagnostics={"error": type(exc).__name__},
        )


def _build_user_prompt(question: str, hits: Sequence[SearchHit]) -> str:
    blocks: List[str] = []
    for hit in hits:
        chunk = hit.chunk
        blocks.append(
            "\n".join(
                [
                    f"[{chunk.chunk_id}]",
                    f"article_id: {chunk.article_id}",
                    f"source_file: {chunk.source_file}",
                    f"section: {chunk.section or '(none)'}",
                    "text:",
                    chunk.text,
                ]
            )
        )
    context = "\n\n---\n\n".join(blocks) if blocks else "(no chunks retrieved)"
    return f"Context chunks:\n\n{context}\n\n---\n\nQuestion: {question}"


def build_generator(config: Config, tokenizer: Callable[[str], List[str]]) -> Generator:
    """Pick a backend from config; ``auto`` prefers Claude when credentials exist."""
    backend = config.generation.backend
    if backend == "auto":
        backend = "anthropic" if anthropic_credentials_present() else "extractive"

    if backend == "anthropic":
        return AnthropicGenerator(
            model=config.generation.model,
            max_tokens=config.generation.max_tokens,
            effort=config.generation.effort,
        )
    if backend == "extractive":
        return ExtractiveGenerator(tokenizer=tokenizer)
    raise ValueError(f"unknown generation backend {backend!r}; use auto, anthropic or extractive")
