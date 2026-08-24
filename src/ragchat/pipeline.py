"""End-to-end RAG pipeline: retrieve -> gate -> generate -> validate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import Config
from .generator import Generator, build_generator
from .grounding import EvidenceGate, validate_citations
from .models import Answer, SearchHit
from .retriever import Retriever


class RAGPipeline:
    """Answers a question against one index namespace, or refuses."""

    def __init__(self, config: Config, retriever: Retriever, generator: Generator) -> None:
        self.config = config
        self.retriever = retriever
        self.generator = generator
        self.gate = EvidenceGate(
            min_top_score=config.grounding.min_top_score,
            min_evidence_coverage=config.grounding.min_evidence_coverage,
            tokenizer=retriever.embedder.tokenize,
            idf=retriever.idf,
        )

    @classmethod
    def open(
        cls,
        config: Config,
        strategy: str | None = None,
        index_dir: Path | None = None,
        generator: Generator | None = None,
    ) -> "RAGPipeline":
        retriever = Retriever.open(config, strategy=strategy, index_dir=index_dir)
        generator = generator or build_generator(config, retriever.embedder.tokenize)
        return cls(config=config, retriever=retriever, generator=generator)

    @property
    def namespace(self) -> str:
        return self.retriever.namespace

    def search(
        self,
        question: str,
        top_k: int | None = None,
        filters: Dict[str, Any] | None = None,
        mode: str | None = None,
    ) -> List[SearchHit]:
        return self.retriever.search(
            question,
            top_k=top_k or self.config.retrieval.top_k,
            filters=filters,
            mode=mode or self.config.retrieval.mode,
        )

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        filters: Dict[str, Any] | None = None,
        mode: str | None = None,
    ) -> Answer:
        effective_mode = mode or self.config.retrieval.mode
        hits = self.search(question, top_k=top_k, filters=filters, mode=effective_mode)
        context = hits[: self.config.generation.max_context_chunks]

        verdict = self.gate.evaluate(question, context)
        if not verdict.sufficient:
            return self._refuse(
                question,
                hits,
                verdict.reason,
                {
                    "evidence": verdict.to_dict(),
                    "namespace": self.namespace,
                    "filters": filters or {},
                    "retrieval_mode": effective_mode,
                },
            )

        result = self.generator.generate(question, context)
        diagnostics = {
            "evidence": verdict.to_dict(),
            "generation": result.diagnostics,
            "namespace": self.namespace,
            "filters": filters or {},
            "retrieval_mode": effective_mode,
        }
        if result.refused:
            return self._refuse(question, hits, result.reason, diagnostics, backend=result.backend)

        report = validate_citations(result.text, [hit.chunk for hit in context])
        diagnostics["citations"] = report.to_dict()
        if not report.valid:
            reason = _citation_failure_reason(report)
            return self._refuse(question, hits, reason, diagnostics, backend=result.backend)

        return Answer(
            question=question,
            refused=False,
            text=result.text,
            citations=report.citations,
            hits=hits,
            backend=result.backend,
            diagnostics=diagnostics,
        )

    def _refuse(
        self,
        question: str,
        hits: Sequence[SearchHit],
        reason: str,
        diagnostics: Optional[dict] = None,
        backend: str = "",
    ) -> Answer:
        return Answer(
            question=question,
            refused=True,
            text=self.config.grounding.refusal_message,
            citations=[],
            hits=list(hits),
            backend=backend or self.generator.name,
            refusal_reason=reason,
            diagnostics=diagnostics or {},
        )


def _citation_failure_reason(report) -> str:
    parts = []
    if report.unresolved_ids:
        parts.append(f"answer cited chunk ids that are not in the retrieved context: {report.unresolved_ids}")
    if report.uncited_claims:
        parts.append(f"{len(report.uncited_claims)} claim(s) carried no citation")
    return "; ".join(parts)
