"""The refusal gate, citation validation and the extractive generator."""

import unittest

from ragchat.embeddings import HashingTfidfEmbedder
from ragchat.generator import ExtractiveGenerator
from ragchat.grounding import (
    INSUFFICIENT_CONTEXT,
    EvidenceGate,
    surface_forms,
    validate_citations,
)
from ragchat.models import Chunk, SearchHit

EMBEDDER = HashingTfidfEmbedder(n_features=4096)


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        strategy="test",
        position=0,
        char_start=0,
        char_end=len(text),
        section="Troubleshooting",
        metadata={
            "article_id": "HC-4002",
            "source_file": "hc-4002.md",
            "product_area": "payments",
            "last_updated": "2026-07-29",
        },
    )


def _hit(chunk: Chunk, rank: int = 1, score: float = 0.5) -> SearchHit:
    return SearchHit(rank=rank, score=score, chunk=chunk)


class SurfaceFormTests(unittest.TestCase):
    def test_plural_and_verb_forms_match(self):
        self.assertIn("delivery", surface_forms("deliveries"))
        self.assertIn("cause", surface_forms("causes"))
        self.assertIn("attempt", surface_forms("attempts"))
        self.assertIn("retry", surface_forms("retried"))

    def test_short_tokens_are_left_alone(self):
        self.assertEqual(surface_forms("sla"), {"sla"})


class EvidenceGateTests(unittest.TestCase):
    def setUp(self):
        self.corpus_terms = set(EMBEDDER.tokenize("err-4032 stored card token vault re-authorise billing migration"))

        def idf(term: str) -> float:
            # Terms the corpus has never seen carry the most weight.
            return 1.5 if term in self.corpus_terms else 5.0

        self.gate = EvidenceGate(
            min_top_score=0.005, min_evidence_coverage=0.55, tokenizer=EMBEDDER.tokenize, idf=idf
        )
        self.chunk = _chunk("c1", "ERR-4032 stored card token vault re-authorise billing migration")

    def test_supported_question_passes(self):
        verdict = self.gate.evaluate("What does ERR-4032 mean for a stored card token?", [_hit(self.chunk)])
        self.assertTrue(verdict.sufficient)

    def test_question_about_absent_topic_is_refused(self):
        verdict = self.gate.evaluate("What is the refund SLA for a disputed charge?", [_hit(self.chunk)])
        self.assertFalse(verdict.sufficient)
        self.assertIn("sla", verdict.missing_terms)

    def test_empty_retrieval_is_refused(self):
        verdict = self.gate.evaluate("anything at all", [])
        self.assertFalse(verdict.sufficient)

    def test_interrogative_words_do_not_count_as_missing_evidence(self):
        """"How long ... how many ... are made" must not be treated as absent content."""
        verdict = self.gate.evaluate(
            "How long does the ERR-4032 vault migration take and how many are made?", [_hit(self.chunk)]
        )
        for word in ("long", "many", "made"):
            self.assertNotIn(word, verdict.missing_terms)


class CitationValidationTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [_chunk("HC-4002#s-0001", "ERR-4032 means the token was not migrated.")]

    def test_resolvable_citation_is_accepted(self):
        report = validate_citations(
            "The token was not migrated to the new vault [HC-4002#s-0001].", self.chunks
        )
        self.assertTrue(report.valid)
        self.assertEqual(len(report.citations), 1)
        self.assertTrue(report.citations[0].resolved)

    def test_invented_chunk_id_is_flagged(self):
        report = validate_citations("Some claim about the migration vault [HC-9999#made-up].", self.chunks)
        self.assertFalse(report.valid)
        self.assertEqual(report.unresolved_ids, ["HC-9999#made-up"])

    def test_uncited_claim_is_flagged(self):
        report = validate_citations(
            "The customer must re-authorise the card before the next invoice runs.", self.chunks
        )
        self.assertFalse(report.valid)
        self.assertEqual(len(report.uncited_claims), 1)

    def test_citation_after_the_full_stop_still_counts(self):
        """Both "claim [id]." and "claim. [id]" are one cited claim."""
        report = validate_citations(
            "The token was not migrated to the new vault. [HC-4002#s-0001]", self.chunks
        )
        self.assertTrue(report.valid, report.uncited_claims)


class ExtractiveGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.generator = ExtractiveGenerator(tokenizer=EMBEDDER.tokenize)
        self.table_chunk = _chunk(
            "HC-4002#s-0001",
            "| Error code | Cause | Fix |\n|---|---|---|\n"
            "| ERR-4032 | token issued before 2026-05-01 | re-authorise the card |",
        )

    def test_table_row_is_rendered_with_its_header_labels(self):
        result = self.generator.generate("What does ERR-4032 mean and what is the fix?", [_hit(self.table_chunk)])
        self.assertFalse(result.refused)
        self.assertIn("Error code: ERR-4032", result.text)
        self.assertIn("Fix: re-authorise the card", result.text)
        self.assertIn("[HC-4002#s-0001]", result.text)

    def test_row_without_a_header_cannot_be_labelled(self):
        headerless = _chunk("HC-4002#b-0002", "| ERR-4032 | token issued before 2026-05-01 | re-authorise the card |")
        result = self.generator.generate("What does ERR-4032 mean and what is the fix?", [_hit(headerless)])
        self.assertFalse(result.refused)
        self.assertNotIn("Error code:", result.text)

    def test_refuses_when_nothing_matches(self):
        unrelated = _chunk("HC-4002#s-0009", "Bank debit mandates are migrated in full.")
        result = self.generator.generate("What is the refund SLA?", [_hit(unrelated)])
        self.assertTrue(result.refused)
        self.assertEqual(result.text, INSUFFICIENT_CONTEXT)

    def test_every_sentence_carries_a_citation(self):
        prose = _chunk(
            "HC-4002#s-0005",
            "Direct customers hitting ERR-4032 to Billing. They will need the physical card. "
            "The flow asks for the full card number.",
        )
        result = self.generator.generate("What do I tell a customer hitting ERR-4032?", [_hit(prose)])
        report = validate_citations(result.text, [prose])
        self.assertTrue(report.valid, report.uncited_claims)


if __name__ == "__main__":
    unittest.main()
