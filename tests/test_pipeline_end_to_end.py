"""End-to-end: ingest a temporary index, then answer, refuse and filter against it."""

import tempfile
import unittest
from pathlib import Path

from ragchat.config import Config
from ragchat.evaluation import evaluate_strategy, load_questions, run_filter_demo
from ragchat.indexer import ingest
from ragchat.pipeline import RAGPipeline

REPO = Path(__file__).resolve().parents[1]


class EndToEndTests(unittest.TestCase):
    """Builds a real index in a temp directory; no network, no API key needed."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.index_dir = Path(cls._tmp.name)
        cls.config = Config.load(REPO / "config.yaml")
        cls.questions = load_questions(REPO / "eval" / "questions.yaml")

        for strategy in ("baseline", "structure-aware"):
            ingest(cls.config, [cls.config.paths.legacy_articles_dir], strategy=strategy,
                   label="legacy", index_dir=cls.index_dir)
            ingest(cls.config, [cls.config.paths.articles_dir], strategy=strategy,
                   label="new-drop", index_dir=cls.index_dir)

        cls.pipeline = RAGPipeline.open(cls.config, strategy="structure-aware", index_dir=cls.index_dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # -- ingest ---------------------------------------------------------

    def test_ingest_is_append_only(self):
        history = self.pipeline.retriever.store.history
        self.assertEqual([event.label for event in history], ["legacy", "new-drop"])
        self.assertEqual(history[0].n_documents, 2)
        self.assertEqual(history[1].n_documents, 6, "only the six new articles should be in the second ingest")

    def test_every_indexed_chunk_has_a_source_file(self):
        for chunk in self.pipeline.retriever.store.chunks:
            self.assertTrue(chunk.source_file)
            self.assertTrue(chunk.product_area)
            self.assertTrue(chunk.last_updated)

    # -- answering ------------------------------------------------------

    def test_known_answer_questions_are_answered_with_resolvable_citations(self):
        for item in self.questions["answerable"][:3]:
            with self.subTest(item["id"]):
                answer = self.pipeline.ask(item["question"])
                self.assertFalse(answer.refused, f"{item['id']} was refused: {answer.refusal_reason}")
                self.assertTrue(answer.citations)
                for citation in answer.citations:
                    self.assertTrue(citation.resolved)
                    self.assertIsNotNone(self.pipeline.retriever.get_chunk(citation.chunk_id))

    def test_cited_chunk_actually_contains_the_known_answer(self):
        item = self.questions["answerable"][0]
        answer = self.pipeline.ask(item["question"])
        supported = False
        for citation in answer.citations:
            chunk = self.pipeline.retriever.get_chunk(citation.chunk_id)
            haystack = " ".join(chunk.text.split()).lower()
            if all(needle.lower() in haystack for needle in item["must_contain"]):
                supported = True
        self.assertTrue(supported, "no cited chunk contains the known answer text")

    def test_out_of_corpus_questions_are_refused(self):
        for item in self.questions["unanswerable"]:
            with self.subTest(item["id"]):
                answer = self.pipeline.ask(item["question"])
                self.assertTrue(answer.refused, f"{item['id']} was answered: {answer.text}")
                self.assertFalse(answer.citations)
                self.assertIn("cannot answer", answer.text.lower())

    def test_refusal_happens_before_generation(self):
        answer = self.pipeline.ask(self.questions["unanswerable"][0]["question"])
        self.assertIn("evidence", answer.diagnostics)
        self.assertNotIn("generation", answer.diagnostics, "the generator should not have been invoked")

    # -- retrieval quality ----------------------------------------------

    def test_hit_at_5_is_measured_over_the_same_questions(self):
        reports = {
            strategy: evaluate_strategy(self.config, strategy, self.questions, index_dir=self.index_dir)
            for strategy in ("baseline", "structure-aware")
        }
        for strategy, report in reports.items():
            with self.subTest(strategy):
                self.assertEqual(report.n_questions, 8)
                self.assertEqual(report.n_hits, 8)

    def test_structure_aware_keeps_more_answers_self_contained(self):
        baseline = evaluate_strategy(self.config, "baseline", self.questions, index_dir=self.index_dir)
        aware = evaluate_strategy(self.config, "structure-aware", self.questions, index_dir=self.index_dir)
        self.assertGreater(aware.n_self_contained, baseline.n_self_contained)
        self.assertEqual(aware.n_self_contained, 8)

    def test_metadata_filter_changes_the_top_result(self):
        demo = run_filter_demo(
            self.config, "structure-aware", self.questions["filter_demo"], index_dir=self.index_dir
        )
        self.assertTrue(demo.changed_top1)
        self.assertEqual(demo.filtered[0].chunk.product_area, "developer-api")
        for hit in demo.filtered:
            self.assertEqual(hit.chunk.product_area, "developer-api")

    def test_filtering_to_an_unknown_value_returns_nothing(self):
        hits = self.pipeline.search("retry window", filters={"product_area": "no-such-area"})
        self.assertEqual(hits, [])

    def test_search_is_deterministic(self):
        first = self.pipeline.search("What does ERR-4032 mean and what is the fix?")
        second = self.pipeline.search("What does ERR-4032 mean and what is the fix?")
        self.assertEqual([h.chunk.chunk_id for h in first], [h.chunk.chunk_id for h in second])
        self.assertEqual([round(h.score, 9) for h in first], [round(h.score, 9) for h in second])

    def test_empty_question_is_refused_not_crashed(self):
        answer = self.pipeline.ask("")
        self.assertTrue(answer.refused)


if __name__ == "__main__":
    unittest.main()
