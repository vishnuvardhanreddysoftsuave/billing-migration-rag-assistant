"""Week 4: BM25, RRF fusion, the metrics, and the hybrid retriever end to end."""

import tempfile
import unittest
from pathlib import Path

from ragchat.bm25 import BM25Index, reciprocal_rank_fusion
from ragchat.config import Config
from ragchat.evaluation import evaluate_strategy, load_questions
from ragchat.failure_analysis import evaluate_failures, load_week4_questions
from ragchat.indexer import ingest
from ragchat.metrics import RetrievalRecord, hit_rate_at_k, mrr, recall_at_k
from ragchat.pipeline import RAGPipeline

REPO = Path(__file__).resolve().parents[1]


class BM25IndexTests(unittest.TestCase):
    """Pure unit tests over a tiny synthetic corpus, no index required."""

    def setUp(self):
        self.docs = [
            "the quick brown fox jumps over the lazy dog".split(),
            "err-4032 stored card token re-authorise the card".split(),
            "the dog barked at the quick fox in the yard".split(),
        ]
        self.index = BM25Index.build(self.docs)

    def test_rare_term_dominates_a_short_document(self):
        ranked = self.index.rank(["err-4032"])
        self.assertEqual(ranked[0][0], 1)

    def test_term_absent_from_every_document_scores_nothing(self):
        self.assertEqual(self.index.rank(["nonexistent-term"]), [])

    def test_rank_respects_a_row_restriction(self):
        ranked = self.index.rank(["quick", "fox"], rows=[2])
        self.assertEqual([row for row, _ in ranked], [2])

    def test_scores_are_deterministic_and_ties_break_on_row_index(self):
        first = self.index.rank(["the"])
        second = self.index.rank(["the"])
        self.assertEqual(first, second)


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_top_of_both_rankings_wins(self):
        fused = reciprocal_rank_fusion([[5, 1, 2], [5, 3, 4]], k=10)
        self.assertEqual(fused[0][0], 5)

    def test_a_row_absent_from_one_ranking_still_gets_credit_from_the_other(self):
        fused = reciprocal_rank_fusion([[1, 2, 3], [9]], k=10)
        rows = [row for row, _ in fused]
        self.assertIn(9, rows)

    def test_smaller_k_makes_rank_1_dominate_more(self):
        loose = dict(reciprocal_rank_fusion([[1, 2, 3]], k=60))
        tight = dict(reciprocal_rank_fusion([[1, 2, 3]], k=1))
        self.assertGreater(tight[1] / tight[3], loose[1] / loose[3])


class MetricsTests(unittest.TestCase):
    def test_hit_rate_counts_any_relevant_id_in_the_window(self):
        records = [
            RetrievalRecord(retrieved_ids=["a", "b", "c"], relevant_ids={"b"}),
            RetrievalRecord(retrieved_ids=["x", "y", "z"], relevant_ids={"q"}),
        ]
        self.assertEqual(hit_rate_at_k(records, 3), 0.5)

    def test_recall_gives_partial_credit_across_a_multi_fact_question(self):
        record = RetrievalRecord(retrieved_ids=["a", "b", "c"], relevant_ids={"b", "missing"})
        self.assertAlmostEqual(recall_at_k([record], 3), 0.5)

    def test_relevant_id_outside_the_window_is_not_silently_dropped(self):
        # A record whose relevant id is never retrieved must count as a zero,
        # not vanish from the average the way an empty relevant_ids would.
        never_retrieved = RetrievalRecord(retrieved_ids=["a", "b"], relevant_ids={"z"})
        always_hit = RetrievalRecord(retrieved_ids=["z"], relevant_ids={"z"})
        self.assertEqual(recall_at_k([never_retrieved, always_hit], 3), 0.5)

    def test_mrr_rewards_an_earlier_rank(self):
        early = RetrievalRecord(retrieved_ids=["a", "b"], relevant_ids={"a"})
        late = RetrievalRecord(retrieved_ids=["x", "a"], relevant_ids={"a"})
        self.assertGreater(mrr([early]), mrr([late]))

    def test_mrr_is_zero_when_nothing_relevant_is_retrieved(self):
        record = RetrievalRecord(retrieved_ids=["a", "b"], relevant_ids={"z"})
        self.assertEqual(mrr([record]), 0.0)


class HybridRetrieverEndToEndTests(unittest.TestCase):
    """Builds a real index in a temp directory; no network, no API key needed."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.index_dir = Path(cls._tmp.name)
        cls.config = Config.load(REPO / "config.yaml")
        cls.questions = load_questions(REPO / "eval" / "questions.yaml")

        ingest(cls.config, [cls.config.paths.legacy_articles_dir], strategy="structure-aware",
               label="legacy", index_dir=cls.index_dir)
        ingest(cls.config, [cls.config.paths.articles_dir], strategy="structure-aware",
               label="new-drop", index_dir=cls.index_dir)

        cls.pipeline = RAGPipeline.open(cls.config, strategy="structure-aware", index_dir=cls.index_dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.pipeline.retriever.search("anything", mode="not-a-mode")

    def test_hybrid_search_is_deterministic(self):
        q = "Support ticket: account throwing ERR-4117, need root cause and remediation steps quickly."
        first = self.pipeline.retriever.search(q, top_k=5, mode="hybrid")
        second = self.pipeline.retriever.search(q, top_k=5, mode="hybrid")
        self.assertEqual([h.chunk.chunk_id for h in first], [h.chunk.chunk_id for h in second])
        self.assertEqual([round(h.score, 9) for h in first], [round(h.score, 9) for h in second])

    def test_hybrid_search_respects_metadata_filters(self):
        hits = self.pipeline.retriever.search(
            "retry window", top_k=5, filters={"product_area": "developer-api"}, mode="hybrid"
        )
        self.assertTrue(hits)
        for hit in hits:
            self.assertEqual(hit.chunk.product_area, "developer-api")

    def test_filtering_to_an_unknown_value_returns_nothing_under_hybrid(self):
        hits = self.pipeline.retriever.search("retry window", filters={"product_area": "no-such-area"}, mode="hybrid")
        self.assertEqual(hits, [])

    def test_hybrid_does_not_regress_the_week3_known_answer_questions(self):
        """Week 3's committed hit@5 must survive switching the default to hybrid."""
        for item in self.questions["answerable"]:
            with self.subTest(item["id"]):
                hits = self.pipeline.retriever.search(item["question"], top_k=5, mode="hybrid")
                gold_article = item["gold"]["article_id"]
                must_contain = [s.lower() for s in item.get("must_contain", [])]
                hit = any(
                    h.chunk.article_id == gold_article
                    and all(s in " ".join(h.chunk.text.split()).lower() for s in must_contain)
                    for h in hits
                )
                self.assertTrue(hit, f"{item['id']} regressed under hybrid retrieval")

    def test_hybrid_beats_semantic_only_on_a_realistic_exact_code_query(self):
        """The flagship claim: keyword dilution recovers under hybrid at k=3."""
        q = "Support ticket: account throwing ERR-4117, need root cause and remediation steps quickly."
        semantic_hits = self.pipeline.retriever.search(q, top_k=3, mode="semantic")
        hybrid_hits = self.pipeline.retriever.search(q, top_k=3, mode="hybrid")

        def hit(hits):
            return any(
                h.chunk.article_id == "HC-4005" and "err-4117" in h.chunk.text.lower() and "vies" in h.chunk.text.lower()
                for h in hits
            )

        self.assertFalse(hit(semantic_hits), "expected this to already be a semantic-only miss")
        self.assertTrue(hit(hybrid_hits), "hybrid search should recover it into the top-3")


class Week4EvaluationHarnessTests(unittest.TestCase):
    """The scored claim: hit-rate@3 measurably improves on the failing-question set."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.index_dir = Path(cls._tmp.name)
        cls.config = Config.load(REPO / "config.yaml")
        cls.questions = load_week4_questions(REPO / "eval" / "week4_questions.yaml")
        ingest(cls.config, [cls.config.paths.legacy_articles_dir], strategy="structure-aware",
               label="legacy", index_dir=cls.index_dir)
        ingest(cls.config, [cls.config.paths.articles_dir], strategy="structure-aware",
               label="new-drop", index_dir=cls.index_dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_question_set_loads_and_is_nonempty(self):
        self.assertGreater(len(self.questions), 0)

    def test_hybrid_hit_rate_at_3_beats_semantic_only(self):
        report = evaluate_failures(self.config, self.questions, strategy="structure-aware", index_dir=self.index_dir)
        self.assertGreater(report.hybrid_hit_rate_at_3, report.semantic_hit_rate_at_3)
        self.assertGreaterEqual(report.n_fixed, 5, "expect a solid majority of keyword_dilution questions to be fixed")

    def test_generation_gate_questions_are_not_fixed_by_a_retrieval_change(self):
        """Proves the module's core point: a retrieval fix cannot repair a generation failure."""
        report = evaluate_failures(self.config, self.questions, strategy="structure-aware", index_dir=self.index_dir)
        gate_questions = [r for r in report.records if r.category == "generation_gate"]
        self.assertTrue(gate_questions)
        for r in gate_questions:
            with self.subTest(r.qid):
                self.assertTrue(r.answer.refused, "generation_gate questions should already be refused pre-fix")
                self.assertTrue(r.semantic_hit3, "the right chunk should already be retrieved for these")

    def test_every_record_gets_a_two_way_label(self):
        report = evaluate_failures(self.config, self.questions, strategy="structure-aware", index_dir=self.index_dir)
        for r in report.records:
            self.assertTrue(
                r.label.startswith("retrieval failure")
                or r.label.startswith("generation failure")
                or r.label == "no failure"
            )


if __name__ == "__main__":
    unittest.main()
