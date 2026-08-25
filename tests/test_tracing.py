"""Week 5: sampling is deterministic and fair, traces are complete, taxonomy is real."""

import tempfile
import unittest
from pathlib import Path

from ragchat.config import Config
from ragchat.indexer import ingest
from ragchat.tracing import (
    DEFAULT_SEED,
    build_taxonomy,
    load_open_coding,
    load_pool,
    pool_category_counts,
    run_trace_sample,
    sample_questions,
    validate_open_coding,
)

REPO = Path(__file__).resolve().parents[1]


class PoolAndSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = load_pool(REPO / "eval" / "week5_questions.yaml")

    def test_pool_has_84_items_with_unique_ids(self):
        self.assertEqual(len(self.pool), 84)
        ids = [item["id"] for item in self.pool]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_pool_item_has_a_category_and_question(self):
        for item in self.pool:
            self.assertTrue(item.get("category"))
            self.assertTrue(item.get("question"))

    def test_error_code_tickets_cover_every_troubleshooting_row_once(self):
        error_items = [item for item in self.pool if item["category"] == "error_code_ticket"]
        self.assertEqual(len(error_items), 51, "one candidate per troubleshooting-table error code")
        codes = {item["must_contain"][0] for item in error_items}
        self.assertEqual(len(codes), 51, "no error code should be duplicated in the pool")

    def test_sample_is_deterministic_given_a_seed(self):
        first = sample_questions(self.pool, n=20, seed=DEFAULT_SEED)
        second = sample_questions(self.pool, n=20, seed=DEFAULT_SEED)
        self.assertEqual([item["id"] for item in first], [item["id"] for item in second])

    def test_different_seeds_draw_different_samples(self):
        a = sample_questions(self.pool, n=20, seed=1)
        b = sample_questions(self.pool, n=20, seed=2)
        self.assertNotEqual([item["id"] for item in a], [item["id"] for item in b])

    def test_sample_has_no_duplicates_and_correct_size(self):
        sample = sample_questions(self.pool, n=20, seed=DEFAULT_SEED)
        ids = [item["id"] for item in sample]
        self.assertEqual(len(ids), 20)
        self.assertEqual(len(ids), len(set(ids)))

    def test_sample_larger_than_pool_raises(self):
        with self.assertRaises(ValueError):
            sample_questions(self.pool, n=1000, seed=DEFAULT_SEED)

    def test_pool_category_counts_matches_manual_count(self):
        counts = pool_category_counts(self.pool)
        self.assertEqual(sum(counts.values()), len(self.pool))
        self.assertEqual(counts["error_code_ticket"], 51)


class OpenCodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = load_pool(REPO / "eval" / "week5_questions.yaml")
        cls.coding = load_open_coding(REPO / "eval" / "week5_open_coding.yaml")

    def test_coding_has_exactly_20_rows_matching_the_committed_sample(self):
        sampled = sample_questions(self.pool, n=20, seed=DEFAULT_SEED)
        sampled_ids = {item["id"] for item in sampled}
        coded_ids = {row["trace_id"] for row in self.coding}
        self.assertEqual(sampled_ids, coded_ids)

    def test_every_row_has_a_note_and_a_severity(self):
        for row in self.coding:
            self.assertTrue(row.get("note"), row["trace_id"])
            self.assertIn(row.get("severity"), (0, 1, 2, 3), row["trace_id"])

    def test_zero_severity_rows_have_no_problem_group(self):
        for row in self.coding:
            if row["severity"] == 0:
                self.assertIsNone(row.get("problem_group"), row["trace_id"])

    def test_taxonomy_is_ranked_by_score_descending(self):
        taxonomy = build_taxonomy(self.coding)
        scores = [row.score for row in taxonomy]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_taxonomy_covers_every_nonzero_severity_row(self):
        taxonomy = build_taxonomy(self.coding)
        grouped_ids = {tid for row in taxonomy for tid in row.trace_ids}
        expected_ids = {row["trace_id"] for row in self.coding if row["severity"] > 0}
        self.assertEqual(grouped_ids, expected_ids)

    def test_top_ranked_problem_is_the_false_refusal_group(self):
        """Documents this week's actual finding; a real drift here should be noticed."""
        taxonomy = build_taxonomy(self.coding)
        self.assertEqual(taxonomy[0].slug, "false_refusal_informal_phrasing")
        self.assertGreaterEqual(taxonomy[0].count, 8)


class TraceRunTests(unittest.TestCase):
    """Builds a real temp index and runs the actual sampler end-to-end. No network needed."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.index_dir = Path(cls._tmp.name)
        cls.config = Config.load(REPO / "config.yaml")
        for strategy in ("structure-aware",):
            ingest(cls.config, [cls.config.paths.legacy_articles_dir], strategy=strategy,
                   label="legacy", index_dir=cls.index_dir)
            ingest(cls.config, [cls.config.paths.articles_dir], strategy=strategy,
                   label="new-drop", index_dir=cls.index_dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_run_trace_sample_produces_one_record_per_sampled_question(self):
        with tempfile.TemporaryDirectory() as out:
            trace_set = run_trace_sample(
                self.config,
                strategy="structure-aware",
                index_dir=self.index_dir,
                n=5,
                seed=DEFAULT_SEED,
                out_dir=Path(out),
            )
            self.assertEqual(trace_set.n_traces, 5)
            self.assertEqual(trace_set.n_answered + trace_set.n_refused, 5)

    def test_trace_record_is_complete_enough_to_replay(self):
        """A trace must carry the question, the config that produced it, and the full answer."""
        with tempfile.TemporaryDirectory() as out:
            trace_set = run_trace_sample(
                self.config,
                strategy="structure-aware",
                index_dir=self.index_dir,
                n=3,
                seed=DEFAULT_SEED,
                out_dir=Path(out),
            )
            record = trace_set.records[0]
            payload = record.to_dict()
            for key in ("trace_id", "category", "question", "config", "answer"):
                self.assertIn(key, payload)
            self.assertIn("retrieval_mode", payload["config"])
            self.assertIn("strategy", payload["config"])
            for key in ("question", "refused", "text", "citations", "hits", "backend", "diagnostics"):
                self.assertIn(key, payload["answer"])

    def test_same_seed_reruns_produce_the_same_trace_ids(self):
        with tempfile.TemporaryDirectory() as out1, tempfile.TemporaryDirectory() as out2:
            first = run_trace_sample(
                self.config, strategy="structure-aware", index_dir=self.index_dir,
                n=6, seed=DEFAULT_SEED, out_dir=Path(out1),
            )
            second = run_trace_sample(
                self.config, strategy="structure-aware", index_dir=self.index_dir,
                n=6, seed=DEFAULT_SEED, out_dir=Path(out2),
            )
            self.assertEqual(
                [r.trace_id for r in first.records], [r.trace_id for r in second.records]
            )

    def test_committed_open_coding_matches_a_fresh_full_sample_run(self):
        """The 20 ids in eval/week5_open_coding.yaml must be exactly what the sampler draws today."""
        with tempfile.TemporaryDirectory() as out:
            trace_set = run_trace_sample(
                self.config, strategy="structure-aware", index_dir=self.index_dir,
                out_dir=Path(out),  # default n=20, seed=DEFAULT_SEED, default pool path
            )
            coding = load_open_coding(REPO / "eval" / "week5_open_coding.yaml")
            validate_open_coding(trace_set, coding)  # raises on any mismatch

    def test_writes_one_markdown_file_and_one_json_file_per_run(self):
        with tempfile.TemporaryDirectory() as out:
            out_dir = Path(out)
            trace_set = run_trace_sample(
                self.config, strategy="structure-aware", index_dir=self.index_dir,
                n=4, seed=DEFAULT_SEED, out_dir=out_dir,
            )
            self.assertTrue((out_dir / "week5_traces.json").is_file())
            md_files = sorted((out_dir / "week5_traces").glob("*.md"))
            self.assertEqual(len(md_files), 4)
            self.assertEqual(
                {f.stem for f in md_files}, {r.trace_id for r in trace_set.records}
            )


if __name__ == "__main__":
    unittest.main()
