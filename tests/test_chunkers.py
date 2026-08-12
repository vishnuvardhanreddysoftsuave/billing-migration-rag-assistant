"""Chunker behaviour, including the property this week's work exists to guarantee."""

import unittest
from pathlib import Path

from ragchat.chunkers import available_strategies, get_chunker
from ragchat.evaluation import contains_table_rows, has_table_header
from ragchat.loader import load_document
from ragchat.models import REQUIRED_METADATA

ARTICLES = Path(__file__).resolve().parents[1] / "data" / "articles"

TABLE_DOC = """---
article_id: T-1
product_area: testing
last_updated: 2026-01-01
---

# Title

## Errors

| Error code | Cause | Fix |
|---|---|---|
| ERR-1 | cause one that is fairly wordy so the table grows past the budget | fix one |
| ERR-2 | cause two that is fairly wordy so the table grows past the budget | fix two |
| ERR-3 | cause three that is fairly wordy so the table grows past the budget | fix three |
| ERR-4 | cause four that is fairly wordy so the table grows past the budget | fix four |
| ERR-5 | cause five that is fairly wordy so the table grows past the budget | fix five |
| ERR-6 | cause six that is fairly wordy so the table grows past the budget | fix six |
"""


def _write(tmpdir: Path, text: str) -> Path:
    path = tmpdir / "doc.md"
    path.write_text(text, encoding="utf-8")
    return path


class ChunkerRegistryTests(unittest.TestCase):
    def test_both_strategies_registered(self):
        self.assertIn("baseline", available_strategies())
        self.assertIn("structure-aware", available_strategies())

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            get_chunker("does-not-exist", 800, 100)

    def test_overlap_must_be_smaller_than_size(self):
        with self.assertRaises(ValueError):
            get_chunker("baseline", 100, 100)


class StructureAwareTableTests(unittest.TestCase):
    """The headline guarantee: a table row is never orphaned from its header."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_split_table_repeats_the_header(self):
        doc = load_document(_write(self.tmpdir, TABLE_DOC))
        chunks = get_chunker("structure-aware", 400, 50).split(doc)
        table_chunks = [c for c in chunks if contains_table_rows(c.text)]

        self.assertGreater(len(table_chunks), 1, "table should have been split for this test to mean anything")
        for chunk in table_chunks:
            self.assertTrue(
                has_table_header(chunk.text),
                f"chunk {chunk.chunk_id} carries table rows without a header row",
            )

    def test_rows_are_never_cut_in_half(self):
        doc = load_document(_write(self.tmpdir, TABLE_DOC))
        for chunk in get_chunker("structure-aware", 400, 50).split(doc):
            for line in chunk.text.splitlines():
                stripped = line.strip()
                if stripped.startswith("| ERR-"):
                    self.assertTrue(stripped.endswith("|"), f"row was cut: {stripped!r}")

    def test_baseline_does_orphan_rows(self):
        """Documents the failure the structure-aware chunker fixes."""
        doc = load_document(_write(self.tmpdir, TABLE_DOC))
        chunks = get_chunker("baseline", 400, 50).split(doc)
        orphaned = [c for c in chunks if contains_table_rows(c.text) and not has_table_header(c.text)]
        self.assertTrue(orphaned, "baseline is expected to orphan rows; if not, the comparison is meaningless")


class RealCorpusTests(unittest.TestCase):
    def setUp(self):
        self.documents = [load_document(p) for p in sorted(ARTICLES.glob("*.md"))]
        self.assertEqual(len(self.documents), 6, "expected the six new help-centre articles")

    def test_structure_aware_never_orphans_across_the_corpus(self):
        for document in self.documents:
            for chunk in get_chunker("structure-aware", 800, 120).split(document):
                if contains_table_rows(chunk.text):
                    self.assertTrue(
                        has_table_header(chunk.text),
                        f"{chunk.chunk_id} in {document.source_file} lost its table header",
                    )

    def test_every_chunk_carries_required_metadata(self):
        for strategy in available_strategies():
            for document in self.documents:
                for chunk in get_chunker(strategy, 800, 120).split(document):
                    for key in REQUIRED_METADATA:
                        self.assertTrue(
                            str(chunk.metadata.get(key, "")).strip(),
                            f"{chunk.chunk_id} is missing {key}",
                        )

    def test_chunk_ids_are_unique_per_strategy(self):
        for strategy in available_strategies():
            ids = [
                chunk.chunk_id
                for document in self.documents
                for chunk in get_chunker(strategy, 800, 120).split(document)
            ]
            self.assertEqual(len(ids), len(set(ids)))

    def test_offsets_point_into_the_source_document(self):
        for document in self.documents:
            for chunk in get_chunker("baseline", 800, 120).split(document):
                self.assertLessEqual(chunk.char_end, len(document.text) + 1)
                self.assertLess(chunk.char_start, chunk.char_end)


if __name__ == "__main__":
    unittest.main()
