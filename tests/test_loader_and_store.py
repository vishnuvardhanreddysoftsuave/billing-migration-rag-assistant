"""Metadata validation on ingest, and vector-store round trips and filtering."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ragchat.embeddings import HashingTfidfEmbedder
from ragchat.loader import IngestError, load_document, load_documents
from ragchat.models import Chunk
from ragchat.store import IngestEvent, StoreError, VectorStore, namespace_for

GOOD = """---
article_id: HC-9001
product_area: payments
last_updated: 2026-02-03
title: Good article
---

# Good article

Some prose about ERR-9001.
"""

MISSING_AREA = """---
article_id: HC-9002
last_updated: 2026-02-03
---

# No product area

Body.
"""


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name: str, text: str) -> Path:
        path = self.tmpdir / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_loads_metadata_and_normalises_dates(self):
        document = load_document(self._write("good.md", GOOD))
        self.assertEqual(document.article_id, "HC-9001")
        self.assertEqual(document.metadata["product_area"], "payments")
        # PyYAML parses an unquoted date into a date object; it must be JSON-safe.
        self.assertIsInstance(document.metadata["last_updated"], str)
        self.assertEqual(document.metadata["last_updated"], "2026-02-03")
        self.assertEqual(document.metadata["source_file"], "good.md")

    def test_missing_required_front_matter_is_a_failed_ingest(self):
        path = self._write("bad.md", MISSING_AREA)
        with self.assertRaises(IngestError) as ctx:
            load_document(path)
        self.assertIn("product_area", str(ctx.exception))

    def test_duplicate_article_ids_are_rejected(self):
        self._write("a.md", GOOD)
        self._write("b.md", GOOD)
        with self.assertRaises(IngestError):
            load_documents([self.tmpdir])

    def test_missing_path_is_reported(self):
        with self.assertRaises(IngestError):
            load_documents([self.tmpdir / "nope"])


def _chunk(chunk_id: str, text: str, **metadata) -> Chunk:
    meta = {
        "article_id": "HC-1",
        "source_file": "hc-1.md",
        "product_area": "payments",
        "last_updated": "2026-01-01",
    }
    meta.update(metadata)
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        strategy="test",
        position=0,
        char_start=0,
        char_end=len(text),
        section="Section",
        metadata=meta,
    )


class StoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        self.embedder = HashingTfidfEmbedder(n_features=4096)

    def tearDown(self):
        self._tmp.cleanup()

    def _event(self, label="test"):
        return IngestEvent(at="now", label=label, source_files=["hc-1.md"], n_documents=1, n_chunks=1, seconds=0.0)

    def _store_with(self, chunks):
        store = VectorStore("ns", self.embedder.spec)
        texts = [c.text for c in chunks]
        store.add(
            chunks=chunks,
            vectors=self.embedder.embed_documents(texts),
            df_delta=self.embedder.document_frequencies(texts),
            event=self._event(),
        )
        return store

    def test_round_trip_through_disk(self):
        store = self._store_with([_chunk("a", "webhook signature mismatch"), _chunk("b", "vat id validation")])
        store.save(self.tmpdir)

        reloaded = VectorStore.load(self.tmpdir, "ns")
        self.assertEqual(len(reloaded.chunks), 2)
        self.assertEqual(reloaded.n_units, 2)
        self.assertIsNotNone(reloaded.get("a"))
        np.testing.assert_allclose(reloaded.df, store.df)

    def test_chunk_missing_required_metadata_is_rejected(self):
        broken = _chunk("x", "text")
        broken.metadata.pop("source_file")
        with self.assertRaises(StoreError) as ctx:
            self._store_with([broken])
        self.assertIn("source_file", str(ctx.exception))

    def test_duplicate_chunk_ids_are_rejected(self):
        with self.assertRaises(StoreError):
            self._store_with([_chunk("dup", "one"), _chunk("dup", "two")])

    def test_metadata_filter_restricts_candidates(self):
        store = self._store_with(
            [
                _chunk("pay", "retry window after a failed charge", product_area="payments"),
                _chunk("api", "retry window for webhook delivery", product_area="developer-api"),
            ]
        )
        query = self.embedder.embed_query("retry window", store.df, store.n_units)

        unfiltered = store.search(query, top_k=5)
        filtered = store.search(query, top_k=5, filters={"product_area": "developer-api"})

        self.assertEqual(len(unfiltered), 2)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(store.chunks[filtered[0][0]].chunk_id, "api")

    def test_filter_matching_is_case_insensitive(self):
        store = self._store_with([_chunk("pay", "dunning retry", product_area="Payments")])
        query = self.embedder.embed_query("dunning retry", store.df, store.n_units)
        self.assertEqual(len(store.search(query, top_k=5, filters={"product_area": "payments"})), 1)

    def test_append_does_not_disturb_existing_vectors(self):
        store = self._store_with([_chunk("a", "first chunk about invoices")])
        store.save(self.tmpdir)
        before = store.matrix.toarray().copy()

        reloaded = VectorStore.load(self.tmpdir, "ns")
        texts = ["second chunk about webhooks"]
        reloaded.add(
            chunks=[_chunk("b", texts[0])],
            vectors=self.embedder.embed_documents(texts),
            df_delta=self.embedder.document_frequencies(texts),
            event=self._event("append"),
        )
        np.testing.assert_allclose(reloaded.matrix.toarray()[:1], before)
        self.assertEqual(len(reloaded.history), 2)

    def test_namespace_encodes_strategy_and_size(self):
        self.assertEqual(namespace_for("baseline", 800, 120), "baseline__cs800_ov120")


if __name__ == "__main__":
    unittest.main()
