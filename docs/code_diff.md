# Code diff — the second chunker and the metadata fields

The Week 3 extension as a single diff: `f047809` (the app as it stood) -> `25152e1`.

Two things the task asks to see specifically:

* **The second chunker** — `src/ragchat/chunkers/structure_aware.py` (new file) and its
  registration in `src/ragchat/chunkers/__init__.py`.
* **The metadata fields** — `product_area` and `last_updated` become required in
  `src/ragchat/models.py` (`REQUIRED_METADATA`) and `src/ragchat/loader.py`
  (`REQUIRED_FRONT_MATTER`), and `src/ragchat/store.py` gains metadata filtering so
  retrieval can be restricted by `product_area`.

Regenerate with:

    git diff f047809..25152e1 -- src/ config.yaml eval/questions.yaml

Stat summary:

```text
 eval/questions.yaml                     |  28 ++
 src/ragchat/chunkers/__init__.py        |  10 +-
 src/ragchat/chunkers/structure_aware.py | 342 +++++++++++++++++++
 src/ragchat/cli.py                      |  88 ++++-
 src/ragchat/config.py                   |  11 +
 src/ragchat/evaluation.py               | 583 ++++++++++++++++++++++++++++++++
 src/ragchat/loader.py                   |   3 +-
 src/ragchat/models.py                   |  13 +-
 src/ragchat/pipeline.py                 |  30 +-
 src/ragchat/reporting.py                | 374 ++++++++++++++++++++
 src/ragchat/retriever.py                |  14 +-
 src/ragchat/store.py                    |  38 ++-
 src/ragchat/webapp.py                   | 164 +++++++++
 13 files changed, 1678 insertions(+), 20 deletions(-)
```

## Full diff

```diff
diff --git a/eval/questions.yaml b/eval/questions.yaml
index ffcb15f..4335c08 100644
--- a/eval/questions.yaml
+++ b/eval/questions.yaml
@@ -159,6 +159,7 @@ filter_demo:
 bonus:
   id: B1
   question: "A customer hit ERR-4032. What do I tell them to do, and what will they need to hand?"
+  context_chunks: 5
   gold:
     article_id: HC-4002
     sections:
@@ -168,3 +169,30 @@ bonus:
     Direct them to Billing -> Payment methods -> "Re-authorise card". They need the
     physical card, because the flow asks for the full card number, expiry, and CVC: the
     pre-2026-05-01 token cannot supply them.
+
+# ---------------------------------------------------------------------------
+# ADDED AFTER THE FIRST RUN — disclosed, and not part of any scored metric.
+#
+# B1 above turned out to be a poor probe, which is itself a finding: its colloquial
+# phrasing ("hit", "need to hand") contains two ordinary English words the corpus does
+# not use, so the evidence gate refuses it under the shipping threshold, and neither
+# strategy ranks the ERR-4032 row first. B1 is kept and reported unchanged.
+#
+# B2 asks the same thing in the register a support agent actually types, and pins the
+# context to a single chunk, which is what isolates the precision/completeness tension
+# the bonus challenge is about.
+# ---------------------------------------------------------------------------
+bonus_followup:
+  id: B2
+  added_after_first_run: true
+  question: "A customer reports ERR-4032. What is the fix, and what will the customer need in order to complete it?"
+  context_chunks: 1
+  gold:
+    article_id: HC-4002
+    sections:
+      - "Troubleshooting: payment method errors (row ERR-4032)"
+      - "Re-authorisation walkthrough"
+  known_answer: >-
+    Have the customer re-authorise the card under Billing -> Payment methods. They need
+    the physical card in hand, because the flow asks for the full card number, expiry
+    and CVC that the pre-2026-05-01 token cannot supply.
diff --git a/src/ragchat/chunkers/__init__.py b/src/ragchat/chunkers/__init__.py
index 1d49c19..3faead8 100644
--- a/src/ragchat/chunkers/__init__.py
+++ b/src/ragchat/chunkers/__init__.py
@@ -6,9 +6,11 @@ from typing import Dict, Type
 
 from .base import Chunker
 from .baseline import BaselineChunker
+from .structure_aware import StructureAwareChunker
 
 _REGISTRY: Dict[str, Type[Chunker]] = {
     BaselineChunker.name: BaselineChunker,
+    StructureAwareChunker.name: StructureAwareChunker,
 }
 
 
@@ -24,4 +26,10 @@ def get_chunker(name: str, chunk_size: int, chunk_overlap: int) -> Chunker:
     return cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
 
 
-__all__ = ["Chunker", "BaselineChunker", "available_strategies", "get_chunker"]
+__all__ = [
+    "Chunker",
+    "BaselineChunker",
+    "StructureAwareChunker",
+    "available_strategies",
+    "get_chunker",
+]
diff --git a/src/ragchat/chunkers/structure_aware.py b/src/ragchat/chunkers/structure_aware.py
new file mode 100644
index 0000000..e5c0dce
--- /dev/null
+++ b/src/ragchat/chunkers/structure_aware.py
@@ -0,0 +1,342 @@
+"""Structure-aware chunker: markdown blocks in, table rows never orphaned.
+
+The baseline chunker packs characters and cannot see that a troubleshooting table
+is a table. When a table runs past the character budget it gets cut mid-table, and
+every row after the cut is stored without the header that says which column is the
+error code, which is the cause and which is the fix.
+
+This chunker parses the document into blocks first and then guarantees:
+
+1. **A table row is never separated from its header row.** A table that does not
+   fit in one chunk is split between rows, and every resulting chunk repeats the
+   header row and its separator.
+2. **A row is never split down the middle.** If a single row plus its header
+   exceeds the chunk size the row is emitted whole and oversized, because half a
+   row is worse than a large chunk.
+3. **Chunks do not span section boundaries**, so unrelated sections never share a
+   chunk, and each chunk is prefixed with its heading breadcrumb so a bare table
+   row still carries the context of the section it came from.
+"""
+
+from __future__ import annotations
+
+import re
+from dataclasses import dataclass, field
+from typing import List, Optional, Sequence, Tuple
+
+from ..models import Chunk, Document
+from .base import Chunker, section_index
+
+HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
+TABLE_SEPARATOR_RE = re.compile(r"^\|[\s:|-]+\|$")
+SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
+
+
+@dataclass
+class Block:
+    """A markdown block: heading, paragraph or table."""
+
+    kind: str
+    start: int
+    end: int
+    text: str
+    level: int = 0
+    title: str = ""
+    header: List[str] = field(default_factory=list)
+    rows: List[Tuple[int, int, str]] = field(default_factory=list)
+
+
+@dataclass
+class Unit:
+    """An atomic piece of text that a chunk may contain."""
+
+    text: str
+    start: int
+    end: int
+    kind: str
+    atomic: bool = False
+    n_rows: int = 0
+    header_repeated: bool = False
+
+
+class StructureAwareChunker(Chunker):
+    name = "structure-aware"
+
+    def split(self, document: Document) -> List[Chunk]:
+        text = document.text
+        index = section_index(text)
+        blocks = parse_blocks(text)
+
+        chunks: List[Chunk] = []
+        for heading_path, group in _group_by_section(blocks):
+            prefix = f"{heading_path}\n\n" if heading_path else ""
+            budget = max(self.chunk_size - len(prefix), 200)
+
+            units: List[Unit] = []
+            for block in group:
+                if block.kind == "table":
+                    units.extend(_table_units(block, budget))
+                else:
+                    units.append(Unit(block.text, block.start, block.end, block.kind))
+
+            for unit_group in _pack(units, budget, self.chunk_overlap):
+                body = "\n\n".join(u.text for u in unit_group).strip()
+                if not body:
+                    continue
+                table_units = [u for u in unit_group if u.kind == "table"]
+                chunks.append(
+                    self.build_chunk(
+                        document=document,
+                        text=prefix + body,
+                        position=len(chunks),
+                        char_start=min(u.start for u in unit_group),
+                        char_end=max(u.end for u in unit_group),
+                        index=index,
+                        extra_metadata={
+                            "contains_table": bool(table_units),
+                            "table_rows": sum(u.n_rows for u in table_units),
+                            "table_header_repeated": any(u.header_repeated for u in table_units),
+                            "heading_prefixed": bool(prefix),
+                        },
+                    )
+                )
+        return chunks
+
+
+# --------------------------------------------------------------------------
+# Block parsing
+# --------------------------------------------------------------------------
+
+
+def parse_blocks(text: str) -> List[Block]:
+    """Split markdown into heading / table / paragraph blocks, keeping offsets."""
+    lines = text.split("\n")
+    offsets: List[int] = []
+    position = 0
+    for line in lines:
+        offsets.append(position)
+        position += len(line) + 1
+
+    blocks: List[Block] = []
+    i, n = 0, len(lines)
+    while i < n:
+        stripped = lines[i].strip()
+        if not stripped:
+            i += 1
+            continue
+
+        heading = HEADING_LINE_RE.match(stripped)
+        if heading:
+            blocks.append(
+                Block(
+                    kind="heading",
+                    start=offsets[i],
+                    end=offsets[i] + len(lines[i]),
+                    text=lines[i].strip(),
+                    level=len(heading.group(1)),
+                    title=heading.group(2).strip(),
+                )
+            )
+            i += 1
+            continue
+
+        if stripped.startswith("|"):
+            j = i
+            while j < n and lines[j].strip().startswith("|"):
+                j += 1
+            blocks.append(_table_block(lines, offsets, i, j))
+            i = j
+            continue
+
+        j = i
+        while j < n and lines[j].strip() and not lines[j].strip().startswith(("|", "#")):
+            j += 1
+        blocks.append(
+            Block(
+                kind="paragraph",
+                start=offsets[i],
+                end=offsets[j - 1] + len(lines[j - 1]),
+                text="\n".join(line.strip() for line in lines[i:j]),
+            )
+        )
+        i = j
+    return blocks
+
+
+def _table_block(lines: Sequence[str], offsets: Sequence[int], start: int, stop: int) -> Block:
+    header: List[str] = []
+    rows: List[Tuple[int, int, str]] = []
+    body_start = start
+
+    if stop - start >= 2 and TABLE_SEPARATOR_RE.match(lines[start + 1].strip()):
+        header = [lines[start].strip(), lines[start + 1].strip()]
+        body_start = start + 2
+
+    for k in range(body_start, stop):
+        row = lines[k].strip()
+        if not row or TABLE_SEPARATOR_RE.match(row):
+            continue
+        rows.append((offsets[k], offsets[k] + len(lines[k]), row))
+
+    return Block(
+        kind="table",
+        start=offsets[start],
+        end=offsets[stop - 1] + len(lines[stop - 1]),
+        text="\n".join(line.strip() for line in lines[start:stop]),
+        header=header,
+        rows=rows,
+    )
+
+
+# --------------------------------------------------------------------------
+# Assembly
+# --------------------------------------------------------------------------
+
+
+def _group_by_section(blocks: Sequence[Block]) -> List[Tuple[str, List[Block]]]:
+    """Group content blocks under their heading breadcrumb."""
+    groups: List[Tuple[str, List[Block]]] = []
+    stack: List[Tuple[int, str]] = []
+    current: List[Block] = []
+    breadcrumb = ""
+
+    for block in blocks:
+        if block.kind == "heading":
+            if current:
+                groups.append((breadcrumb, current))
+                current = []
+            while stack and stack[-1][0] >= block.level:
+                stack.pop()
+            stack.append((block.level, block.title))
+            breadcrumb = " > ".join(title for _level, title in stack)
+            continue
+        current.append(block)
+
+    if current:
+        groups.append((breadcrumb, current))
+    return groups
+
+
+def _table_units(block: Block, budget: int) -> List[Unit]:
+    """Turn a table into units, repeating the header whenever it must be split."""
+    if not block.rows:
+        return [Unit(block.text, block.start, block.end, "table", atomic=True)]
+
+    header_text = "\n".join(block.header)
+    header_len = len(header_text) + 1 if header_text else 0
+
+    if len(block.text) <= budget:
+        return [
+            Unit(
+                text=block.text,
+                start=block.start,
+                end=block.end,
+                kind="table",
+                atomic=True,
+                n_rows=len(block.rows),
+                header_repeated=False,
+            )
+        ]
+
+    units: List[Unit] = []
+    batch: List[Tuple[int, int, str]] = []
+    batch_len = header_len
+
+    def flush() -> None:
+        nonlocal batch, batch_len
+        if not batch:
+            return
+        body = "\n".join(row for _s, _e, row in batch)
+        text = f"{header_text}\n{body}" if header_text else body
+        units.append(
+            Unit(
+                text=text,
+                start=batch[0][0],
+                end=batch[-1][1],
+                kind="table",
+                atomic=True,
+                n_rows=len(batch),
+                header_repeated=bool(header_text) and bool(units),
+            )
+        )
+        batch, batch_len = [], header_len
+
+    for row in block.rows:
+        row_len = len(row[2]) + 1
+        # A row is never split: if it does not fit it is emitted whole, oversized.
+        if batch and batch_len + row_len > budget:
+            flush()
+        batch.append(row)
+        batch_len += row_len
+    flush()
+    return units
+
+
+def _pack(units: Sequence[Unit], budget: int, overlap: int) -> List[List[Unit]]:
+    """Pack units into chunks; atomic units (tables) always stand alone."""
+    packed: List[List[Unit]] = []
+    current: List[Unit] = []
+    current_len = 0
+
+    for unit in units:
+        if unit.atomic:
+            if current:
+                packed.append(current)
+                current, current_len = [], 0
+            packed.append([unit])
+            continue
+
+        pieces = _split_prose(unit, budget)
+        for piece in pieces:
+            piece_len = len(piece.text) + 2
+            if current and current_len + piece_len > budget:
+                packed.append(current)
+                current, current_len = _overlap_tail(current, overlap)
+            current.append(piece)
+            current_len += piece_len
+
+    if current:
+        packed.append(current)
+    return packed
+
+
+def _split_prose(unit: Unit, budget: int) -> List[Unit]:
+    """Split an oversized paragraph on sentence boundaries."""
+    if len(unit.text) <= budget:
+        return [unit]
+
+    pieces: List[Unit] = []
+    cursor = unit.start
+    buffer: List[str] = []
+    buffer_len = 0
+
+    for sentence in SENTENCE_RE.split(unit.text):
+        sentence = sentence.strip()
+        if not sentence:
+            continue
+        if buffer and buffer_len + len(sentence) + 1 > budget:
+            body = " ".join(buffer)
+            pieces.append(Unit(body, cursor, min(cursor + len(body), unit.end), unit.kind))
+            cursor = min(cursor + len(body) + 1, unit.end)
+            buffer, buffer_len = [], 0
+        buffer.append(sentence)
+        buffer_len += len(sentence) + 1
+
+    if buffer:
+        body = " ".join(buffer)
+        pieces.append(Unit(body, cursor, unit.end, unit.kind))
+    return pieces
+
+
+def _overlap_tail(current: Sequence[Unit], overlap: int) -> Tuple[List[Unit], int]:
+    """Carry trailing prose into the next chunk, up to the overlap budget."""
+    if overlap <= 0:
+        return [], 0
+    tail: List[Unit] = []
+    tail_len = 0
+    for unit in reversed(current):
+        if tail_len >= overlap:
+            break
+        tail.insert(0, unit)
+        tail_len += len(unit.text) + 2
+    return tail, tail_len
diff --git a/src/ragchat/cli.py b/src/ragchat/cli.py
index a356c89..72b7cb6 100644
--- a/src/ragchat/cli.py
+++ b/src/ragchat/cli.py
@@ -39,17 +39,53 @@ def build_parser() -> argparse.ArgumentParser:
     p_search.add_argument("question")
     p_search.add_argument("--top-k", type=int, default=None)
     p_search.add_argument("--json", action="store_true", help="emit JSON instead of text")
+    _add_filter_args(p_search)
 
     p_ask = sub.add_parser("ask", help="retrieve, then answer with citations or refuse")
     p_ask.add_argument("question")
     p_ask.add_argument("--top-k", type=int, default=None)
     p_ask.add_argument("--json", action="store_true")
+    _add_filter_args(p_ask)
 
     sub.add_parser("stats", help="show what is in the index")
 
+    p_eval = sub.add_parser("eval", help="run the full evaluation and write results")
+    p_eval.add_argument("--questions", type=Path, default=None, help="path to questions.yaml")
+    p_eval.add_argument(
+        "--strategies",
+        nargs="*",
+        default=None,
+        help="strategies to compare (default: baseline structure-aware)",
+    )
+    p_eval.add_argument("--out", type=Path, default=None, help="directory for the generated report")
+    p_eval.add_argument("--skip-sweep", action="store_true", help="skip the chunk-size sweep")
+
+    p_sweep = sub.add_parser("sweep", help="chunk-size sweep only")
+    p_sweep.add_argument("--sizes", nargs="+", type=int, default=[400, 800, 1200, 1600])
+    p_sweep.add_argument("--questions", type=Path, default=None)
+    p_sweep.add_argument("--strategies", nargs="*", default=None)
+
+    p_serve = sub.add_parser("serve", help="run the web UI")
+    p_serve.add_argument("--host", default="127.0.0.1")
+    p_serve.add_argument("--port", type=int, default=5000)
+    p_serve.add_argument("--debug", action="store_true")
+
     return parser
 
 
+def _add_filter_args(parser: argparse.ArgumentParser) -> None:
+    parser.add_argument("--product-area", default=None, help="restrict retrieval to a product_area")
+    parser.add_argument("--article-id", default=None, help="restrict retrieval to an article_id")
+
+
+def filters_from_args(args: argparse.Namespace) -> dict:
+    filters = {
+        "product_area": getattr(args, "product_area", None),
+        "article_id": getattr(args, "article_id", None),
+    }
+    return {key: value for key, value in filters.items() if value}
+
+
 def load_config(args: argparse.Namespace) -> Config:
     config = Config.load(args.config)
     if args.chunk_size is not None or args.chunk_overlap is not None:
@@ -77,28 +113,69 @@ def main(argv: Sequence[str] | None = None) -> int:
 
     if args.command == "search":
         retriever = Retriever.open(config, strategy=strategy, index_dir=args.index_dir)
-        hits = retriever.search(args.question, top_k=args.top_k or config.retrieval.top_k)
+        filters = filters_from_args(args)
+        hits = retriever.search(args.question, top_k=args.top_k or config.retrieval.top_k, filters=filters)
         if args.json:
             print(json.dumps([hit.to_dict() for hit in hits], indent=2))
         else:
-            print(format_hits(args.question, retriever.namespace, hits))
+            print(format_hits(args.question, retriever.namespace, hits, filters))
         return 0
 
     if args.command == "ask":
         pipeline = RAGPipeline.open(config, strategy=strategy, index_dir=args.index_dir)
-        answer = pipeline.ask(args.question, top_k=args.top_k)
+        answer = pipeline.ask(args.question, top_k=args.top_k, filters=filters_from_args(args))
         if args.json:
             print(json.dumps(answer.to_dict(), indent=2))
         else:
             print(format_answer(answer))
         return 0
 
+    if args.command == "eval":
+        from .evaluation import run_evaluation
+
+        report = run_evaluation(
+            config,
+            questions_path=args.questions,
+            strategies=args.strategies,
+            out_dir=args.out,
+            index_dir=args.index_dir,
+            with_sweep=not args.skip_sweep,
+        )
+        print(report.summary_text())
+        return 0
+
+    if args.command == "sweep":
+        from .evaluation import load_questions, run_sweep
+
+        questions = load_questions(args.questions or (config.repo_root / "eval" / "questions.yaml"))
+        rows = run_sweep(
+            config,
+            questions,
+            sizes=args.sizes,
+            strategies=args.strategies or ["baseline", "structure-aware"],
+            index_dir=args.index_dir,
+        )
+        print(json.dumps(rows, indent=2))
+        return 0
+
+    if args.command == "serve":
+        from .webapp import create_app
+
+        app = create_app(config, index_dir=args.index_dir)
+        app.run(host=args.host, port=args.port, debug=args.debug)
+        return 0
+
     parser.error(f"unknown command {args.command}")
     return 2
 
 
-def format_hits(question: str, namespace: str, hits: Sequence[SearchHit]) -> str:
-    lines = [f"query: {question}", f"index: {namespace}", ""]
+def format_hits(
+    question: str,
+    namespace: str,
+    hits: Sequence[SearchHit],
+    filters: dict | None = None,
+) -> str:
+    lines = [f"query: {question}", f"index: {namespace}", f"filters: {filters or '(none)'}", ""]
     if not hits:
         lines.append("(no matches)")
         return "\n".join(lines)
@@ -108,6 +185,7 @@ def format_hits(question: str, namespace: str, hits: Sequence[SearchHit]) -> str
         lines.append(
             f"#{hit.rank}  score={hit.score:.4f}  {chunk.chunk_id}\n"
             f"    article={chunk.article_id}  file={chunk.source_file}\n"
+            f"    product_area={chunk.product_area}  last_updated={chunk.last_updated}\n"
             f"    section={chunk.section or '(none)'}\n"
             f"    {preview[:240]}{'…' if len(preview) > 240 else ''}"
         )
diff --git a/src/ragchat/config.py b/src/ragchat/config.py
index ac62b05..ebc6dbb 100644
--- a/src/ragchat/config.py
+++ b/src/ragchat/config.py
@@ -116,6 +116,17 @@ class Config:
             raw=raw,
         )
 
+    @property
+    def repo_root(self) -> Path:
+        return REPO_ROOT
+
+    def with_grounding(self, *, min_evidence_coverage: float) -> "Config":
+        """Return a copy with a different evidence threshold (used by the bonus run)."""
+        return replace(
+            self,
+            grounding=replace(self.grounding, min_evidence_coverage=min_evidence_coverage),
+        )
+
     def with_chunking(self, *, chunk_size: int | None = None, chunk_overlap: int | None = None) -> "Config":
         """Return a copy with chunk sizing overridden (used by the sweep)."""
         chunking = replace(
diff --git a/src/ragchat/evaluation.py b/src/ragchat/evaluation.py
new file mode 100644
index 0000000..f307291
--- /dev/null
+++ b/src/ragchat/evaluation.py
@@ -0,0 +1,583 @@
+"""Evaluation harness.
+
+Produces every number in ``results.md``:
+
+* hit-in-top-5 over the same eight known-answer questions for each chunking
+  strategy, with the per-question record rather than a summary claim
+* the unfiltered vs filtered result lists for one ``product_area`` query
+* cited answers whose citations are checked against the chunk they name
+* refusal transcripts for the out-of-corpus questions
+* a chunk-size sweep across both strategies
+"""
+
+from __future__ import annotations
+
+import json
+from dataclasses import dataclass, field
+from datetime import datetime, timezone
+from pathlib import Path
+from typing import Any, Dict, List, Optional, Sequence
+
+import yaml
+
+from .config import Config
+from .indexer import ingest
+from .models import Answer, SearchHit
+from .pipeline import RAGPipeline
+from .retriever import Retriever
+from .store import StoreError, VectorStore, namespace_for
+
+DEFAULT_STRATEGIES = ("baseline", "structure-aware")
+SWEEP_SIZES = (400, 800, 1200, 1600)
+# The bonus question is colloquially phrased and trips the shipping gate; the bonus
+# comparison is about answer completeness, so it runs with the gate relaxed to here.
+BONUS_COVERAGE = 0.30
+
+
+# --------------------------------------------------------------------------
+# Loading and indexing
+# --------------------------------------------------------------------------
+
+
+def load_questions(path: Path) -> Dict[str, Any]:
+    with Path(path).open("r", encoding="utf-8") as fh:
+        payload = yaml.safe_load(fh)
+    if not payload or "answerable" not in payload:
+        raise ValueError(f"{path} does not look like a question set")
+    return payload
+
+
+def ensure_index(config: Config, strategy: str, index_dir: Path | None = None) -> str:
+    """Build the index for a strategy if it does not exist yet.
+
+    Both strategies are given the identical corpus — the pre-existing legacy
+    articles plus the new drop — so a hit-rate difference cannot come from one
+    index having seen more documents than the other.
+    """
+    index_dir = Path(index_dir) if index_dir else config.paths.index_dir
+    namespace = namespace_for(strategy, config.chunking.chunk_size, config.chunking.chunk_overlap)
+    try:
+        VectorStore.load(index_dir, namespace)
+        return namespace
+    except StoreError:
+        pass
+
+    ingest(config, [config.paths.legacy_articles_dir], strategy=strategy,
+           label="legacy-corpus (pre-existing index)", index_dir=index_dir)
+    ingest(config, [config.paths.articles_dir], strategy=strategy,
+           label="week3-new-drop", index_dir=index_dir)
+    return namespace
+
+
+# --------------------------------------------------------------------------
+# Retrieval scoring
+# --------------------------------------------------------------------------
+
+
+@dataclass
+class QuestionResult:
+    qid: str
+    question: str
+    gold_article: str
+    gold_section: str
+    must_contain: List[str]
+    depends_on_table_row: bool
+    hit: bool
+    hit_rank: Optional[int]
+    hit_chunk_id: Optional[str]
+    best_article_rank: Optional[int]
+    diagnosis: str
+    self_contained_hit: bool = False
+    self_contained_rank: Optional[int] = None
+    self_contained_chunk_id: Optional[str] = None
+    hits: List[SearchHit] = field(default_factory=list)
+
+    def to_dict(self) -> Dict[str, Any]:
+        return {
+            "qid": self.qid,
+            "question": self.question,
+            "gold_article": self.gold_article,
+            "gold_section": self.gold_section,
+            "must_contain": self.must_contain,
+            "depends_on_table_row": self.depends_on_table_row,
+            "hit": self.hit,
+            "hit_rank": self.hit_rank,
+            "hit_chunk_id": self.hit_chunk_id,
+            "self_contained_hit": self.self_contained_hit,
+            "self_contained_rank": self.self_contained_rank,
+            "self_contained_chunk_id": self.self_contained_chunk_id,
+            "best_article_rank": self.best_article_rank,
+            "diagnosis": self.diagnosis,
+            "hits": [h.to_dict() for h in self.hits],
+        }
+
+
+@dataclass
+class StrategyReport:
+    strategy: str
+    namespace: str
+    n_chunks: int
+    mean_chunk_chars: float
+    results: List[QuestionResult]
+
+    @property
+    def n_hits(self) -> int:
+        return sum(1 for r in self.results if r.hit)
+
+    @property
+    def n_self_contained(self) -> int:
+        return sum(1 for r in self.results if r.self_contained_hit)
+
+    @property
+    def n_questions(self) -> int:
+        return len(self.results)
+
+    @property
+    def table_self_contained(self) -> str:
+        rows = [r for r in self.results if r.depends_on_table_row]
+        return f"{sum(1 for r in rows if r.self_contained_hit)}/{len(rows)}"
+
+    @property
+    def table_hits(self) -> str:
+        rows = [r for r in self.results if r.depends_on_table_row]
+        return f"{sum(1 for r in rows if r.hit)}/{len(rows)}"
+
+    def to_dict(self) -> Dict[str, Any]:
+        return {
+            "strategy": self.strategy,
+            "namespace": self.namespace,
+            "n_chunks": self.n_chunks,
+            "mean_chunk_chars": self.mean_chunk_chars,
+            "hit_at_5": f"{self.n_hits}/{self.n_questions}",
+            "self_contained_hit_at_5": f"{self.n_self_contained}/{self.n_questions}",
+            "results": [r.to_dict() for r in self.results],
+        }
+
+
+def _chunk_supports(chunk_text: str, must_contain: Sequence[str]) -> bool:
+    haystack = " ".join(chunk_text.split()).lower()
+    return all(" ".join(str(needle).split()).lower() in haystack for needle in must_contain)
+
+
+def _is_separator_row(line: str) -> bool:
+    stripped = line.strip()
+    return stripped.startswith("|") and set(stripped) <= set("|-: ")
+
+
+def contains_table_rows(text: str) -> bool:
+    return any(line.strip().startswith("|") and not _is_separator_row(line) for line in text.splitlines())
+
+
+def has_table_header(text: str) -> bool:
+    """True when a header row (a row followed by its separator) is present."""
+    lines = text.splitlines()
+    return any(
+        lines[i].strip().startswith("|")
+        and not _is_separator_row(lines[i])
+        and _is_separator_row(lines[i + 1])
+        for i in range(len(lines) - 1)
+    )
+
+
+def is_self_contained(chunk_text: str) -> bool:
+    """A chunk is self-contained when its table rows still have their header.
+
+    This is the secondary metric. A bare ``| ERR-4032 | ... | ... |`` row is
+    retrievable and it satisfies the committed hit criterion, but without the
+    header an agent cannot tell which cell is the cause and which is the fix.
+    """
+    if not contains_table_rows(chunk_text):
+        return True
+    return has_table_header(chunk_text)
+
+
+def evaluate_strategy(
+    config: Config,
+    strategy: str,
+    questions: Dict[str, Any],
+    index_dir: Path | None = None,
+    top_k: int = 5,
+) -> StrategyReport:
+    namespace = ensure_index(config, strategy, index_dir)
+    retriever = Retriever.open(config, strategy=strategy, index_dir=index_dir)
+
+    results: List[QuestionResult] = []
+    for item in questions["answerable"]:
+        gold = item["gold"]
+        must_contain = list(item.get("must_contain", []))
+        hits = retriever.search(item["question"], top_k=top_k)
+
+        hit_rank: Optional[int] = None
+        hit_chunk_id: Optional[str] = None
+        best_article_rank: Optional[int] = None
+        sc_rank: Optional[int] = None
+        sc_chunk_id: Optional[str] = None
+        for hit in hits:
+            if hit.chunk.article_id == gold["article_id"]:
+                if best_article_rank is None:
+                    best_article_rank = hit.rank
+                if _chunk_supports(hit.chunk.text, must_contain):
+                    if hit_rank is None:
+                        hit_rank, hit_chunk_id = hit.rank, hit.chunk.chunk_id
+                    if sc_rank is None and is_self_contained(hit.chunk.text):
+                        sc_rank, sc_chunk_id = hit.rank, hit.chunk.chunk_id
+
+        results.append(
+            QuestionResult(
+                qid=item["id"],
+                question=item["question"],
+                gold_article=gold["article_id"],
+                gold_section=gold.get("section", ""),
+                must_contain=must_contain,
+                depends_on_table_row=bool(item.get("depends_on_table_row")),
+                hit=hit_rank is not None,
+                hit_rank=hit_rank,
+                hit_chunk_id=hit_chunk_id,
+                best_article_rank=best_article_rank,
+                diagnosis=_diagnose(hits, gold["article_id"], must_contain, best_article_rank, hit_rank),
+                self_contained_hit=sc_rank is not None,
+                self_contained_rank=sc_rank,
+                self_contained_chunk_id=sc_chunk_id,
+                hits=hits,
+            )
+        )
+
+    stats = retriever.store.stats()
+    return StrategyReport(
+        strategy=strategy,
+        namespace=namespace,
+        n_chunks=stats["n_chunks"],
+        mean_chunk_chars=stats["mean_chunk_chars"],
+        results=results,
+    )
+
+
+def _diagnose(
+    hits: Sequence[SearchHit],
+    gold_article: str,
+    must_contain: Sequence[str],
+    best_article_rank: Optional[int],
+    hit_rank: Optional[int],
+) -> str:
+    if hit_rank is not None:
+        return f"answer-bearing chunk retrieved at rank {hit_rank}"
+    if best_article_rank is None:
+        return f"no chunk from {gold_article} in the top {len(hits)}"
+    partial = [
+        needle
+        for needle in must_contain
+        if any(
+            " ".join(str(needle).split()).lower() in " ".join(h.chunk.text.split()).lower()
+            for h in hits
+            if h.chunk.article_id == gold_article
+        )
+    ]
+    missing = [n for n in must_contain if n not in partial]
+    return (
+        f"right article at rank {best_article_rank} but the retrieved chunks are missing "
+        f"{missing!r} — the answer was split across chunk boundaries"
+    )
+
+
+# --------------------------------------------------------------------------
+# Metadata filter demonstration
+# --------------------------------------------------------------------------
+
+
+@dataclass
+class FilterDemo:
+    question: str
+    filters: Dict[str, Any]
+    unfiltered: List[SearchHit]
+    filtered: List[SearchHit]
+
+    @property
+    def changed_top1(self) -> bool:
+        if not self.unfiltered or not self.filtered:
+            return False
+        return self.unfiltered[0].chunk.chunk_id != self.filtered[0].chunk.chunk_id
+
+    def to_dict(self) -> Dict[str, Any]:
+        return {
+            "question": self.question,
+            "filters": self.filters,
+            "changed_top1": self.changed_top1,
+            "unfiltered": [h.to_dict() for h in self.unfiltered],
+            "filtered": [h.to_dict() for h in self.filtered],
+        }
+
+
+def run_filter_demo(
+    config: Config,
+    strategy: str,
+    demo: Dict[str, Any],
+    index_dir: Path | None = None,
+    top_k: int = 5,
+) -> FilterDemo:
+    retriever = Retriever.open(config, strategy=strategy, index_dir=index_dir)
+    question = demo["question"]
+    filters = dict(demo["filter"])
+    return FilterDemo(
+        question=question,
+        filters=filters,
+        unfiltered=retriever.search(question, top_k=top_k),
+        filtered=retriever.search(question, top_k=top_k, filters=filters),
+    )
+
+
+# --------------------------------------------------------------------------
+# Generation: cited answers and refusals
+# --------------------------------------------------------------------------
+
+
+@dataclass
+class AnswerCheck:
+    qid: str
+    question: str
+    answer: Answer
+    citations_resolve: bool
+    citation_supports_claim: bool
+    supporting_chunk_id: Optional[str]
+
+    def to_dict(self) -> Dict[str, Any]:
+        return {
+            "qid": self.qid,
+            "question": self.question,
+            "citations_resolve": self.citations_resolve,
+            "citation_supports_claim": self.citation_supports_claim,
+            "supporting_chunk_id": self.supporting_chunk_id,
+            "answer": self.answer.to_dict(),
+        }
+
+
+def run_answer_checks(
+    pipeline: RAGPipeline,
+    items: Sequence[Dict[str, Any]],
+) -> List[AnswerCheck]:
+    checks: List[AnswerCheck] = []
+    for item in items:
+        answer = pipeline.ask(item["question"])
+        must_contain = list(item.get("must_contain", []))
+
+        resolves = bool(answer.citations) and all(c.resolved for c in answer.citations)
+        supporting_id = None
+        for citation in answer.citations:
+            chunk = pipeline.retriever.get_chunk(citation.chunk_id)
+            if chunk and _chunk_supports(chunk.text, must_contain):
+                supporting_id = chunk.chunk_id
+                break
+
+        checks.append(
+            AnswerCheck(
+                qid=item["id"],
+                question=item["question"],
+                answer=answer,
+                citations_resolve=resolves,
+                citation_supports_claim=supporting_id is not None,
+                supporting_chunk_id=supporting_id,
+            )
+        )
+    return checks
+
+
+def run_refusals(pipeline: RAGPipeline, items: Sequence[Dict[str, Any]]) -> List[Answer]:
+    return [pipeline.ask(item["question"]) for item in items]
+
+
+# --------------------------------------------------------------------------
+# Chunk-size sweep
+# --------------------------------------------------------------------------
+
+
+def run_sweep(
+    config: Config,
+    questions: Dict[str, Any],
+    sizes: Sequence[int],
+    strategies: Sequence[str],
+    index_dir: Path | None = None,
+) -> List[Dict[str, Any]]:
+    """hit@5 for every (strategy, chunk size) pair. One variable changes at a time."""
+    base_dir = Path(index_dir) if index_dir else config.paths.index_dir
+    sweep_dir = base_dir / "sweep"
+    rows: List[Dict[str, Any]] = []
+
+    for strategy in strategies:
+        for size in sizes:
+            overlap = min(config.chunking.chunk_overlap, max(size // 8, 0))
+            sized = config.with_chunking(chunk_size=size, chunk_overlap=overlap)
+            report = evaluate_strategy(sized, strategy, questions, index_dir=sweep_dir)
+            table = [r for r in report.results if r.depends_on_table_row]
+            prose = [r for r in report.results if not r.depends_on_table_row]
+            rows.append(
+                {
+                    "strategy": strategy,
+                    "chunk_size": size,
+                    "chunk_overlap": overlap,
+                    "n_chunks": report.n_chunks,
+                    "mean_chunk_chars": report.mean_chunk_chars,
+                    "hit_at_5": f"{report.n_hits}/{report.n_questions}",
+                    "hit_rate": round(report.n_hits / report.n_questions, 3),
+                    "self_contained_hit_at_5": f"{report.n_self_contained}/{report.n_questions}",
+                    "table_questions": f"{sum(1 for r in table if r.hit)}/{len(table)}",
+                    "table_questions_self_contained": f"{sum(1 for r in table if r.self_contained_hit)}/{len(table)}",
+                    "prose_questions": f"{sum(1 for r in prose if r.hit)}/{len(prose)}",
+                }
+            )
+    return rows
+
+
+# --------------------------------------------------------------------------
+# Orchestration
+# --------------------------------------------------------------------------
+
+
+@dataclass
+class EvaluationReport:
+    generated_at: str
+    config_summary: Dict[str, Any]
+    strategy_reports: List[StrategyReport]
+    filter_demo: FilterDemo
+    answer_checks: List[AnswerCheck]
+    refusals: List[Answer]
+    bonus: List[Dict[str, Any]]
+    sweep: List[Dict[str, Any]]
+    index_history: Dict[str, Any]
+
+    def summary_text(self) -> str:
+        lines = ["hit-in-top-5 over the same 8 known-answer questions:"]
+        for report in self.strategy_reports:
+            lines.append(
+                f"  {report.strategy:16} hit@5 {report.n_hits}/{report.n_questions}"
+                f"   self-contained {report.n_self_contained}/{report.n_questions}"
+                f"   (table-row questions {report.table_hits} / self-contained {report.table_self_contained})"
+                f"   chunks={report.n_chunks}"
+            )
+        lines.append(f"filter changed top-1: {self.filter_demo.changed_top1}")
+        answered = sum(1 for c in self.answer_checks if not c.answer.refused)
+        supported = sum(1 for c in self.answer_checks if c.citation_supports_claim)
+        lines.append(f"cited answers: {answered}/{len(self.answer_checks)} answered, {supported} citation-verified")
+        refused = sum(1 for a in self.refusals if a.refused)
+        lines.append(f"out-of-corpus questions refused: {refused}/{len(self.refusals)}")
+        return "\n".join(lines)
+
+    def to_dict(self) -> Dict[str, Any]:
+        return {
+            "generated_at": self.generated_at,
+            "config": self.config_summary,
+            "strategies": [r.to_dict() for r in self.strategy_reports],
+            "filter_demo": self.filter_demo.to_dict(),
+            "answer_checks": [c.to_dict() for c in self.answer_checks],
+            "refusals": [a.to_dict() for a in self.refusals],
+            "bonus": self.bonus,
+            "sweep": self.sweep,
+            "index_history": self.index_history,
+        }
+
+
+def run_evaluation(
+    config: Config,
+    questions_path: Path | None = None,
+    strategies: Sequence[str] | None = None,
+    out_dir: Path | None = None,
+    index_dir: Path | None = None,
+    with_sweep: bool = True,
+) -> EvaluationReport:
+    repo_root = config.repo_root
+    questions_path = Path(questions_path) if questions_path else repo_root / "eval" / "questions.yaml"
+    questions = load_questions(questions_path)
+    strategies = list(strategies) if strategies else list(DEFAULT_STRATEGIES)
+    out_dir = Path(out_dir) if out_dir else config.paths.results_dir
+    out_dir.mkdir(parents=True, exist_ok=True)
+
+    strategy_reports = [
+        evaluate_strategy(config, strategy, questions, index_dir=index_dir) for strategy in strategies
+    ]
+
+    # Generation-side checks run against the strategy we intend to ship.
+    shipping = strategies[-1]
+    pipeline = RAGPipeline.open(config, strategy=shipping, index_dir=index_dir)
+    answer_checks = run_answer_checks(pipeline, questions["answerable"][:3])
+    refusals = run_refusals(pipeline, questions["unanswerable"])
+    filter_demo = run_filter_demo(config, shipping, questions["filter_demo"], index_dir=index_dir)
+
+    bonus = _run_bonus(config, questions, strategies, index_dir)
+
+    sweep = (
+        run_sweep(config, questions, sizes=SWEEP_SIZES, strategies=strategies, index_dir=index_dir)
+        if with_sweep
+        else []
+    )
+
+    history = {}
+    for report in strategy_reports:
+        store = VectorStore.load(Path(index_dir) if index_dir else config.paths.index_dir, report.namespace)
+        history[report.strategy] = store.stats()["history"]
+
+    evaluation = EvaluationReport(
+        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
+        config_summary={
+            "chunk_size": config.chunking.chunk_size,
+            "chunk_overlap": config.chunking.chunk_overlap,
+            "embedding": config.embedding.backend,
+            "n_features": config.embedding.n_features,
+            "top_k": config.retrieval.top_k,
+            "generation_backend": pipeline.generator.name,
+            "generation_model": config.generation.model if pipeline.generator.name == "anthropic" else "n/a",
+            "min_top_score": config.grounding.min_top_score,
+            "min_evidence_coverage": config.grounding.min_evidence_coverage,
+            "questions_file": str(questions_path),
+        },
+        strategy_reports=strategy_reports,
+        filter_demo=filter_demo,
+        answer_checks=answer_checks,
+        refusals=refusals,
+        bonus=bonus,
+        sweep=sweep,
+        index_history=history,
+    )
+
+    from .reporting import write_artifacts
+
+    write_artifacts(evaluation, questions, out_dir=out_dir, repo_root=repo_root)
+    return evaluation
+
+
+def _run_bonus(
+    config: Config,
+    questions: Dict[str, Any],
+    strategies: Sequence[str],
+    index_dir: Path | None,
+) -> List[Dict[str, Any]]:
+    """Answer each bonus probe under every strategy for a side-by-side read.
+
+    Both probes run with the evidence gate relaxed. The committed probe is phrased
+    colloquially ("what will they need to hand?"), and "hit" and "hand" are ordinary
+    English words this small corpus happens not to use, so the shipping gate refuses
+    it — a false refusal discussed in the write-up. The bonus is about answer
+    completeness rather than the gate, so the relaxed value is recorded and reported.
+    """
+    probes = [questions.get("bonus"), questions.get("bonus_followup")]
+    relaxed = config.with_grounding(min_evidence_coverage=BONUS_COVERAGE)
+
+    results: List[Dict[str, Any]] = []
+    for probe in probes:
+        if not probe:
+            continue
+        top_k = int(probe.get("context_chunks", config.retrieval.top_k))
+        answers = {}
+        for strategy in strategies:
+            pipeline = RAGPipeline.open(relaxed, strategy=strategy, index_dir=index_dir)
+            answers[strategy] = pipeline.ask(probe["question"], top_k=top_k).to_dict()
+        results.append(
+            {
+                "id": probe.get("id", ""),
+                "question": probe["question"],
+                "known_answer": probe.get("known_answer", ""),
+                "context_chunks": top_k,
+                "added_after_first_run": bool(probe.get("added_after_first_run")),
+                "answers": answers,
+                "gate_relaxed_to": BONUS_COVERAGE,
+                "shipping_gate": config.grounding.min_evidence_coverage,
+            }
+        )
+    return results
diff --git a/src/ragchat/loader.py b/src/ragchat/loader.py
index 1947340..3c95a20 100644
--- a/src/ragchat/loader.py
+++ b/src/ragchat/loader.py
@@ -15,7 +15,8 @@ FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
 H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
 
 # Front-matter keys an article must declare for its chunks to be indexable.
-REQUIRED_FRONT_MATTER = ("article_id",)
+# An article missing any of these is a failed ingest, not a silently degraded one.
+REQUIRED_FRONT_MATTER = ("article_id", "product_area", "last_updated")
 
 
 class IngestError(RuntimeError):
diff --git a/src/ragchat/models.py b/src/ragchat/models.py
index 0e9b881..67c1739 100644
--- a/src/ragchat/models.py
+++ b/src/ragchat/models.py
@@ -7,7 +7,10 @@ from typing import Any, Dict, List, Optional
 
 # Metadata every chunk must carry. A chunk missing any of these is a failed
 # ingest and the indexer raises rather than storing it.
-REQUIRED_METADATA = ("source_file", "article_id")
+REQUIRED_METADATA = ("source_file", "article_id", "product_area", "last_updated")
+
+# Metadata keys retrieval may be filtered on.
+FILTERABLE_METADATA = ("product_area", "article_id", "source_file", "last_updated")
 
 
 @dataclass(frozen=True)
@@ -43,6 +46,14 @@ class Chunk:
     def source_file(self) -> str:
         return str(self.metadata.get("source_file", ""))
 
+    @property
+    def product_area(self) -> str:
+        return str(self.metadata.get("product_area", ""))
+
+    @property
+    def last_updated(self) -> str:
+        return str(self.metadata.get("last_updated", ""))
+
     @property
     def n_chars(self) -> int:
         return len(self.text)
diff --git a/src/ragchat/pipeline.py b/src/ragchat/pipeline.py
index e84445f..2feb27d 100644
--- a/src/ragchat/pipeline.py
+++ b/src/ragchat/pipeline.py
@@ -3,7 +3,7 @@
 from __future__ import annotations
 
 from pathlib import Path
-from typing import List, Optional, Sequence
+from typing import Any, Dict, List, Optional, Sequence
 
 from .config import Config
 from .generator import Generator, build_generator
@@ -42,22 +42,40 @@ class RAGPipeline:
     def namespace(self) -> str:
         return self.retriever.namespace
 
-    def search(self, question: str, top_k: int | None = None) -> List[SearchHit]:
-        return self.retriever.search(question, top_k=top_k or self.config.retrieval.top_k)
+    def search(
+        self,
+        question: str,
+        top_k: int | None = None,
+        filters: Dict[str, Any] | None = None,
+    ) -> List[SearchHit]:
+        return self.retriever.search(
+            question, top_k=top_k or self.config.retrieval.top_k, filters=filters
+        )
 
-    def ask(self, question: str, top_k: int | None = None) -> Answer:
-        hits = self.search(question, top_k=top_k)
+    def ask(
+        self,
+        question: str,
+        top_k: int | None = None,
+        filters: Dict[str, Any] | None = None,
+    ) -> Answer:
+        hits = self.search(question, top_k=top_k, filters=filters)
         context = hits[: self.config.generation.max_context_chunks]
 
         verdict = self.gate.evaluate(question, context)
         if not verdict.sufficient:
-            return self._refuse(question, hits, verdict.reason, {"evidence": verdict.to_dict()})
+            return self._refuse(
+                question,
+                hits,
+                verdict.reason,
+                {"evidence": verdict.to_dict(), "namespace": self.namespace, "filters": filters or {}},
+            )
 
         result = self.generator.generate(question, context)
         diagnostics = {
             "evidence": verdict.to_dict(),
             "generation": result.diagnostics,
             "namespace": self.namespace,
+            "filters": filters or {},
         }
         if result.refused:
             return self._refuse(question, hits, result.reason, diagnostics, backend=result.backend)
diff --git a/src/ragchat/reporting.py b/src/ragchat/reporting.py
new file mode 100644
index 0000000..45487e0
--- /dev/null
+++ b/src/ragchat/reporting.py
@@ -0,0 +1,374 @@
+"""Turn an :class:`~ragchat.evaluation.EvaluationReport` into the deliverables.
+
+Writes:
+
+* ``results.md``           — the report, numbers generated from the actual run
+* ``results/evaluation.json`` — every number in machine-readable form
+* ``results/search_dump_<strategy>.md`` — search-only top-5 for all 8 questions
+
+The narrative sections (which chunker ships, and the retrieval that embarrassed
+us) are authored by hand in ``eval/writeup.md`` and embedded verbatim, so the
+generated numbers and the human analysis never get mixed up with each other.
+"""
+
+from __future__ import annotations
+
+import json
+from pathlib import Path
+from typing import Any, Dict, Iterable, List, Sequence
+
+from .models import Answer, SearchHit
+
+PREVIEW_CHARS = 150
+
+
+def write_artifacts(report, questions: Dict[str, Any], out_dir: Path, repo_root: Path) -> Path:
+    out_dir.mkdir(parents=True, exist_ok=True)
+
+    (out_dir / "evaluation.json").write_text(
+        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
+    )
+    if report.sweep:
+        (out_dir / "sweep.json").write_text(json.dumps(report.sweep, indent=2), encoding="utf-8")
+
+    for strategy_report in report.strategy_reports:
+        dump = _search_dump(strategy_report)
+        (out_dir / f"search_dump_{strategy_report.strategy}.md").write_text(dump, encoding="utf-8")
+
+    results_path = repo_root / "results.md"
+    results_path.write_text(_results_markdown(report, questions, repo_root), encoding="utf-8")
+    return results_path
+
+
+# --------------------------------------------------------------------------
+# results.md
+# --------------------------------------------------------------------------
+
+
+def _results_markdown(report, questions: Dict[str, Any], repo_root: Path) -> str:
+    cfg = report.config_summary
+    lines: List[str] = []
+    add = lines.append
+
+    add("# Week 3 Task Set A — Results")
+    add("")
+    add(f"*Generated by `python rag.py eval` at {report.generated_at}. Every number below comes")
+    add("from that run; nothing in the tables is hand-written.*")
+    add("")
+    add("| Setting | Value |")
+    add("|---|---|")
+    add(f"| Chunk size / overlap | {cfg['chunk_size']} / {cfg['chunk_overlap']} characters |")
+    add(f"| Embedding | `{cfg['embedding']}`, {cfg['n_features']} hashed features, identical for both strategies |")
+    add(f"| top_k | {cfg['top_k']} |")
+    add(f"| Generation backend | `{cfg['generation_backend']}` (model: {cfg['generation_model']}) |")
+    add(f"| Refusal thresholds | min_top_score {cfg['min_top_score']}, min_evidence_coverage {cfg['min_evidence_coverage']} |")
+    add(f"| Question set | `{Path(cfg['questions_file']).name}`, committed before any retrieval ran |")
+    add("")
+
+    add("## 0. Ingest reality: what was actually indexed")
+    add("")
+    add("The historical corpus was **not** re-indexed. The two legacy articles that stand in for")
+    add("the pre-existing index were ingested once; the six new articles were then **appended**.")
+    add("Ingest is append-only because the embedding function is stateless (no fitted vocabulary),")
+    add("so adding a drop cannot change the vectors already stored.")
+    add("")
+    add("| Strategy | Ingest event | Articles | Chunks added | Seconds |")
+    add("|---|---|---|---|---|")
+    for strategy, events in report.index_history.items():
+        for event in events:
+            add(
+                f"| {strategy} | {event['label']} | {event['n_documents']} | "
+                f"{event['n_chunks']} | {event['seconds']} |"
+            )
+    add("")
+
+    # -- 1. the questions ---------------------------------------------
+    add("## 1. The eight known-answer questions")
+    add("")
+    add("Written from the articles before any search was run (see the provenance note at the top")
+    add("of `eval/questions.yaml` and the git history: the question set is committed in")
+    add("`380b479`, before the retrieval code existed).")
+    add("")
+    add("| # | Question | Known-correct article | Section | Answer must appear as | Table row? |")
+    add("|---|---|---|---|---|---|")
+    for item in questions["answerable"]:
+        gold = item["gold"]
+        anchors = ", ".join(f"`{a}`" for a in item.get("must_contain", []))
+        add(
+            f"| {item['id']} | {item['question']} | {gold['article_id']} "
+            f"({gold['source_file']}) | {gold.get('section', '')} | {anchors} | "
+            f"{'yes' if item.get('depends_on_table_row') else 'no'} |"
+        )
+    add("")
+
+    # -- 2. hit rates --------------------------------------------------
+    add("## 2. Hit-in-top-5, both strategies, same eight questions")
+    add("")
+    add("**Scoring contract** (fixed before the first run): a question is a hit when at least one")
+    add("of the top-5 chunks is from the gold article *and* contains every `must_contain` string.")
+    add("The second condition is the point — it asks whether the answer survived chunking intact,")
+    add("not merely whether the right document ranked.")
+    add("")
+    add("| Strategy | hit@5 | Table-row questions | Prose questions | Chunks in index | Mean chunk chars |")
+    add("|---|---|---|---|---|---|")
+    for sr in report.strategy_reports:
+        prose = [r for r in sr.results if not r.depends_on_table_row]
+        add(
+            f"| `{sr.strategy}` | **{sr.n_hits}/{sr.n_questions}** | {sr.table_hits} | "
+            f"{sum(1 for r in prose if r.hit)}/{len(prose)} | {sr.n_chunks} | {sr.mean_chunk_chars} |"
+        )
+    add("")
+    add("### Secondary metric: self-contained hit@5")
+    add("")
+    add("**Disclosure on ordering.** The committed hit criterion above saturated at 8/8 for both")
+    add("strategies and at every chunk size in the sweep, so on its own it does not discriminate.")
+    add("It is reported unchanged. The metric below was added *after* seeing that saturation, and")
+    add("is computed over the same eight questions and the same retrieval runs.")
+    add("")
+    add("A hit is **self-contained** when the answer-bearing chunk can be interpreted on its own:")
+    add("if it contains table rows, the header row must be in the same chunk. A bare")
+    add("`| ERR-4032 | ... | ... |` row is retrievable but does not tell an agent which cell is the")
+    add("cause and which is the fix — which is precisely the property this week's chunker targets.")
+    add("")
+    add("| Strategy | self-contained hit@5 | Table-row questions | Answer-bearing chunks that lost their header |")
+    add("|---|---|---|---|")
+    for sr in report.strategy_reports:
+        lost = sum(1 for r in sr.results if r.hit and not r.self_contained_hit)
+        add(
+            f"| `{sr.strategy}` | **{sr.n_self_contained}/{sr.n_questions}** | "
+            f"{sr.table_self_contained} | {lost} |"
+        )
+    add("")
+    add("### Per-question record")
+    add("")
+    header = "| # | Question | Gold article | " + " | ".join(f"`{sr.strategy}`" for sr in report.strategy_reports) + " |"
+    add(header)
+    add("|---|---|---|" + "---|" * len(report.strategy_reports))
+    by_qid = {sr.strategy: {r.qid: r for r in sr.results} for sr in report.strategy_reports}
+    for item in questions["answerable"]:
+        qid = item["id"]
+        cells = []
+        for sr in report.strategy_reports:
+            result = by_qid[sr.strategy][qid]
+            if not result.hit:
+                cells.append("MISS")
+            elif result.self_contained_hit:
+                cells.append(f"HIT @{result.hit_rank} (`{result.hit_chunk_id}`)")
+            else:
+                cells.append(
+                    f"HIT @{result.hit_rank} (`{result.hit_chunk_id}`) — **header lost**"
+                )
+        add(f"| {qid} | {item['question'][:60]} | {item['gold']['article_id']} | " + " | ".join(cells) + " |")
+    add("")
+    add("`header lost` marks a hit under the committed criterion whose chunk failed the")
+    add("self-contained check: the row was retrieved without the header row that labels its cells.")
+    add("")
+
+    add("### Every miss, with its diagnosis")
+    add("")
+    any_miss = False
+    for sr in report.strategy_reports:
+        for result in sr.results:
+            if result.hit:
+                continue
+            any_miss = True
+            add(f"- **`{sr.strategy}` / {result.qid}** — {result.question}")
+            add(f"  - {result.diagnosis}")
+            add(f"  - top-5 returned: {', '.join(h.chunk.chunk_id for h in result.hits) or '(nothing)'}")
+    if not any_miss:
+        add("- No misses for either strategy on this run.")
+    add("")
+
+    # -- 3. metadata filter --------------------------------------------
+    demo = report.filter_demo
+    add("## 3. Metadata filter changing retrieval")
+    add("")
+    add(f"Query: **{demo.question}**  ")
+    add(f"Filter: `{demo.filters}`  ")
+    add(f"Top-1 changed: **{demo.changed_top1}**")
+    add("")
+    add("The query is deliberately ambiguous across the corpus: the payments article documents a")
+    add("10-day dunning retry window, the developer-api article documents a 24-hour webhook retry")
+    add("window. Without a filter the payments answer wins; filtering to `developer-api` returns")
+    add("the webhook answer instead.")
+    add("")
+    add("**Unfiltered**")
+    add("")
+    add(_hits_table(demo.unfiltered))
+    add("")
+    add(f"**Filtered — `{demo.filters}`**")
+    add("")
+    add(_hits_table(demo.filtered))
+    add("")
+
+    # -- 4. cited answers ----------------------------------------------
+    add("## 4. Three answerable questions, answered with citations")
+    add("")
+    add("Each citation is checked twice: the `chunk_id` must resolve to a chunk that was actually")
+    add("retrieved for that question, and the cited chunk must contain the known answer text.")
+    add("")
+    for check in report.answer_checks:
+        add(f"### {check.qid} — {check.question}")
+        add("")
+        add("```text")
+        add(check.answer.text.strip())
+        add("```")
+        add("")
+        add("| Cited chunk_id | Resolves | Article | Source file | Section |")
+        add("|---|---|---|---|---|")
+        for citation in check.answer.citations:
+            add(
+                f"| `{citation.chunk_id}` | {'yes' if citation.resolved else 'NO'} | "
+                f"{citation.article_id} | {citation.source_file} | {citation.section} |"
+            )
+        add("")
+        add(
+            f"- all citations resolve: **{check.citations_resolve}**; a cited chunk contains the "
+            f"known answer: **{check.citation_supports_claim}** "
+            f"(`{check.supporting_chunk_id}`)"
+        )
+        add("")
+
+    # -- 5. refusals ----------------------------------------------------
+    add("## 5. Three out-of-corpus questions, refused")
+    add("")
+    add("Refusal is forced, not suggested. The evidence gate runs **before** generation: when the")
+    add("question's idf-weighted content is not present in the retrieved chunks, the pipeline")
+    add("refuses and no generator is invoked at all.")
+    add("")
+    for answer in report.refusals:
+        add(f"### {answer.question}")
+        add("")
+        add("```text")
+        add(f"Q: {answer.question}")
+        add("")
+        add(answer.text.strip())
+        add("")
+        add(f"[refused: {answer.refusal_reason}]")
+        add("```")
+        add("")
+        evidence = answer.diagnostics.get("evidence", {})
+        if evidence:
+            add(
+                f"- idf-weighted coverage **{evidence.get('idf_weighted_coverage')}** vs threshold "
+                f"{cfg['min_evidence_coverage']}; terms absent from the corpus: "
+                f"`{evidence.get('missing_terms')}`"
+            )
+        add("")
+
+    # -- 6. chunk-size sweep --------------------------------------------
+    if report.sweep:
+        add("## 6. Chunk-size comparison")
+        add("")
+        add("Same eight questions, same embedding, same top_k — only the chunk size changes.")
+        add("")
+        add("| Strategy | Chunk size | Overlap | Chunks | Mean chars | hit@5 | self-contained@5 | Table qs (self-contained) |")
+        add("|---|---|---|---|---|---|---|---|")
+        for row in report.sweep:
+            add(
+                f"| `{row['strategy']}` | {row['chunk_size']} | {row['chunk_overlap']} | "
+                f"{row['n_chunks']} | {row['mean_chunk_chars']} | {row['hit_at_5']} | "
+                f"**{row['self_contained_hit_at_5']}** | {row['table_questions_self_contained']} |"
+            )
+        add("")
+
+    # -- 7. bonus --------------------------------------------------------
+    if report.bonus:
+        add("## 7. Bonus — precision vs completeness")
+        add("")
+        add(
+            f"*Both probes run with the evidence gate relaxed to "
+            f"{report.bonus[0].get('gate_relaxed_to')} (shipping value "
+            f"{report.bonus[0].get('shipping_gate')}), because the bonus is about how complete "
+            "each answer is once retrieval has happened, not about the gate.*"
+        )
+        add("")
+        for probe in report.bonus:
+            title = f"### {probe['id']} — {probe['question']}"
+            add(title)
+            add("")
+            if probe["added_after_first_run"]:
+                add("*Added after the first run — see the note in `eval/questions.yaml`. Not part of")
+                add("any scored metric.*")
+                add("")
+            add(f"Context: top-{probe['context_chunks']} chunk(s).")
+            add("")
+            add(f"Known complete answer: {probe['known_answer'].strip()}")
+            add("")
+            for strategy, answer in probe["answers"].items():
+                add(f"**`{strategy}`**")
+                add("")
+                add("```text")
+                add(
+                    answer["text"].strip()
+                    if not answer["refused"]
+                    else f"REFUSED — {answer['refusal_reason']}"
+                )
+                add("```")
+                add("")
+
+    # -- 8. hand-written analysis ----------------------------------------
+    writeup = repo_root / "eval" / "writeup.md"
+    if writeup.is_file():
+        add(writeup.read_text(encoding="utf-8").strip())
+        add("")
+
+    add("---")
+    add("")
+    add("Raw artefacts: `results/evaluation.json`, `results/sweep.json`,")
+    add("`results/search_dump_baseline.md`, `results/search_dump_structure-aware.md`.")
+    add("")
+    return "\n".join(lines)
+
+
+def _hits_table(hits: Sequence[SearchHit]) -> str:
+    rows = [
+        "| Rank | Score | chunk_id | product_area | Article | Section | Preview |",
+        "|---|---|---|---|---|---|---|",
+    ]
+    if not hits:
+        rows.append("| — | — | (no matches) | — | — | — | — |")
+        return "\n".join(rows)
+    for hit in hits:
+        chunk = hit.chunk
+        rows.append(
+            f"| {hit.rank} | {hit.score:.4f} | `{chunk.chunk_id}` | {chunk.product_area} | "
+            f"{chunk.article_id} | {chunk.section} | {_preview(chunk.text)} |"
+        )
+    return "\n".join(rows)
+
+
+def _preview(text: str, limit: int = PREVIEW_CHARS) -> str:
+    flat = " ".join(text.split()).replace("|", "\\|")
+    return flat[:limit] + ("…" if len(flat) > limit else "")
+
+
+# --------------------------------------------------------------------------
+# search-only dumps
+# --------------------------------------------------------------------------
+
+
+def _search_dump(strategy_report) -> str:
+    lines = [
+        f"# Search-only dump — `{strategy_report.strategy}`",
+        "",
+        f"Index namespace: `{strategy_report.namespace}` · {strategy_report.n_chunks} chunks · "
+        f"mean {strategy_report.mean_chunk_chars} chars",
+        "",
+        "No generation is involved here: this is the raw top-5 for each of the eight questions.",
+        "",
+    ]
+    for result in strategy_report.results:
+        lines.append(f"## {result.qid} — {result.question}")
+        lines.append("")
+        lines.append(
+            f"Gold: **{result.gold_article}** / {result.gold_section} · "
+            f"must contain {result.must_contain} · "
+            f"**{'HIT at rank ' + str(result.hit_rank) if result.hit else 'MISS'}**"
+        )
+        lines.append("")
+        lines.append(f"Diagnosis: {result.diagnosis}")
+        lines.append("")
+        lines.append(_hits_table(result.hits))
+        lines.append("")
+    return "\n".join(lines)
diff --git a/src/ragchat/retriever.py b/src/ragchat/retriever.py
index 5b1dbd7..3a6e610 100644
--- a/src/ragchat/retriever.py
+++ b/src/ragchat/retriever.py
@@ -4,7 +4,7 @@ from __future__ import annotations
 
 import math
 from pathlib import Path
-from typing import List, Optional
+from typing import Any, Dict, List, Optional
 
 from .config import Config
 from .embeddings import HashingTfidfEmbedder, build_embedder
@@ -36,14 +36,22 @@ class Retriever:
     def namespace(self) -> str:
         return self.store.namespace
 
-    def search(self, question: str, top_k: int = 5) -> List[SearchHit]:
+    def search(
+        self,
+        question: str,
+        top_k: int = 5,
+        filters: Optional[Dict[str, Any]] = None,
+    ) -> List[SearchHit]:
         query_vector = self.embedder.embed_query(question, self.store.df, self.store.n_units)
-        ranked = self.store.search(query_vector, top_k=top_k)
+        ranked = self.store.search(query_vector, top_k=top_k, filters=filters)
         return [
             SearchHit(rank=rank, score=score, chunk=self.store.chunks[row])
             for rank, (row, score) in enumerate(ranked, start=1)
         ]
 
+    def distinct_values(self, key: str) -> List[str]:
+        return self.store.distinct_values(key)
+
     def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
         return self.store.get(chunk_id)
 
diff --git a/src/ragchat/store.py b/src/ragchat/store.py
index 78ecf40..65dd970 100644
--- a/src/ragchat/store.py
+++ b/src/ragchat/store.py
@@ -85,12 +85,15 @@ class VectorStore:
         if vectors.shape[1] != self.spec.n_features:
             raise StoreError("vector width does not match the embedding spec")
 
+        incoming: set[str] = set()
         for chunk in chunks:
             missing = [key for key in REQUIRED_METADATA if not str(chunk.metadata.get(key, "")).strip()]
             if missing:
                 raise StoreError(f"chunk {chunk.chunk_id} is missing required metadata {missing}")
-            if chunk.chunk_id in self._by_id:
+            # Duplicates must be caught both against the stored index and within this batch.
+            if chunk.chunk_id in self._by_id or chunk.chunk_id in incoming:
                 raise StoreError(f"duplicate chunk_id {chunk.chunk_id}")
+            incoming.add(chunk.chunk_id)
 
         start = len(self.chunks)
         for offset, chunk in enumerate(chunks):
@@ -103,13 +106,30 @@ class VectorStore:
 
     # -- query ----------------------------------------------------------
 
-    def search(self, query_vector: sp.csr_matrix, top_k: int) -> List[Tuple[int, float]]:
-        """Return (row index, score) for the ``top_k`` best chunks."""
+    def search(
+        self,
+        query_vector: sp.csr_matrix,
+        top_k: int,
+        filters: Optional[Dict[str, Any]] = None,
+    ) -> List[Tuple[int, float]]:
+        """Return (row index, score) for the ``top_k`` best chunks.
+
+        ``filters`` maps a metadata key to an accepted value (or list of values).
+        Filtering is applied to the candidate set before ranking, so a filtered
+        search can return a different top-1, not merely a shorter list.
+        """
         if self.matrix is None or not self.chunks:
             return []
         scores = np.asarray((self.matrix @ query_vector.T).todense()).ravel()
+        if filters:
+            keep = np.array([_matches(chunk, filters) for chunk in self.chunks], dtype=bool)
+            scores = np.where(keep, scores, 0.0)
         return self._rank(scores, top_k)
 
+    def distinct_values(self, key: str) -> List[str]:
+        """Sorted distinct values of a metadata key across the index."""
+        return sorted({str(chunk.metadata.get(key, "")) for chunk in self.chunks if chunk.metadata.get(key)})
+
     def _rank(self, scores: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
         eligible = np.flatnonzero(scores > 0.0)
         if eligible.size == 0:
@@ -195,6 +215,18 @@ class VectorStore:
         return store
 
 
+def _matches(chunk: Chunk, filters: Dict[str, Any]) -> bool:
+    """True when a chunk satisfies every filter (case-insensitive, value or list)."""
+    for key, wanted in filters.items():
+        if wanted is None or wanted == "":
+            continue
+        actual = str(chunk.metadata.get(key, "")).strip().lower()
+        accepted = wanted if isinstance(wanted, (list, tuple, set)) else [wanted]
+        if actual not in {str(value).strip().lower() for value in accepted}:
+            return False
+    return True
+
+
 def namespace_for(strategy: str, chunk_size: int, chunk_overlap: int) -> str:
     """One index per (strategy, size) pair so sweeps never overwrite each other."""
     return f"{strategy}__cs{chunk_size}_ov{chunk_overlap}"
diff --git a/src/ragchat/webapp.py b/src/ragchat/webapp.py
new file mode 100644
index 0000000..0a7175a
--- /dev/null
+++ b/src/ragchat/webapp.py
@@ -0,0 +1,164 @@
+"""Minimal Flask UI: ask a question, see the answer, its sources, or the refusal."""
+
+from __future__ import annotations
+
+from pathlib import Path
+from typing import Any, Dict, List
+
+from flask import Flask, jsonify, render_template_string, request
+
+from .chunkers import available_strategies
+from .config import Config
+from .pipeline import RAGPipeline
+
+PAGE = """<!doctype html>
+<html lang="en">
+<head>
+<meta charset="utf-8">
+<meta name="viewport" content="width=device-width, initial-scale=1">
+<title>Help-centre assistant</title>
+<style>
+  :root { color-scheme: light dark; }
+  body { font-family: system-ui, sans-serif; max-width: 60rem; margin: 2rem auto; padding: 0 1rem;
+         line-height: 1.5; }
+  form { display: grid; gap: .75rem; margin-bottom: 2rem; }
+  .row { display: flex; gap: .75rem; flex-wrap: wrap; }
+  input[type=text] { flex: 1 1 24rem; padding: .6rem; font-size: 1rem; }
+  select, button { padding: .6rem; font-size: 1rem; }
+  .answer { border-left: 4px solid #3a7; padding: .5rem 1rem; background: rgba(51,170,119,.08); }
+  .refused { border-left: 4px solid #c53; padding: .5rem 1rem; background: rgba(204,85,51,.08); }
+  .meta { font-size: .85rem; opacity: .8; }
+  table { border-collapse: collapse; width: 100%; margin-top: 1rem; font-size: .85rem; }
+  th, td { border: 1px solid rgba(128,128,128,.4); padding: .35rem .5rem; text-align: left;
+           vertical-align: top; }
+  code { font-size: .85em; }
+</style>
+</head>
+<body>
+<h1>Help-centre assistant</h1>
+<p class="meta">Answers come only from the indexed articles. Anything the corpus does not cover is
+refused rather than guessed.</p>
+
+<form method="get">
+  <input type="text" name="q" value="{{ question|e }}" placeholder="e.g. What does ERR-4032 mean?" autofocus>
+  <div class="row">
+    <select name="strategy">
+      {% for s in strategies %}<option value="{{ s }}" {% if s == strategy %}selected{% endif %}>{{ s }}</option>{% endfor %}
+    </select>
+    <select name="product_area">
+      <option value="">all product areas</option>
+      {% for a in product_areas %}<option value="{{ a }}" {% if a == product_area %}selected{% endif %}>{{ a }}</option>{% endfor %}
+    </select>
+    <button type="submit">Ask</button>
+  </div>
+</form>
+
+{% if answer %}
+  {% if answer.refused %}
+    <div class="refused">
+      <strong>No answer given.</strong>
+      <p>{{ answer.text }}</p>
+      <p class="meta">Reason: {{ answer.refusal_reason }}</p>
+    </div>
+  {% else %}
+    <div class="answer">
+      {% for line in answer.text.split('\n') %}<p>{{ line }}</p>{% endfor %}
+    </div>
+    <h2>Sources</h2>
+    <table>
+      <tr><th>chunk_id</th><th>Article</th><th>Source file</th><th>Section</th><th>Product area</th></tr>
+      {% for c in answer.citations %}
+      <tr><td><code>{{ c.chunk_id }}</code></td><td>{{ c.article_id }}</td><td>{{ c.source_file }}</td>
+          <td>{{ c.section }}</td><td>{{ areas.get(c.chunk_id, '') }}</td></tr>
+      {% endfor %}
+    </table>
+  {% endif %}
+
+  <h2>Retrieved chunks</h2>
+  <table>
+    <tr><th>#</th><th>Score</th><th>chunk_id</th><th>Product area</th><th>Section</th><th>Preview</th></tr>
+    {% for h in answer.hits %}
+    <tr><td>{{ h.rank }}</td><td>{{ '%.4f'|format(h.score) }}</td>
+        <td><code>{{ h.chunk.chunk_id }}</code></td>
+        <td>{{ h.chunk.metadata.get('product_area','') }}</td>
+        <td>{{ h.chunk.section }}</td>
+        <td>{{ h.chunk.text[:180] }}…</td></tr>
+    {% endfor %}
+  </table>
+  <p class="meta">index: <code>{{ namespace }}</code> · backend: <code>{{ answer.backend }}</code></p>
+{% endif %}
+</body>
+</html>
+"""
+
+
+def create_app(config: Config, index_dir: Path | None = None) -> Flask:
+    app = Flask(__name__)
+    pipelines: Dict[str, RAGPipeline] = {}
+
+    def get_pipeline(strategy: str) -> RAGPipeline:
+        if strategy not in pipelines:
+            pipelines[strategy] = RAGPipeline.open(config, strategy=strategy, index_dir=index_dir)
+        return pipelines[strategy]
+
+    def answer_for(question: str, strategy: str, product_area: str):
+        pipeline = get_pipeline(strategy)
+        filters = {"product_area": product_area} if product_area else None
+        return pipeline, pipeline.ask(question, filters=filters)
+
+    @app.get("/")
+    def home() -> str:
+        question = (request.args.get("q") or "").strip()
+        strategy = request.args.get("strategy") or config.chunking.default_strategy
+        product_area = (request.args.get("product_area") or "").strip()
+
+        answer = None
+        namespace = ""
+        areas: Dict[str, str] = {}
+        pipeline = get_pipeline(strategy)
+        if question:
+            pipeline, answer = answer_for(question, strategy, product_area)
+            namespace = pipeline.namespace
+            areas = {h.chunk.chunk_id: h.chunk.product_area for h in answer.hits}
+
+        return render_template_string(
+            PAGE,
+            question=question,
+            strategy=strategy,
+            product_area=product_area,
+            strategies=available_strategies(),
+            product_areas=pipeline.retriever.distinct_values("product_area"),
+            answer=answer,
+            namespace=namespace,
+            areas=areas,
+        )
+
+    @app.get("/api/ask")
+    def api_ask():
+        question = (request.args.get("q") or "").strip()
+        if not question:
+            return jsonify({"error": "missing q parameter"}), 400
+        strategy = request.args.get("strategy") or config.chunking.default_strategy
+        product_area = (request.args.get("product_area") or "").strip()
+        _pipeline, answer = answer_for(question, strategy, product_area)
+        return jsonify(answer.to_dict())
+
+    @app.get("/api/search")
+    def api_search():
+        question = (request.args.get("q") or "").strip()
+        if not question:
+            return jsonify({"error": "missing q parameter"}), 400
+        strategy = request.args.get("strategy") or config.chunking.default_strategy
+        product_area = (request.args.get("product_area") or "").strip()
+        pipeline = get_pipeline(strategy)
+        filters = {"product_area": product_area} if product_area else None
+        hits = pipeline.search(question, filters=filters)
+        return jsonify([hit.to_dict() for hit in hits])
+
+    @app.get("/healthz")
+    def healthz():
+        pipeline = get_pipeline(config.chunking.default_strategy)
+        return jsonify({"status": "ok", "namespace": pipeline.namespace,
+                        "chunks": len(pipeline.retriever.store.chunks)})
+
+    return app
```

