# Billing Migration Help-Centre Assistant

A retrieval-augmented (RAG) chatbot over a customer-support help centre. It answers only
from the indexed articles, cites a resolvable `chunk_id` for every claim internally, and
**refuses** anything the corpus does not cover instead of guessing.

*Originally built as a Week 3 "AI Engineering League" practical (Task Set A): ingest a new
help-centre drop, compare two chunking strategies on questions with known answers, and
prove the app won't hallucinate.*

This repository contains the base app **plus** the week's extension: a second,
structure-aware chunking strategy that never separates a table row from its header, full
chunk metadata with retrieval-time filtering, a chat UI, and an evaluation harness that
produces [results.md](results.md) from a real measured run — not hand-written numbers.

## Headline numbers

| | `baseline` | `structure-aware` |
|---|---|---|
| hit-in-top-5 (committed metric) | 8/8 | 8/8 |
| self-contained hit-in-top-5 | 3/8 | **8/8** |
| table-row questions, self-contained | 0/4 | **4/4** |

Full report, per-question records, filter demonstration, cited answers, refusal
transcripts, chunk-size sweep and the written analysis: **[results.md](results.md)**.

## Quick start

```bash
pip install -r requirements.txt

python rag.py ingest --label "legacy-corpus" data/legacy_articles   # the pre-existing index
python rag.py ingest --label "week3-new-drop"                       # the six new articles

python rag.py ask "What does ERR-4032 mean and what is the fix?"
python rag.py ask "What is the refund SLA for a disputed charge?"   # refuses
python rag.py eval                                                   # regenerates results.md

streamlit run streamlit_app.py                                       # chat UI, on :8501
python rag.py serve                                                  # dev UI + JSON API, on :5000
```

No API key is required. Without Anthropic credentials the pipeline uses a deterministic
extractive generator that quotes the retrieved chunks verbatim; with credentials it uses
Claude (`claude-opus-5`) automatically. Both paths are grounded, cited and can refuse.

## Frontends

| | Run with | What you get |
|---|---|---|
| **Chat UI** (recommended) | `streamlit run streamlit_app.py` | Plain chatbot-style conversation. Greetings/thanks get a natural reply; real questions go through the full grounded pipeline. No chunk IDs or source tables shown — just the answer. |
| **Dev UI** | `python rag.py serve` | Shows the retrieved chunks and resolved citations alongside the answer — useful for inspecting *why* the app answered or refused. |
| **CLI** | `python rag.py ask "..."` | Scriptable; `--json` for machine-readable output. |

> **Windows note:** if your global Python has an older `protobuf` pinned by another package
> (e.g. TensorFlow), Streamlit may fail to import. Rather than upgrading `protobuf` globally
> and risking that other package, create an isolated environment for this project:
> `python -m venv .venv && .venv/Scripts/pip install -r requirements.txt`, then run commands
> through `.venv/Scripts/python.exe` (or activate the venv first).

## How it works

```
articles ──▶ loader ──▶ chunker ──▶ embedder ──▶ vector store
 (front      (metadata   (baseline   (stateless    (append-only,
  matter)     required)   or          hashing,      metadata
                          structure-  lnc.ltc)      filters)
                          aware)
                                                        │
question ──▶ retriever ──▶ evidence gate ──▶ generator ──▶ citation validator ──▶ answer
                              │                                    │
                              └── refuse (no model call) ──────────┴── refuse (fail closed)
```

**Chunking.** `baseline` is a recursive character splitter: it walks a separator hierarchy
and packs fixed-size windows, which cuts troubleshooting tables wherever the budget runs
out. `structure-aware` parses markdown blocks first and guarantees that a table row is
never separated from its header row, that a row is never split down the middle, and that
chunks never span section boundaries.

**Embedding.** A stateless hashing vectoriser with classic `lnc.ltc` weighting (log term
frequency on documents, log-tf × idf on queries, both L2 normalised). Nothing is fitted, so
appending a new drop cannot change the vectors already stored — ingest is genuinely
incremental. The same embedder is used for every strategy and every chunk size, so a
difference between two runs is attributable to chunking alone.

**Refusal is forced, not suggested.** An evidence gate runs *before* generation and refuses
when the question's idf-weighted content is not present in the retrieved chunks — no model
is called at all. Whatever the generator returns is then citation-validated: a claim with
no citation, or a citation naming a chunk that was not retrieved, fails closed into a
refusal. See `src/ragchat/grounding.py`.

## Layout

```
config.yaml               all tunable settings (thresholds, sizes, model, paths)
rag.py                    CLI launcher, no install needed
streamlit_app.py          chatbot-style chat UI (streamlit run streamlit_app.py)
data/articles/            the six new help-centre articles
data/legacy_articles/     two older articles standing in for the pre-existing index
eval/questions.yaml       8 known-answer + 3 out-of-corpus questions, committed first
eval/writeup.md           hand-written analysis embedded into results.md
src/ragchat/
  chunkers/               baseline.py, structure_aware.py, registry
  loader.py               front matter + required-metadata validation
  embeddings.py           stateless hashing embedder
  store.py                append-only vector store, metadata filtering
  indexer.py              ingest pipeline
  retriever.py            search + idf lookup
  grounding.py            evidence gate, citation validation
  generator.py            Claude and extractive backends
  pipeline.py             retrieve -> gate -> generate -> validate
  evaluation.py           the harness behind `rag.py eval`
  reporting.py            writes results.md and the search dumps
  webapp.py               Flask UI + JSON API
tests/                    48 tests, no network required
results/                  generated artefacts (json + search dumps)
docs/code_diff.md         the diff adding the second chunker and the metadata fields
```

## CLI

| Command | What it does |
|---|---|
| `ingest [paths]` | Chunk, embed and **append** documents to the index |
| `search "q"` | Search only, no generation (`--product-area`, `--top-k`, `--json`) |
| `ask "q"` | Retrieve, then answer with citations or refuse |
| `eval` | Full evaluation; regenerates `results.md` and `results/` |
| `sweep --sizes 400 800` | Chunk-size sweep only |
| `stats` | What is in the index, including the ingest history |
| `serve` | Web UI and JSON API (`/api/ask`, `/api/search`, `/healthz`) |

Global flags: `--strategy baseline|structure-aware`, `--chunk-size`, `--chunk-overlap`,
`--index-dir`, `--config`. Each `(strategy, size, overlap)` combination gets its own index
namespace, so sweeps never overwrite each other.

## Tests

```bash
python -m unittest discover -s tests -t .
```

48 tests covering chunker guarantees (including the header/row property on the real
corpus), metadata validation, store round-trips and filtering, the refusal gate, citation
validation, and an end-to-end pass that builds a real index in a temp directory.

## Configuration notes

Everything tunable lives in `config.yaml`. Two values deserve comment:

- `grounding.min_evidence_coverage` (0.55) is the refusal threshold, calibrated so the
  eight answerable questions (0.63–1.00) sit above it and the three out-of-corpus ones
  (0.15–0.46) below. This is fitted on the evaluation set — see the limitations section of
  `results.md`.
- `generation.max_tokens` (8000) leaves room for adaptive thinking on `claude-opus-5`,
  where `max_tokens` caps thinking *and* response text together.
