# Billing Migration Help-Centre Assistant

A retrieval-augmented (RAG) chatbot over a customer-support help centre. It answers only
from the indexed articles, cites a resolvable `chunk_id` for every claim internally, and
**refuses** anything the corpus does not cover instead of guessing.

*Built as a running "AI Engineering League" practical. Week 3 (Task Set A): ingest a new
help-centre drop, compare two chunking strategies on questions with known answers, and prove
the app won't hallucinate. Week 4 (Task Set A): label a failing question's failure as
retrieval (wrong document) or generation (right document, wrong answer), make one retrieval
change, and prove it with a before/after number. Week 5 (Task Set A): read ~20 real traces
drawn by a fair random sample, write an honest note on each, and turn that into a ranked,
named taxonomy of what actually goes wrong.*

This repository contains the base app, the Week 3 extension (a second, structure-aware
chunking strategy, chunk metadata with retrieval-time filtering, a chat UI, and an evaluation
harness that produces [results.md](results.md)), the Week 4 extension (a hybrid
semantic+BM25 retriever, a failing-question harness that produces
[results_week4.md](results_week4.md), and an inspection view for looking at any question's
retrieval and answer side by side), and the Week 5 extension (a random-sampling trace harness
and an open-coding/taxonomy report that produces [results_week5.md](results_week5.md)).

## Headline numbers

**Week 3 — chunking.**

| | `baseline` | `structure-aware` |
|---|---|---|
| hit-in-top-5 (committed metric) | 8/8 | 8/8 |
| self-contained hit-in-top-5 | 3/8 | **8/8** |
| table-row questions, self-contained | 0/4 | **4/4** |

Full report, per-question records, filter demonstration, cited answers, refusal
transcripts, chunk-size sweep and the written analysis: **[results.md](results.md)**.

**Week 4 — retrieval.** 18 questions caught failing against Week 3's semantic-only search;
one change (BM25 keyword search, fused with semantic search by reciprocal rank fusion):

| | semantic (before) | hybrid (after) |
|---|---|---|
| hit-rate@3 | 2/18 (11%) | **11/18 (61%)** |
| recall@3 | 0.065 | **0.593** |
| MRR@10 | 0.262 | **0.310** |

9 of 16 originally-failing questions fixed, 0 regressed against Week 3's own question set.
Two questions (`N1`, `N2`) are deliberately left unfixed — the correct chunk was already
retrieved at rank 1-2; the pipeline still refused, which is a *generation* failure a
retrieval change cannot and should not touch. Full per-question evidence, what did and did
not get fixed, and the mechanism: **[results_week4.md](results_week4.md)**.

**Week 5 — error analysis.** 20 questions drawn by `random.Random(5).sample(...)` from an
84-question pool built to cover the whole corpus (every troubleshooting-table error code,
prose facts, colloquial paraphrases, out-of-corpus questions, and hard multi-part/ambiguous
cases) — run through the pipeline exactly as it ships today, read one at a time, and coded by
hand before any grouping happened:

| | Count |
|---|---|
| Traces sampled / answered / refused | 20 / 7 / 13 |
| Refusals that were actually wrong (corpus had the answer) | 11 / 13 |
| Top-ranked problem | **False refusal on answerable questions (informal phrasing)** — 10/20 traces, score 30 |

The evidence gate's idf-weighted coverage check refuses many answerable, informally-phrased
support tickets even when the correct chunk is retrieved at rank 1, because ordinary filler
words ("getting", "trying", "hey", "idea"...) are absent from this small corpus and not on the
gate's stoplist. One rarer but more dangerous pattern also surfaced: a colloquially-phrased
question got a fully-cited, confident answer pulled from the *wrong* article. Full per-trace
notes, the ranked taxonomy, and the fix target chosen for next week (not yet implemented):
**[results_week5.md](results_week5.md)**.

## Quick start

```bash
pip install -r requirements.txt

python rag.py ingest --label "legacy-corpus" data/legacy_articles   # the pre-existing index
python rag.py ingest --label "week3-new-drop"                       # the six new articles

python rag.py ask "What does ERR-4032 mean and what is the fix?"
python rag.py ask "What is the refund SLA for a disputed charge?"   # refuses
python rag.py eval                                                   # regenerates results.md
python rag.py eval-failures                                          # regenerates results_week4.md
python rag.py trace-sample                                           # Week 5: sample 20 questions, record complete traces
python rag.py eval-errors                                            # regenerates results_week5.md

python rag.py inspect "Support ticket: account throwing ERR-4117, need root cause and remediation steps quickly." --compare
                                                                      # question, fetched chunks (both retrieval modes), and the final answer, side by side

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
             (semantic       │                                    │
              or hybrid)     └── refuse (no model call) ──────────┴── refuse (fail closed)
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

**Retrieval.** `semantic` is the Week 3 cosine search over the hashed vectors, unchanged.
`hybrid` (the shipped default) additionally ranks every chunk by BM25 keyword score,
computed at query time from the same tokenizer the embedder uses, and fuses the two rankings
with reciprocal rank fusion (RRF). The reason: the semantic vectors are L2-normalised, so a
long troubleshooting-table chunk with several distinct error codes dilutes each individual
code's contribution, while BM25's score is a plain idf-weighted sum where one rare, exact
term — an error code appearing nowhere else in the corpus — dominates regardless of how many
other terms the chunk carries. See `src/ragchat/bm25.py` and
[results_week4.md](results_week4.md) for the measured effect.

**Refusal is forced, not suggested.** An evidence gate runs *before* generation and refuses
when the question's idf-weighted content is not present in the retrieved chunks — no model
is called at all. Whatever the generator returns is then citation-validated: a claim with
no citation, or a citation naming a chunk that was not retrieved, fails closed into a
refusal. See `src/ragchat/grounding.py`.

## Layout

```
config.yaml               all tunable settings (thresholds, sizes, model, paths, retrieval mode)
rag.py                    CLI launcher, no install needed
streamlit_app.py          chatbot-style chat UI (streamlit run streamlit_app.py)
data/articles/            the six new help-centre articles
data/legacy_articles/     two older articles standing in for the pre-existing index
eval/questions.yaml       8 known-answer + 3 out-of-corpus questions, committed first (Week 3)
eval/writeup.md           hand-written analysis embedded into results.md (Week 3)
eval/week4_questions.yaml 18 failing questions, caught empirically against the Week 3 index (Week 4)
eval/week4_writeup.md     hand-written analysis embedded into results_week4.md (Week 4)
eval/week5_questions.yaml 84-question candidate pool for random trace sampling (Week 5)
eval/week5_open_coding.yaml  one honest note + severity + problem group per sampled trace (Week 5)
eval/week5_writeup.md     hand-written analysis embedded into results_week5.md (Week 5)
src/ragchat/
  chunkers/               baseline.py, structure_aware.py, registry
  loader.py               front matter + required-metadata validation
  embeddings.py           stateless hashing embedder
  bm25.py                 BM25 keyword index + reciprocal rank fusion
  store.py                append-only vector store, metadata filtering
  indexer.py              ingest pipeline
  retriever.py            search (semantic and hybrid) + idf lookup
  grounding.py            evidence gate, citation validation
  generator.py            Claude and extractive backends
  pipeline.py             retrieve -> gate -> generate -> validate
  metrics.py              hit-rate@k, recall@k, MRR
  evaluation.py           the harness behind `rag.py eval` (Week 3)
  failure_analysis.py     the harness behind `rag.py eval-failures` (Week 4)
  tracing.py              random sampling, trace collection, open-coding taxonomy (Week 5)
  reporting.py            writes results.md / results_week4.md / results_week5.md and the search dumps
  webapp.py               Flask UI + JSON API
tests/                    89 tests, no network required
results/                  generated artefacts (json + search dumps + week5_traces/*.md)
docs/code_diff.md         the diff adding the second chunker and the metadata fields
```

## CLI

| Command | What it does |
|---|---|
| `ingest [paths]` | Chunk, embed and **append** documents to the index |
| `search "q"` | Search only, no generation (`--product-area`, `--top-k`, `--retrieval-mode`, `--json`) |
| `ask "q"` | Retrieve, then answer with citations or refuse (`--retrieval-mode semantic\|hybrid`) |
| `inspect "q"` | The question, what was fetched, and the final answer, side by side (`--compare` shows both retrieval modes) |
| `eval` | Full Week 3 evaluation; regenerates `results.md` and `results/` |
| `eval-failures` | Week 4 evaluation; regenerates `results_week4.md` and `results/week4_evaluation.json` |
| `trace-sample` | Week 5: draw a random sample from the question pool and record complete traces (`--n`, `--seed`, `--pool`, `--out`) |
| `eval-errors` | Week 5 evaluation; regenerates `results_week5.md` from a fresh trace sample plus `eval/week5_open_coding.yaml` |
| `sweep --sizes 400 800` | Chunk-size sweep only |
| `stats` | What is in the index, including the ingest history |
| `serve` | Web UI and JSON API (`/api/ask`, `/api/search`, `/healthz`) |

Global flags: `--strategy baseline|structure-aware`, `--chunk-size`, `--chunk-overlap`,
`--index-dir`, `--config`. Each `(strategy, size, overlap)` combination gets its own index
namespace, so sweeps never overwrite each other. `--retrieval-mode semantic|hybrid` on
`search`/`ask`/`inspect` overrides `config.yaml`'s `retrieval.mode` for one call.

## Tests

```bash
python -m unittest discover -s tests -t .
```

89 tests covering chunker guarantees (including the header/row property on the real
corpus), metadata validation, store round-trips and filtering, the refusal gate, citation
validation, an end-to-end pass that builds a real index in a temp directory, BM25 and RRF
fusion correctness, the retrieval metrics, the Week 4 claim that hybrid search improves
hit-rate@3 on the failing-question set without regressing Week 3's own questions, and the
Week 5 claims that the trace sample is deterministic given its seed, every sampled trace has
exactly one open-coding note, and the taxonomy it builds is ranked correctly.

## Configuration notes

Everything tunable lives in `config.yaml`. A few values deserve comment:

- `grounding.min_evidence_coverage` (0.55) is the refusal threshold, calibrated so the
  eight answerable questions (0.63–1.00) sit above it and the three out-of-corpus ones
  (0.15–0.46) below. This is fitted on the evaluation set — see the limitations section of
  `results.md`. `results_week4.md`'s `N1`/`N2` show it does not generalise to colloquial
  support-ticket phrasing; that limitation is carried over unchanged, not fixed, this week.
- `generation.max_tokens` (8000) leaves room for adaptive thinking on `claude-opus-5`,
  where `max_tokens` caps thinking *and* response text together.
- `retrieval.mode` (`hybrid`) ships as the new default because it measurably improves
  hit-rate@3 on realistic support-ticket phrasing with zero regressions on Week 3's own
  question set — see `results_week4.md` for the before/after and `retrieval.rrf_k` /
  `bm25_k1` / `bm25_b` for the fusion constants.
