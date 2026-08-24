## 4. Why hybrid search, and only hybrid search

The brief for this week is explicit that the point is not to try everything — it's to make
**one** change and know what it did. `src/ragchat/bm25.py` adds a keyword (BM25) ranker
alongside the existing semantic search and fuses the two with reciprocal rank fusion (RRF).
Reranking, query rewriting and HyDE were deliberately not touched, for two reasons beyond
"the brief said one change": first, this project's "semantic" search is itself a bag-of-words
hashing vectoriser with idf weighting, not a dense neural embedding — so the failure mode
worth fixing here is specifically the *weighting* of exact terms, which is exactly what a
keyword ranker targets, not a meaning-representation problem a reranker or query rewrite
would address. Second, adding a keyword ranker costs nothing in the project's own terms: no
new dependency, no network call, no model — `BM25Index` is a plain Python class scored at
query time from the same tokenizer the embedder already uses.

## 5. The mechanism, diagnosed before it was fixed

Every `keyword_dilution` question in `eval/week4_questions.yaml` follows the same shape: a
realistic support-ticket sentence ("Support ticket: account throwing ERR-4117, need root
cause and remediation steps quickly.") that names an error code appearing exactly once in
the whole corpus. Under semantic-only search the answer-bearing chunk consistently lands at
rank 4-6, never in the top-3 — checked systematically across the corpus's error codes with a
fixed template, not cherry-picked. Under hybrid search the same chunk lands at rank 2-3.

The reason is the semantic vectors' own normalisation, not "semantic search is fuzzy" — the
same conclusion `results.md` section 9 reached for question B1 last week, now confirmed as a
general pattern rather than a one-off. The `lnc.ltc` vectors are L2-normalised: a chunk's
every term is divided by the norm of *all* its terms, so a troubleshooting-table chunk
carrying several distinct error codes dilutes each individual code's contribution. Generic
words that recur in every table across all six articles — "account", "customer", "fix" —
then contribute similarly across many chunks regardless of which one actually answers the
question, and a document with several of those common terms can out-rank the document with
the one rare, decisive term. BM25's length penalty (`b=0.75`) is far gentler, and its score
is a plain sum of idf-weighted, frequency-saturating contributions — a term that appears in
the corpus exactly once (the error code) dominates that sum regardless of how many other
terms the chunk carries. Fusing the two rankings by rank (RRF) rather than raw score is what
lets BM25's win on these queries pull the chunk back into the top-3 without needing the two
scales to be comparable.

## 6. `rrf_k`: swept, and honestly reported as not load-bearing here

RRF's damping constant was swept from 1 to 5000 against the full question set. hit-rate@3
did not move at all across that range — 11/18 at every value tested, `results/week4_evaluation.json`
records the run this claim was checked against. `rrf_k=10` ships anyway, because it is
proportional to this corpus's size (46 chunks) rather than the web-search literature default
of 60 tuned for corpora of thousands of documents, but the honest finding is that the two
rankers already agree closely enough in relative order on this question set that the exact
damping constant does not decide any outcome reported here. A corpus where the two rankers
disagreed more sharply would very likely make this a real tuning decision; this one did not.

## 7. What the change did NOT fix, and why that is the right outcome

**`semantic_gap` (G1-G4): still failing under hybrid, by design.** These four questions
paraphrase their way around the corpus's own vocabulary entirely — "the tax ID doesn't match
because of a merger or rename" instead of naming ERR-4118, "nothing happened when the
customer tried to migrate early" instead of naming ERR-4006. A keyword ranker has no exact
term to find any more than the semantic ranker does; hybrid search cannot recover a failure
that was never about keywords. G3 moved from rank 10 to rank 8 under hybrid — still a miss at
k=3, and a reminder that "fixed" and "improved" are not the same claim.

**`generation_gate` (N1, N2): retrieval was never the problem.** For both questions the
correct chunk is retrieved at rank 1-2 under *either* mode — checked and confirmed in
`tests/test_hybrid_retrieval.py`'s `test_generation_gate_questions_are_not_fixed_by_a_retrieval_change`.
The pipeline still refuses, because the evidence gate's idf-weighted coverage check (`config.yaml`'s
`grounding.min_evidence_coverage: 0.55`, unchanged from Week 3) sees colloquial words absent
from this small corpus — "wants", "'ll", "stuck", "deal" — score maximum idf for being absent
and swamp the one real signal, the same mechanism `results.md` section 9 diagnosed for
question B1. Changing the retriever cannot touch this: the failure lives entirely downstream
of retrieval, in the gate. This is the module's central claim made concrete with two
worked examples: the fix that helps K1-K12 is provably irrelevant to N1 and N2, because they
are not the same kind of wrong.

**Three `keyword_dilution` questions (K3, K11, K12) remain misses even under hybrid.** BM25
moves K11 from rank 6 to rank 4 and leaves K3 and K12 unmoved — an improvement in two of three
cases that still isn't enough to cross into the top-3. `results_week4.md` section 3 lists
these with their exact ranks; they are reported, not hidden, because "did they notice which
failures their change did NOT fix" is exactly what a partial before/after number is for.

## 8. Limitations worth knowing before trusting these numbers

- **18 questions is a small, illustrative set**, built the same way B1/B2 were — caught
  failing empirically, not sampled at a size that supports a statistically powered claim.
- **`recall@3` came out lower than `hit-rate@3`** (0.065 -> 0.593 vs 0.111 -> 0.611), which
  looks backwards until the reason is stated plainly: three questions (`G3`, `N1`, `N2`) turned
  out to have their `must_contain` phrase — "migration credit", "2026-09-30", "validation
  pending" — genuinely present in two or three different chunks of the same article, not one.
  Recall wants *all* of them in the top-3; hit-rate only ever wanted one. That is a real,
  disclosed property of these three questions' phrasing, not a bug in the metric, and it is
  exactly the kind of divergence between hit-rate and recall the module asks us to notice.
- **The evidence-gate threshold is unchanged from Week 3** and was calibrated on Week 3's own
  11-question set. N1/N2 are direct evidence it does not generalise to a support agent's
  actual colloquial register — a limitation carried over, not introduced, this week.
- **Ships as the new default.** `config.yaml`'s `retrieval.mode` is now `hybrid`, and
  `tests/test_hybrid_retrieval.py::HybridRetrieverEndToEndTests.test_hybrid_does_not_regress_the_week3_known_answer_questions`
  confirms all eight of Week 3's committed questions still hit at k=5 under it — the same
  standard Week 3 held `structure-aware` to before it replaced `baseline` as the shipped
  default.
