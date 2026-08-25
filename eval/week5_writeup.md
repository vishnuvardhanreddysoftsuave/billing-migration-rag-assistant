## Analysis

### What reading 20 real traces found that the dashboards did not

Week 3 and Week 4 both scored the app on question sets built to demonstrate a specific,
already-suspected mechanism (does a chunker preserve a table row; does hybrid search beat
semantic-only on keyword-diluted queries). That is the right tool for proving a fix works, but
it cannot surface a problem nobody has hypothesised yet, because every question in those sets
was chosen *because* it was expected to behave a particular way.

This week's sample was built to avoid that: `eval/week5_questions.yaml` treats "every
troubleshooting-table error code across the whole corpus, in a rotation of five realistic
support-ticket phrasings, plus prose facts, colloquial paraphrases, out-of-corpus questions and
deliberately hard cases" as the traffic distribution, and draws 20 of those 84 candidates with
`random.Random(5).sample(...)` — a seed fixed before the draw, not re-rolled after seeing the
result. What came back was not the failure mode either of the first two weeks anticipated.

### The headline: refusal rate the corpus does not justify

7 of the 20 sampled questions were answered; 13 were refused. Only 2 of those 13 refusals
(`O2`, `O8`) were on questions the corpus genuinely cannot answer — the pool's overall
out-of-corpus share is 8/84 (~10%), so a random sample of 20 would be expected to draw roughly
2 unanswerable questions, which is exactly what happened. The other 11 refusals were on
questions the corpus *does* answer: 9 straightforward error-code lookups (`E-4002`, `E-4004`,
`E-4007`, `E-4035`, `E-4036`, `E-4088`, `E-4155`, `E-4203`, `E-4206`), the two-part `M2`, and —
worth calling out on its own — `C8`, which did not refuse but answered confidently from the
wrong article entirely.

Reading the traces individually, not just the pass/fail count, is what made the mechanism
visible. In `E-4004`, `E-4036`, and `E-4206` the correct chunk is retrieved at rank 1 — this is
not a retrieval problem at all — and the pipeline refuses anyway. The evidence gate's
refusal reason names the culprit directly every time: `missing_terms` is never a real gap in
the corpus, it is a handful of ordinary conversational words the question happened to use —
*getting, idea, trying, info, going, hey, mentions, resolution, throwing, root, remediation,
steps, quickly*. None of those are in `grounding.py`'s `QUESTION_WORDS` stoplist, none of them
occur anywhere in this ~46-chunk corpus, so each one is assigned the *maximum* idf weight the
corpus can produce — and a short question built mostly out of boilerplate phrasing can lose
more than half its idf-weighted mass to words that were never meant to carry information in the
first place.

The five support-ticket templates split cleanly on this axis. Every question built from
*"{code} — can you tell me what this means and what I should do?"* passed the gate (`E-4151`,
`E-4156`, `E-4205`, 3/3) — not because it is any more informative than the others, but because
its filler words ("tell", "mean", "need") happen to already be on the stoplist. Every question
built from the other four templates failed (9/9): *"Support ticket: account throwing X..."*,
*"Customer says they got X when trying to update..."*, *"Getting X ... any idea why..."*, and
*"Hey, a customer's ticket mentions X..."*. That is a coincidence of which words a hand-written
stoplist happened to include, not a real difference in how answerable the underlying question
is — and it means the gate's behaviour on any new phrasing is close to unpredictable.

This also lands directly on a specific, checkable claim from Week 4: `E-4088` and `E-4203`
use the *exact* "Support ticket: account throwing X, need root cause and remediation steps
quickly" phrasing that `eval/week4_questions.yaml`'s K-series used to prove hybrid search fixes
retrieval on this corpus. It still does — the answer-bearing chunk is in the top-5 for both.
But Week 4's harness measured that "before" state under semantic-only retrieval and never ran
the resulting hybrid-search "after" state through the full pipeline for this template, so it
never noticed that a second, independent gate — the evidence gate, not the retriever — refuses
this exact phrasing regardless of which retrieval mode found the right chunk. Week 4's fix was
real and correctly measured on its own terms; it just was not the last thing standing between
this template and a working answer.

### A second, rarer, more dangerous pattern

`C8` asked a colloquial paraphrase of the currency-conversion question (the underlying fact is
`ERR-4093`; the pool's direct, formally-phrased version of the same fact, `E-4093`, exists but
was not drawn into this particular sample). Retrieval drifted to the legacy invoice article (`HC-1007`),
which shares generic "billing period" vocabulary with the question but has nothing to do with
currency. The evidence gate passed it (57% coverage — enough generic overlap to clear the bar),
and the extractive generator, scoring sentences by raw term overlap, picked three grounded,
correctly-cited sentences about invoice-period anchoring and presented them as the answer. The
result is indistinguishable in form from a correct answer: full citations, resolved chunk ids,
no refusal — and it is about the wrong thing. This happened once in 20 traces, far less often
than the refusal pattern, but it is the more dangerous failure mode of the two: a refusal costs
a customer a wasted message; a wrong-but-confident answer can send an agent down the wrong
remediation path without any signal to double-check it.

### Named problems, ranked

See the taxonomy table above (`results_week5.md` §2) for the counts; in order:

1. **False refusal on answerable questions (informal phrasing)** — 10/20 traces, severity 3,
   score 30. By far the largest and most severe group: it is the majority outcome for the
   error-code-lookup traffic this app almost certainly sees the most of in practice.
2. **Off-topic confident answer after retrieval drift** — 1/20, severity 3, score 3. Rare in
   this sample, but the single worst individual outcome observed, and worth flagging even
   though the frequency x severity score ranks it below problem 1 — a purely mechanical
   ranking cannot distinguish "rare and merely annoying" from "rare and actively misleading",
   and this is the latter.
3. **Ambiguous question refused instead of clarified** — 1/20, severity 2, score 2.
4. **Correct answer padded with a tangential citation** — 1/20, severity 1, score 1.

### The one problem to attack next: false refusals on informal phrasing

Frequency x severity puts this first, and it would still be first on judgment alone: it is
half of everything sampled, it is severe (the corpus has the answer and the customer does not
get it), and unlike problem 2 it is *systematic* rather than a one-off — the template split
(9/9 fail, 3/3 pass) shows this is not noise, it is a predictable property of which ordinary
words a phrasing happens to contain.

The mechanism is now specific enough to name a fix without guessing: `QUESTION_WORDS` in
`grounding.py` is a fixed, hand-written stoplist of interrogative scaffolding, and it was
calibrated against Week 3's eight formally-phrased questions, which never needed words like
"getting" or "trying" filtered out because they were not written in that register. It was
never re-checked against realistic support-ticket phrasing before this week.

**Prediction, to be verified empirically before it is shipped, not assumed:** widening the
filler-word handling in the evidence gate (either extending `QUESTION_WORDS` with the
conversational filler this sample actually surfaced, or making the coverage calculation
robust to a small number of unmatched terms rather than penalising every one at full idf
weight) should recover most or all of the 10 false refusals in this taxonomy group, without
reducing the true refusal rate on `O2`/`O8`-style out-of-corpus questions, and without
increasing `C8`-style wrong-topic answers (a coverage fix does not touch retrieval or
generation, so it should not be able to make problem 2 worse). The falsifiable check for next
week is the same one Week 4 used: re-run `python rag.py eval-errors` (or a fresh seeded sample)
before and after the change and report the before/after count on this exact taxonomy group,
plus a regression check that `eval/questions.yaml`'s three out-of-corpus questions are still
refused. This is deliberately left unimplemented this week — Week 5's task is the taxonomy and
the target, not the fix.
