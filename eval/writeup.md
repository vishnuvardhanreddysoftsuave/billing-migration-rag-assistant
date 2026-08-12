## 8. Which chunker ships, and why

**The structure-aware chunker ships.** On the metric committed before the run it wins
nothing: both strategies score 8/8 hit-in-top-5, at every chunk size from 400 to 1600
characters. The case for it rests entirely on the second measurement — whether the
retrieved chunk is usable once you have it. There, `baseline` scores 3/8 self-contained
against 8/8 for `structure-aware`, and 0/4 against 4/4 on the four questions whose answer
is a troubleshooting table row: in five of eight cases the baseline chunker retrieved the
right row without the header row that says which cell is the cause and which is the fix.
That difference is visible downstream in the answers themselves — with the header present
the ERR-4032 answer reads `Error code: ERR-4032; Cause: …; Fix: …`, and without it the
same answer degrades to a bare `ERR-4032 | … | …` that a support agent has to guess at.
It is also the more robust choice under a knob nobody tunes carefully in practice: across
the sweep, `structure-aware` holds 8/8 self-contained at 400, 800, 1200 and 1600
characters, while `baseline` wanders between 3/8 and 6/8 and only recovers at 1600 because
the chunk grew big enough to swallow an entire table by accident. The costs are real and
worth stating: at larger chunk sizes it produces more chunks than the baseline (37 vs 24
at 1600) because it refuses to merge across section boundaries, its chunks are smaller on
average (651 vs 1318 chars at 1600), and prefixing every chunk with its heading breadcrumb
puts heading words into the embedding — which, as §9 shows, can actively hurt.

## 9. The retrieval that embarrassed us

The bonus question **B1** — *"A customer hit ERR-4032. What do I tell them to do, and what
will they need to hand?"* — is the same question as Q1 in a support agent's register, and
both chunkers ranked the wrong chunk first. `structure-aware` returned
`HC-4001#structure-aware-0008`, a section titled **"What to tell customers"** that is about
migration comms and does not mention ERR-4032 at all. The ERR-4032 row chunk came third.

The diagnosis is not "semantic search is fuzzy". Measured per token, the query's six
content terms carry these weights: `hit` 4.85, `hand` 4.85, `tell` 4.16, `err-4032` 3.75,
`need` 3.24, `customer` 2.21. The single decisive term is *outweighed by three colloquial
ones*, two of which (`hit`, `hand`) the corpus never uses and therefore score maximum idf
for being absent — the same property that makes the refusal gate work correctly turns into
noise inside the ranking function. `tell` and `customer` then both match a section whose
heading is literally "What to tell customers". Our own breadcrumb prefix made it worse:
that chunk scores 0.0718 with the breadcrumb and 0.0557 without it, a 29% self-inflicted
boost from repeating "What to tell customers" in the text we embed. Length normalisation
finished the job — the correct 796-character table chunk (0.0453) is penalised against a
403-character one for being longer.

Two consequences we did not paper over. First, the same colloquial phrasing pushes the
evidence gate below its threshold, so B1 is **refused outright** at the shipping setting —
a false refusal on a question the corpus answers perfectly well. We would rather ship that
failure than its opposite, but it is a failure. Second, it is why B2 exists: the same
question in neutral phrasing retrieves the right row at rank 1. The honest summary is that
this pipeline is sensitive to question register, and nothing in the 8-question set
measured that, because we wrote all eight in the same neutral voice.

The bonus challenge itself is only half-demonstrated, and we will say so plainly. B2 with
a single-chunk context shows the precision half: `structure-aware` retrieves the ERR-4032
row precisely and answers the *fix* correctly, but cannot say the customer needs the
physical card in hand, because that sentence lives in the "Re-authorisation walkthrough"
section two chunks away — precise retrieval, incomplete answer. What it does not show is
`baseline` producing a *better* answer to compare against: its top-1 chunk for B2 does not
contain ERR-4032 at all, so it refuses. At the shipping top_k of 5 both chunks are
retrieved and both strategies answer completely, which is the practical mitigation — the
tension is real at small context sizes and disappears when you give the generator room.

## 10. Limitations worth knowing before trusting these numbers

- **The primary metric saturated.** Eight questions over six articles is a small set, and
  hit@5 hit its ceiling immediately. The number that moved is the secondary one, added
  after seeing that saturation and reported as such.
- **The refusal threshold is calibrated on the same 11 questions it is evaluated on.**
  The answerable questions score 0.63–1.00 on idf-weighted coverage and the out-of-corpus
  ones 0.15–0.46, so 0.55 separates them with margin — but a threshold fitted on the
  evaluation set is not evidence that it generalises. B1 is the counter-example that it
  does not always.
- **Generation ran on the deterministic extractive backend**, because no Anthropic
  credentials were present in this environment. The Claude path (`claude-opus-5`) is
  implemented and selected automatically when credentials exist; the citation validation,
  the refusal gate and the evidence thresholds are backend-independent, but the answer
  *prose* in §4 is extractive rather than model-written.
