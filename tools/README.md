# Stability tooling

Two scripts that answer two different questions. Run them in order — the first
is free and narrows what the second has to explain.

Neither writes to `aop_rag.db`.

## 1. `stability_check.py` — is the deterministic half deterministic?

Everything after extraction is supposed to be a pure function of the Table 1
row set: the same rows should give the same Table 2, the same counts and the
same synthesis prompt regardless of what order they arrive in. Row order is not
stable in practice, because `record_id` follows upload order.

```bash
python tools/stability_check.py --db aop_rag.db --permutations 30 \
    --edge "oligodendrocyte differentiation"
```

No model calls, no cost. Exit code 0 when every stage was order-independent.

**Example result on a test corpus:**

| stage | verdict |
|---|---|
| `table2_raw` | FAIL — 17 fields moved |
| `table2_normalized` | FAIL — 17 fields moved |
| `edge_counts` | PASS |
| `synthesis_input` | FAIL — prompt text differs |

Only one edge is affected: the six-row target edge. The 57 singleton chain
steps have nothing to aggregate, so they cannot move. The counts and the
confidence score are stable; what moves is the *text*, in three distinct ways:

- **REORDERED** (`all_taxa`, `study_designs`, `supporting_dois`,
  `sex_applicability`, `essentiality_evidence`, …) — `_join_unique` preserves
  encounter order, so the same content comes out in a different sequence.
- **TRUNCATED** (`chemical_stressors` `limit=8`, `measured_as` `limit=4`,
  `null_findings` `limit=4`, `empirical_evidence_summary` `limit=4`) — the
  limit keeps a *different subset* depending on which row was seen first.
  Content is silently lost, and which content is lost is arbitrary.
- **REPLACED** (`ker_description`, `study_contexts`) — `_first_non_null` takes
  whichever row happened to be first.

All three reach the synthesis prompt, so the narrative and the OECD ratings are
not reproducible even when the row set is identical.

## 2. `replicate_run.py` — how much does the model actually move?

Runs the two calls per paper that decide which edge a paper lands on
(`extract_pathway_rows`), k times, into a separate database. Skips the five
per-KER evidence steps: they change the prose inside a row, not whether the row
exists, and they are most of the cost.

```bash
export ANTHROPIC_API_KEY=...
python tools/replicate_run.py \
    --pdfs ./papers \
    --upstream "Voltage-gated sodium channel" \
    --downstream "Oligodendrocyte differentiation" \
    --provider anthropic --model claude-opus-5 \
    --temperature 0.1 --budget-scale 2.0 \
    --k 5 --out replicates.db

python tools/replicate_run.py --report-only --out replicates.db
```

N papers × 2 calls × 5 replicates. Resumable — a paper already
recorded for a replicate is skipped, so an interrupted run continues where it
stopped.

It pins two things that would otherwise be uncontrolled variables and records
their hashes: the parsed PDF text (parsed once, reused for every replicate) and
the expanded target vocabulary (built once). If either hash changes between
measurement sessions, the sessions are not comparable.

**What it reports**

- *Supporting papers per replicate* — the number your synthesis would have
  shown, once per replicate, as a list. This is the quantity that was moving.
- *Per-paper anchor rate* — for each paper, in how many of the k replicates it
  produced a step running directly from the upstream anchor to the downstream
  anchor. `5/5` and `0/5` are stable; anything between is where the variance
  lives.
- *Majority-vote count* — what the count would be if a paper counted when it
  anchored in more than half the replicates.
- *Agreement rate* — the fraction of papers that were unanimous. This is a
  reportable methods number, not an embarrassment.

## Why the anchor rate is the right thing to measure

In both existing runs the target edge is the **only** multi-paper edge; every
other edge is a single paper's chain step. So "how many papers support this
KER" reduces to one binary question per paper: did the model reconstruct that
paper's chain as `upstream → downstream` directly, or as
`upstream → [intermediate] → downstream`?

`_canonical_anchor` snaps an event onto the curator's wording only on an exact
normalised string match, which is correct — it must not merge
"decreased Nav channel subunit transcripts" into "Voltage-gated sodium channel".
The consequence is that a paper counts toward the target edge only when the
model emitted the anchor label verbatim as a chain endpoint. That is a single
stochastic decision per paper, and it is the whole story.
