# Spectral Flood Walk LM — V1b Episodic Memory Spec

## Summary

`V1b` keeps the `V1a` controller and changes only one thing:

> after each scored validation chunk, write a dumb episodic entry; then let a bounded fixed-function refiner turn those entries into more reusable summaries.

This is the first stage that directly tests the new thesis:

- writes should be cheap and naive
- refinement should happen later
- the controller should read from both raw memory and refined summaries

## What V1b Is Trying To Answer

`V1a` answered:

> does static semantic memory help enough to justify itself?

`V1b` answers:

> does same-stream append-only episodic memory help when we stop asking the write path to be perfect?

And more specifically:

1. Is raw append-only memory useful at all?
2. Does fixed-function refinement help beyond raw append?
3. Are the gains, if any, large enough to justify a later learned coprocessor?

## Core Decomposition

### Entity 1 — Controller

The controller is the same small transformer used in the `V1a` baseline:

- causal transformer
- tied embedding / LM head
- no special retrieval loss
- no write-optimized latent

The controller only does two extra things during evaluation:

```text
1. emits a pooled hidden state used as a query
2. emits that same pooled state as the raw write payload after scoring
```

### Entity 2 — Raw Episodic Pool

After each scored batch/window:

```text
pooled_hidden -> append raw key/value entry
```

Each raw entry stores:

- normalized key
- normalized value
- surprise score from the scored batch
- retrieval hit count
- age
- bucket id

The raw pool is append-only within the eval pass unless capacity is exceeded.

### Entity 3 — Fixed-Function Refiner

The first refiner is deliberately not learned.

It runs on dirty buckets only and does two things:

1. chooses the most important raw entries in the bucket by:

```text
priority = 1 + hit_count + surprise_weight * surprise
```

2. builds a small number of summary entries by greedy similarity clustering:

```text
if cosine(key_i, key_j) >= merge_threshold:
    treat them as one cluster
```

Each cluster becomes one summary key/value pair.

The controller never writes summaries directly. Summaries are purely a refinement product.

## Retrieval Path

For each eval batch:

1. compute pooled query from the controller hidden state
2. hash query to one local bucket
3. retrieve from:
   - raw entries in that bucket
   - summary entries in that bucket, if refinement is enabled
4. take top-`k` by exact similarity
5. form a weighted memory context
6. fuse that context back into the hidden stream before decoding logits

The current fusion rule is intentionally simple:

```python
fused_hidden = normalize(hidden + alpha * retrieved_context.unsqueeze(1))
```

This is not meant to be final. It is the smallest honest test of whether the episodic path helps at all.

## Scheduling

`V1b` uses bounded maintenance instead of true asynchronous overlap.

### Score-before-update

For each eval batch:

1. read from the currently committed raw/summarized pool
2. score the batch
3. update retrieval hit counts
4. append new raw entries
5. optionally run maintenance if the schedule fires

This preserves the README rule that validation tokens may only affect future predictions after they have already been graded.

### Maintenance Schedule

The fixed-function refiner uses:

- `maintenance_every`: how often to run maintenance
- `maintenance_budget_buckets`: maximum dirty buckets processed per maintenance pass
- `maintenance_source_limit`: max raw entries considered when building summaries for one bucket

The important property is that maintenance is bounded. It can never silently eat the whole eval budget.

## Current Implementation Defaults

The exploratory defaults in [spectral_flood_walk_v1b.py](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/spectral_flood_walk_v1b.py) are:

- `episodic_bucket_count=512`
- `episodic_max_entries=65536`
- `episodic_topk=16`
- `episodic_read_alpha=0.20`
- `maintenance_every=16`
- `maintenance_budget_buckets=16`
- `maintenance_source_limit=64`
- `summary_per_bucket=4`
- `merge_similarity=0.94`

These are not tuned. They are intended to make the first pod run answer the right architectural question.

## Evaluation Modes

`V1b` runs three eval modes from one trained controller:

### `controller`

No episodic memory at all.

This is the control.

### `raw`

Append-only raw episodic memory.

No summaries.

This answers:

> does dumb memory help before any refinement?

### `refined`

Append-only raw memory plus fixed-function bucket summaries.

This answers:

> does post-write refinement help more than raw append alone?

## What Counts As Success

The immediate success criterion is not leaderboard quality. It is directional clarity.

Good outcomes:

- `raw` beats `controller`
- or `refined` beats both `controller` and `raw`

Especially strong:

- `refined` > `raw` > `controller`

That would justify a later learned coprocessor.

Bad outcomes:

- `raw` and `refined` both lose to `controller`
- `refined` is no better than `raw`
- refinement cost grows but summary count/hit usage stays low

That would mean the “write dumb, refine smart” thesis still needs a different memory object.

## Metrics To Log

Per run, `result.json` should include:

- controller `val_bpb`
- raw-memory `val_bpb`
- refined-memory `val_bpb`
- deltas vs controller
- train throughput
- eval throughput per mode
- raw pool MB estimate
- summary pool MB estimate
- average raw candidates per query
- average summary candidates per query
- average hit mass per query
- refined bucket count
- total maintenance time

## Open Questions

1. Is pooled hidden state the right raw write payload, or should the write value be a different projection?
2. Is bucket-local retrieval enough to show signal before more elaborate routing?
3. Is greedy clustering plus summary prototypes the right first refiner, or should the next step be a tiny learned local MLP?
4. Does refinement help because summaries are good, or only because they change retrieval density?
