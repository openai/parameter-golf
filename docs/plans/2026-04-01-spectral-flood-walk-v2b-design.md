# Spectral Flood Walk LM — V2b Persistent Latent Adaptation

## Summary

`V2b` starts from a different first principle than `v0`, `v1a`, `v1b`, or `V2a`.

The question is no longer:

> how do we use more VRAM?

The question is:

> if hardware were not the limiting factor, what inference-time learner would we actually want?

The answer is not "just a bigger cache" and not "just a per-batch optimizer state."

The ideal learner has four parts:

1. a strong base transformer with the best exact context we can afford
2. a **persistent latent memory** that survives across the whole validation stream
3. an optional **ephemeral batch delta** for fast local adaptation
4. score-first online updates so the system only learns from already-evaluated tokens

In compact form:

```python
h_t = base_hidden(prefix_t)
m_t = persistent_memory.lookup(context_t)
d_t = batch_delta.lookup_or_optimize(context_t)
logits_t = lm_head(h_t + m_t + d_t)
probs_t = softmax(logits_t)
```

and after scoring:

```python
persistent_memory.update(context_t, scored_error_t)
batch_delta.optimize(scored_error_t)
```

The defining identity of `V2b` is:

> persistent online latent adaptation across the validation stream.

That is what keeps it distinct from both:

- token-level n-gram expert schemes
- pure SLOT-style ephemeral per-batch optimization

## What V2b Is Trying To Answer

`V2b` asks one narrow question:

> does persistent hidden-space adaptation across the validation stream buy materially more `val_bpb` than a strong frozen host alone?

More specifically:

1. does persistent latent memory help beyond exact context and standard sliding-window eval?
2. is persistent memory stronger than a pure per-batch delta?
3. do persistent memory and a tiny ephemeral delta complement each other?

## Why V2a Was Not Enough

`V2a` taught us something important.

The logit-residual tables were:

- legal
- simple
- stable

but on the real `8xH100` host run the gain collapsed from the exploratory `~ -0.10` range to near zero.

That suggests the real opportunity is not:

- better token-context hash corrections

It is:

- better **latent adaptation geometry**

`V2b` is the response to that lesson.

## Distinction From SLOT

`V2b` should learn from SLOT without collapsing into "our SLOT clone."

The clean distinction is:

- `SLOT`: ephemeral latent optimization for the current scored batch
- `V2b`: **persistent keyed latent memory** that survives across batches and documents

If removing the current batch destroys the learned state, the mechanism is not distinctly ours.

If the model still benefits from structure learned on earlier scored text, it is.

So the design rule is:

> persistent stream memory is primary; ephemeral batch delta is optional.

The intended decomposition is:

```python
h_t = base_hidden(prefix_t)
m_t = persistent_memory.lookup(context_t)   # survives across stream
d_t = batch_delta                           # optional short-lived helper
logits_t = lm_head(h_t + m_t + d_t)
```

`m_t` is the main idea.
`d_t` is there only if it helps.

## Hardware-Unconstrained Ideal

If hardware were not the limiting factor, the ideal inference-time learner would look like:

```python
h_t = base_hidden(prefix_t, long_exact_context=True)

m_t = persistent_stream_memory.lookup(
    context=context_t,
    hidden=h_t,
    recent_errors=history_t,
)

d_t = local_optimizer_state.solve(
    current_batch=batch_t,
    current_errors=scored_errors_t,
)

logits_t = lm_head(h_t + m_t + d_t)
probs_t = softmax(logits_t)
```

and after scoring:

```python
persistent_stream_memory.write(context_t, h_t, scored_error_t)
persistent_stream_memory.refine()
local_optimizer_state.step(scored_error_t)
```

The important properties are:

1. exact recent context remains first-class
2. memory is latent, not token-frequency only
3. memory is persistent across the whole eval stream
4. the learner adapts after scoring, not before

That is the ideal we should approximate, not the exact implementation we must ship.

## Hardware-Aware Realization

The contest-shaped realization of that ideal should be:

### 1. Strong Host

Start from a serious host, not a toy.

Recommended primary host:

- a clean `4096`-vocab strong transformer in the style of the simplified high-performing stack
- no BigramHash if possible
- no speculative extras unless they already proved value

Recommended backup host:

- the proven `1.1233`-lineage `11L/512d` host we already grafted against

The reason to prefer the `4096`-vocab clean host is simple:

- it is stronger lexically
- it spends fewer tokens per document
- it leaves more room for online adaptation to do something distinctive
- it does not already contain a strong local-pattern competitor like BigramHash

### 2. Persistent Latent Memory

At eval time we maintain a keyed table of latent correction vectors:

```python
M: context_id -> delta_h
```

where:

- `delta_h in R^d_latent`
- `d_latent` starts small enough to be practical
- the table starts empty or near-empty and grows online

The simplest first realization is:

```python
h_t = base_hidden(prefix_t)
ctx_t = router(prefix_t)
delta_h_t = M.lookup(ctx_t)
logits_t = lm_head(h_t + delta_h_t)
```

This is already a full normalized distribution because:

```python
probs_t = softmax(lm_head(h_t + delta_h_t))
```

No target-token-only shortcuts are involved.

### 3. Optional Ephemeral Batch Delta

Only after persistent memory is working should `V2b` add a small batch-local delta:

```python
logits_t = lm_head(h_t + delta_h_t + d_batch)
```

This is the part that may look SLOT-like, but it is not the center of gravity.

The intended purpose is:

- catch immediate local structure within the current scored batch
- let persistent memory absorb what recurs across time

### 4. Exact Context Still Comes First

`V2b` does not replace exact context.

The base model should still use:

- strong sliding-window eval
- the best legal exact-context protocol the host supports

Persistent latent adaptation is layered on top of that.

## What The Persistent Memory Should Store

The stored object should be a latent correction that the base head already knows how to interpret:

```python
delta_h_t in R^d_model
```

or a lower-rank latent that is expanded before the LM head:

```python
z_t in R^r
delta_h_t = P @ z_t
```

The first version should prefer direct hidden-space deltas over logit-space deltas because:

- they are closer to the representation the host model actually reasons in
- they match the spirit of successful latent adaptation methods better
- they let the host head remain the final projection

## Routing

Routing exists only to make the persistent memory tractable.

It should not be oversold as semantic magic.

The first router should be simple:

- recent-token context hash
- possibly mixed with a coarse hidden-state signature
- fixed and cheap

For example:

```python
ctx_t = mix(
    rolling_hash(last_n_tokens),
    sign_hash(stopgrad(h_t))
) % table_size
```

The router's job is:

- give the memory a stable address
- spread writes and reads
- preserve some local context identity

It is not required to perfectly cluster meaning.

## Online Update Rule

After scoring, we can derive a hidden-space gradient signal from the LM head.

If:

```python
logits_t = W_out @ (h_t + delta_h_t)
```

then the cross-entropy gradient with respect to the adapted hidden state is:

```python
g_h = W_out.T @ (probs_t - one_hot(target_t))
```

That gives the natural persistent-memory update:

```python
delta_h[ctx_t] <- decay * delta_h[ctx_t] - eta * g_h
```

or an EMA-smoothed version:

```python
grad_ema[ctx_t] <- rho * grad_ema[ctx_t] + (1 - rho) * g_h
delta_h[ctx_t] <- decay * delta_h[ctx_t] - eta * grad_ema[ctx_t]
```

This is the central `V2b` rule.

It is:

- score-first
- fully online
- naturally tied to the host head

## Score-First Legality

`V2b` must preserve the same legality discipline as `V2a.1`.

Only scored positions may produce updates.
Only already-scored tokens may influence future predictions.

So the implementation rule is:

```python
score_mask = build_eval_score_mask(...)
g_h = hidden_grad_from_scored_positions(...)
memory.update(ctx_ids[score_mask], g_h[score_mask])
```

This should be enforced in code and explicitly verified in the README for any submission branch.

## Artifact Strategy

The artifact should stay focused on the host model.

Default plan:

- almost all bytes go to the host
- persistent latent tables start empty at eval time
- the artifact stores only small memory machinery:
  - router metadata
  - optional projection `P`
  - code

This follows the lesson from `V2a.1`:

- online adaptation was the only part with a real chance to matter
- static seed memory was not worth fighting the artifact budget for

So the first `V2b` artifact should be:

- host model
- tiny latent-memory scaffolding
- no large precomputed table payload

## Runtime Memory

If hardware were free, we would let persistent latent memory grow very large.

In the contest setting, we should still prefer useful memory over gratuitous memory fill.

The initial per-slot state might be:

```text
delta_h:      d_model * 2B or 4B
grad_ema:     d_model * 2B or 4B
count / age:  a few bytes
```

At `d_model = 512`, even fp16 hidden deltas are about `1KB` each, which is already much larger than the tiny `V2a` coefficient tables and therefore much more likely to use real VRAM if the table grows over time.

That is a feature, not a bug.

## Phases

### V2b-0 — Persistent Memory Only

No ephemeral batch delta.

Just:

```python
logits_t = lm_head(h_t + M.lookup(ctx_t))
```

Questions:

1. does persistent latent memory help at all?
2. how much of the gain survives on a strong host?
3. does the memory become meaningfully nonzero over the stream?

### V2b-1 — Persistent Memory + Tiny Batch Delta

Only if `V2b-0` is positive.

Add a very small batch-local free delta:

```python
logits_t = lm_head(h_t + M.lookup(ctx_t) + d_batch)
```

Questions:

1. is there a short-horizon gain that persistent memory misses?
2. does the combined system outperform either piece alone?

### V2b-2 — Memory Maintenance / Refinement

Only if the first two phases are promising.

This is where a maintenance process or learned refiner might enter:

- merge similar memory slots
- stabilize high-value slots
- compress stale slots

This is explicitly later work, not day-one scope.

## Host Matrix

The experiment matrix should stay small.

### Primary Host

A clean `4096`-vocab strong host with minimal competing local-pattern machinery.

Modes:

- host-only
- host + persistent memory
- host + persistent memory + tiny batch delta

### Backup Host

The `1.1233`-lineage host we already know how to graft against.

Modes:

- host-only
- host + persistent memory

The backup host is mainly there to check whether a result is host-specific.

## What Success Looks Like

Good outcomes:

- persistent memory alone beats the frozen host
- the gain is larger than the `~ -0.00016` we saw from `V2a.1` on the real `8x` host
- the gain survives longer runs and multiple seeds
- memory statistics show real accumulation instead of near-zero updates

Especially strong:

- the clean `4096`-vocab host leaves enough room for persistent memory to matter
- the combined persistent+batch system clearly outperforms either alone

Bad outcomes:

- persistent memory is flat on a strong host
- only the ephemeral batch delta helps
- updates stay too small to matter
- the host's own exact-context machinery already absorbs the whole benefit

## Recommended First Implementation

The first `V2b` patch set should be minimal:

1. choose the clean strong host
2. expose a hook at the hidden state right before the LM head
3. add a persistent memory table keyed by scored-position context
4. update that table with hidden-space gradients after scoring
5. compare:
   - host-only
   - host + persistent memory

Only if that is positive should we add the tiny batch delta.

## Bottom Line

`V2b` should be built from the hardware-unconstrained ideal first and only then trimmed into a contest-shaped implementation.

The ideal is:

> exact context + persistent latent memory + optional local optimization

The contest realization should preserve the most important of those pieces:

> persistent online latent adaptation across the validation stream.

That is the core idea worth carrying forward.
