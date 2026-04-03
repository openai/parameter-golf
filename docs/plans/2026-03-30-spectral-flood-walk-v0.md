# Spectral Flood Walk LM — V0 Retrieval Eval Spec

## Summary

`v0` is the first retrieval-only version of the Spectral Flood Walk LM. Its purpose is not to max out eval-time state yet. Its purpose is to answer one narrow question cleanly:

> Does a fixed, training-derived retrieval pool improve prediction on unseen validation text when the model is trained to query and interpret that pool?

`v0` deliberately excludes:

- online eval-time memory growth
- eval-time TTT
- approximate search
- cap-driven austerity around auxiliary eval-time state

If `v0` works even modestly, `v1` becomes much more interesting because the pool can then start accumulating entries from the actual validation stream.

## Rules And Machine Envelope

We optimize for the current published challenge rules:

- `16MB` artifact
- `600s` train
- `600s` eval
- `8x H100 SXM`

Relevant references:

- [README.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/README.md#L6)
- [README.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/README.md#L128)
- [README.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/README.md#L184)

Representative validation size from an accepted run:

- `62,021,632` validation tokens: [train.log](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/train.log#L1373)

Current stance on the proposed auxiliary eval-state cap:

- Ignore it for `v0`
- [PR #886](https://github.com/openai/parameter-golf/pull/886) is still an open RFC, not an adopted rule

## Design Principles

1. Retrieval must be exact in `v0`.
2. Every prediction must be a full normalized distribution over the whole vocabulary.
3. The learned artifact stays tiny; the runtime retrieval path does the heavy lifting.
4. The pool entry should be rich enough that the decoder can interpret retrieved context at read time.
5. `v0` should isolate whether retrieval helps, without performance confounds from approximate search or giant online memory.

## What V0 Is Testing

`v0` is explicitly testing whether training-derived memory transfers:

- The pool is derived from training data only.
- The pool is fixed at eval start.
- The model is trained with retrieval in the loop so eval is not relying on a train/eval mismatch.

If retrieval helps in this setting, then `v1` can add online append-only memory and test whether same-stream writes amplify the gain.

## Architecture Overview

The `v0` system has four learned components:

1. A small bank of recurrent experts.
2. A gating path that mixes the expert states.
3. A query projection used for retrieval.
4. A read decoder that cross-attends over retrieved runtime states and emits full-vocab logits.

The recurrent experts are not generic GRUs. `v0` uses a scan-friendly affine recurrence so the training path can still benefit from parallel-scan style implementations:

```python
# token-conditioned, state-linear expert step
a_t = sigmoid(W_a @ x_t + b_a)                 # [128]
g_t = W_g @ x_t + b_g                          # [r]
u_t = W_u @ x_t + b_u                          # [128]

s_{t+1} = a_t * s_t + u_t + U @ (g_t * (V.T @ s_t))
```

Where:

- `a_t` and `g_t` depend on the token/input path, not on the previous state
- the transition remains affine in `s_t`
- `U @ (g_t * (V.T @ s_t))` is the low-rank "spectral flood walk" update

This is the concrete recurrence family for `v0` unless experiments refute it.

High-level eval loop:

```python
h_i = [expert_i(chunk_prefix, h_prev_i) for i in range(8)]
g = softmax(gate(recent_context))
h = sum(g[i] * proj_i(h_i[i]) for i in range(8))      # [512]

q = quantize_int8(norm(W_q @ h))                      # [128]

scores_local = q @ keys_local.T                       # exact local scan
topk_scores, topk_idx = scores_local.topk(64)

cands = all_gather_topk(topk_scores, topk_idx)        # small exchange only
nbr_states, nbr_meta = fetch_states(cands)            # 64 runtime states

ctx = read_decoder(h, nbr_states, nbr_meta)           # contextual interpretation
logits = lm_head(ctx)                                 # full vocab logits
probs = softmax(logits, dim=-1)                       # sums to 1 exactly
score(chunk_targets, probs)
```

## Locked V0 Choices

These are the `v0` defaults unless experiments immediately refute them.

| Question | Locked choice | Reason |
|---|---|---|
| Chunk size | `8` tokens | Small enough to keep retrieval frequent, large enough to keep controller stable |
| Neighbors | `64` | Good default, cheap at `v0` scale |
| Retrieval | Exact brute-force scan | Search cost is effectively zero at `v0` pool size |
| Read decoder | Cross-attention first | Simplest decoder that can interpret retrieved state contextually |
| Payload dtype | `FP8` runtime state | Better dynamic range than `int8`, native H100 support |
| Seed format | Latent codes + learned expansion basis | Necessary to fit the artifact budget |
| Expert specialization | Purely learned | Better than hand-designed roles in `v0` |
| Metadata | `expert_id` + position only | Enough to start, no speculative extras |
| Training bank size | Match eval seed-pool size | Small enough to simulate directly during training |

## Controller And Query Shapes

Starting shapes:

- experts: `8`
- per-expert hidden size: `128`
- fused controller state: `512`
- query/key dimension: `128`
- query/key dtype at retrieval time: `int8`
- retrieved neighbors: `64`

These numbers are intentionally modest. `v0` is a retrieval-interface test, not a final scaling statement.

Default low-rank recurrence shape per expert:

- hidden size: `128`
- low-rank update rank: `8`

This keeps the experts cheap enough that retrieval remains the dominant novel part of the model, not the recurrence itself.

## Artifact Budget

Initial target split:

| Component | Budget |
|---|---:|
| Tied token embedding / LM head | `2.0MB` |
| 8 recurrent experts + gate | `2.5MB` |
| Query / read / write projections | `2.0MB` |
| Read decoder | `3.5MB` |
| Payload expansion basis | `2.0MB` |
| Seed codes + metadata | `4.0MB` |
| **Total** | **`16.0MB`** |

This is a planning budget, not a final byte-exact layout.

## Seed Pool Compression Math

This was the main correction to the earlier draft.

If we say:

- `64K` seed entries
- `8KB` runtime payload per entry

then the expanded pool is:

```text
64K * 8KB = 512MB
```

That clearly does **not** fit into `4MB` of artifact budget via plain quantization or zstd.

So the honest `v0` design is:

- artifact stores **compact latent seed codes**
- eval startup uses a learned basis to expand those codes into runtime payloads

For example:

- `64B` code per entry
- `64K` entries

gives:

```text
64K * 64B = 4MB
```

which fits the seed-code budget.

The learned expansion basis then maps those compact codes into `8KB` runtime states at eval start.

That means:

- `v0` seed entries are **decoded runtime states**
- `v1` online entries will be the first version that truly stores direct raw state written during eval

This distinction matters for training. The model must learn that:

```python
seed_code -> expansion_basis -> runtime_state -> read_decoder -> logits
```

is a useful pipeline.

## Runtime Seed Pool

At eval start:

1. Load the artifact.
2. Decode the seed codes into runtime payloads.
3. Shard the decoded keys and states across 8 GPUs.

Initial `v0` scale:

- `64K-128K` seed entries total across the box
- roughly `0.5GB-1.0GB` expanded pool total, depending on exact entry count and payload width

This is intentionally small enough that scan cost is effectively negligible, which keeps `v0` focused on whether retrieval helps at all.

## Why Exact Scan Is The Default

At `128K` entries with `128B` keys, the total key set is only:

```text
128K * 128B = 16MB
```

Scanning `16MB` is effectively free on H100-class bandwidth. That means `v0` should not pay any of the engineering or correctness complexity of approximate search.

This is one of the key simplifications of `v0`:

- exact scan
- fixed pool
- small enough to be obviously affordable

If retrieval fails here, the failure is about the model and the memory interface, not the systems layer.

## Read Decoder

The `v0` read path should use standard cross-attention over the retrieved runtime states.

Reasoning:

- simpler than a Perceiver-style latent reader
- expressive enough to let the controller interpret retrieved states contextually
- keeps the design legible while we test whether the retrieval signal exists at all

Minimal form:

```python
def read_decoder(h, nbr_states, nbr_meta):
    q = W_attn_q(h)                  # [d]
    k = W_attn_k(nbr_states)         # [64, d]
    v = W_attn_v(nbr_states)         # [64, d]
    a = softmax((q @ k.T) / sqrt(d))
    read = a @ v
    return mlp(torch.cat([h, read, meta_embed(nbr_meta)], dim=-1))
```

The resulting `ctx` then flows through the LM head and a single full-vocab softmax.

## Probability Correctness

This is non-negotiable.

Retrieval does **not** produce sparse token votes or only compute the probability of the correct token. Instead, retrieval influences a contextual state, which is then decoded into full-vocab logits.

Canonical form:

```python
ctx = read_decoder(h, nbr_states, nbr_meta)
logits = lm_head(ctx)                  # [vocab]
probs = softmax(logits, dim=-1)        # sums to 1
loss = -log(probs[target])
```

This avoids the exact failure mode that invalidated the broken n-gram/cache line.

## Training Implications

`v0` should train with retrieval in the loop from day one.

Training should match eval on:

- controller interface
- query/key path
- seed-pool size
- read decoder
- full-vocab normalized decode

The most important extra objective introduced by the seed-code design is that the expansion basis must preserve what the reader needs.

The simplest first version is:

```python
L = L_nll + lambda_recon * ||expand(code(h_seed)) - stopgrad(h_seed_target)||^2
```

Possible variants:

- reconstruct controller hidden state directly
- reconstruct read-space features instead of raw hidden state
- reconstruct only what improves downstream NLL through the read decoder

The exact reconstruction target remains open, but the need for some consistency signal is real.

## Non-Goals For V0

`v0` intentionally does not try to solve:

- giant eval-time pool growth
- `40-60GB/GPU` state occupancy
- variable payload width
- append-only online memory
- TTT
- cap-aware eviction or retention policies

Those belong to `v1` and later.

## Success Criteria

`v0` is a success if it does all of the following:

1. Produces a fully normalized distribution at every scored position.
2. Runs an exact fixed-pool retrieval path end-to-end on GPU.
3. Shows that retrieval helps or at least plausibly helps relative to controller-only decoding.
4. Demonstrates that training-derived retrieval state transfers to validation text well enough to justify `v1`.

If retrieval adds only noise, `v0` has still succeeded scientifically because it tells us the fixed-seed path is not enough by itself.

## Open Questions

These remain genuinely open after the current `v0` spec.

### 1. What should the expansion basis reconstruct?

Options:

- controller hidden state
- read-decoder input state
- a separate learned retrieval state

This is probably the most important unresolved modeling question in `v0`.

### 2. How many seed entries are actually useful?

The planned `64K-128K` range is a good starting point, but the effective number may be smaller if training-derived retrieval does not transfer strongly enough.

### 3. Does `64` neighbors beat `32` or `128` in practice?

`64` is a good default, but the real winner depends on how redundant the seed pool is and how much diversity the read decoder can exploit.

### 4. How small can the seed codes get before the read signal collapses?

`64B` per entry is the planning assumption because it fits the budget cleanly, but the useful code width might need to be larger or smaller.

### 5. Is simple cross-attention enough?

Cross-attention is the right `v0` default. If retrieval helps but underwhelms, the next suspect is the reader.

### 6. How much does expert identity matter?

We are storing `expert_id` from day one, but `v0` still needs to show whether the read decoder actually uses expert provenance or mostly ignores it.

## Expected Follow-On

If `v0` works, `v1` should add:

- append-only online writes after scoring
- direct runtime raw-state entries
- larger runtime pool growth across validation
- eventually base-plus-delta variable payloads

That is where the architecture starts exploiting the full eval-time hardware envelope rather than just proving the retrieval interface.
