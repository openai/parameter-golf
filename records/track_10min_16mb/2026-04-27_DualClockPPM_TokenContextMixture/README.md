# DualClock PPM: legal token-context mixture for evaluation

val_bpb: **1.0803** | seed 42 | 8xH100 SXM | brotli artifact 15978 KB

## Summary

A score-first, eval-time mixture between the neural model and two causal
Dirichlet-backoff PPM experts:

1. a **global** PPM expert that accumulates over the whole validation stream
2. a **document-local** PPM expert that resets at chunk boundaries

Per-token probability is a fixed-share Bayesian mixture over the three experts.
Final BPB on seed 42: **1.080326** (vs neural-only 1.080301 in the same run; the
mixer auto-collapses weights to the neural model when it dominates).

## Result

| component | val_bpb | eval_time |
|-----------|--------:|----------:|
| pre-quantization post-EMA | 1.087045 | 6.8s |
| quantized | 1.098230 | 8.6s |
| quantized + sliding window | 1.081633 | 91.1s |
| **quantized + TTT + DualClock mixture** | **1.080326** | 361.9s |

Final mixture weights converged: `[0.99950, 0.00035, 0.00015]` over (neural, global PPM, local PPM).

## Theory

### Each PPM expert is a normalized distribution over the token vocabulary

Let `q_0(a) = π(a)` be a causal running unigram prior. For order `k`,

```
q_k(a) = (c_k(a) + α · q_{k-1}(a)) / (N_k + α)
```

where `c_k(a)` is the continuation count for token `a` after the current
order-`k` context and `N_k = Σ_b c_k(b)`. By induction, `Σ_a q_k(a) = 1`.

### The mixture is normalized

For experts `q_i(a)` and weights `w_i ≥ 0, Σ w_i = 1`,
`p(a) = Σ_i w_i · q_i(a)` is a full distribution over the official token alphabet.

### Causality

At token position `t`, every expert reads only the strict prefix `x_1..x_{t-1}`.
After the score for `x_t` is recorded, counts and mixture weights update using `x_t`.
Within a chunk, prefix-rank counters ensure a token never contributes to its own
predicted probability.

### Switching-expert protection

Fixed-share Bayesian update across chunks:

```
post_i = w_i · exp(-(L_i - L_min))
post  /= Σ post
w_new  = (1 - share) · post + share · prior
```

This bounds the cumulative log-loss against the best switching expert across
chunks, which matters when the dominant expert changes across documents or
phases of a document.

## Implementation

At evaluation time, every chunk is scored under `torch.no_grad()` before any
TTT update on that chunk. As each chunk is scored, the per-token neural NLL
is gathered across DDP ranks via `all_reduce(MIN)` on a buffer initialized to
`+∞`, since each rank scores a disjoint window subset and the score buffer is
later consumed by a single deterministic mixture pass that runs on every rank.

Per chunk, we construct two causal PPM experts. Each expert maintains, for
every order `k = 1..K`, a hash-bucketed count table `cnt[k]` of shape `(B·V,)`
(int32) holding continuation counts and a context-mass table `tot[k]` of
shape `(B,)`. The order-`k` context for a target at chunk position `i` is the
FNV-rolling hash of the last `k` source tokens, computed in a vectorized loop
over `k`. The hash modulo `B` gives the bucket id; the resulting
`(ctx_hash, target_token)` pair indexes into `cnt[k]`. Predictions are
produced by the recursive Dirichlet-backoff smoothing chain
`q_k(a) = (cnt_k(a) + α·q_{k-1}(a)) / (tot_k + α)` where `q_0` is a causal
running unigram prior over the full SP8192 vocabulary, so each expert is by
construction a fully normalized distribution over the official token alphabet.

Within a chunk, the strictly causal contribution from already-seen positions
is enforced by a vectorized prefix-rank counter (`_dc_prank`): for the `n`
positions in the chunk it returns, for each position, the count of identical
earlier-position keys, which is the chunk-local correction that lets us score
the entire chunk in parallel without ever leaking a token's identity into
its own predicted probability. After the chunk's score is finalized, the
count tables and the unigram base prior are updated with that chunk's tokens
via GPU `index_add_`. The global expert never resets; the document-local
expert resets at chunk boundaries as a doc-boundary proxy.

The three experts (neural, global PPM, local PPM) are blended through a
fixed-share Bayesian mixer. Each chunk yields a per-expert mean log-loss
`L_i`; the chunk-end posterior is `post_i ∝ w_i · exp(-(L_i - L_min))`, then
`w_new = (1 - share) · post + share · prior` with prior `(0.90, 0.07, 0.03)`
and `share = 0.005`. This bounds cumulative log-loss against the best
switching expert across chunks while keeping a small floor of probability on
the slower-moving experts so they can re-engage if the dominant expert ever
changes. Mixture weights for chunk `c+1` are a deterministic function of
chunks `0..c`'s losses, which is the same legality property the neural TTT
update path relies on.

## Hyperparameters

| variable | default |
|----------|---------|
| DC_ENABLED | 1 |
| DC_ORDER_G | 6 |
| DC_ORDER_L | 8 |
| DC_BUCKETS_G | 2048 |
| DC_BUCKETS_L | 2048 |
| DC_ALPHA_G | 1.0 |
| DC_ALPHA_L | 0.5 |
| DC_EPS_UNI | 0.25 |
| DC_SHARE | 0.005 |
| DC_PRIOR | "0.90,0.07,0.03" |
| DC_DOC_RESET | 1 |

## Run

```bash
SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
DC_ENABLED=1 DC_ORDER_G=6 DC_ORDER_L=8 DC_BUCKETS_G=2048 DC_BUCKETS_L=2048 \
DC_ALPHA_G=1.0 DC_ALPHA_L=0.5 DC_SHARE=0.005 DC_PRIOR='0.90,0.07,0.03' DC_DOC_RESET=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## What this experiment shows

The Bayesian mixer over a strong TTT'd 36M-param neural model and two
hash-bucketed PPM experts converges to ~99.95% weight on the neural expert
within the first few chunks. The switching-expert guarantee provides safety,
but the small-vocab PPM experts at orders ≤ 8 with bucket size 2048 do not
hold information the neural model has not already absorbed. Net effect at
this configuration: indistinguishable from neural-only TTT (1.080326 vs
1.080301; well within run-to-run variance).
