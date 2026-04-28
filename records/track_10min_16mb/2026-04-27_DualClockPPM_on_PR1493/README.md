# DualClock PPM: global + doc-local PPM mixture on PR #1493 stack

val_bpb: TBD | 8xH100 SXM | brotli artifact ~16 MB

## Single change vs PR #1493

Two causal PPM/Dirichlet-backoff experts are constructed at evaluation time over
already-scored validation tokens:

1. a **global** PPM expert (orders 1..K_g, hash bucketed), accumulating over the whole
   validation stream;
2. a **document-local** PPM expert (orders 1..K_l, hash bucketed), reset at chunk
   boundaries as a doc-boundary proxy.

Each expert is a normalized distribution over the full SP8192 vocabulary built by the
recursive smoothing chain `q_k(a) = (cnt_k(a) + alpha * q_{k-1}(a)) / (tot_k + alpha)`,
with `q_0` a causal running unigram prior. The neural model is a third expert. Final
per-token probability is a fixed-share Bayesian mixture; weights update only after the
chunk is fully scored.

## Distributions

```
q_0(a) = (uni_count(a) + eps) / (total + eps * V)             # unigram base prior
q_k(a) = (cnt_k(a) + alpha * q_{k-1}(a)) / (tot_k + alpha)    # PPM smoothing chain
p(a)   = w_N * p_N(a) + w_G * q_g^{K_g}(a) + w_L * q_l^{K_l}(a)
```

Hashes are FNV-style rolling over the last k tokens with order-mix; collisions yield a
coarser context state. The mixture weights are updated per chunk via a fixed-share
Bayesian rule on chunk log-loss:

```
post_i  = w_i * exp(-(L_i - L_min))
post   /= sum(post)
w_new   = (1 - share) * post + share * prior
```

## Theoretical motivation

Standard Bayesian mixing of K experts has cumulative log-loss within `log K` of the
single best fixed expert. Fixed-share extends this to track a switching expert across
chunks, which is exactly what is needed when the best expert changes across documents
or phases. Tiny LMs under 16 MB systematically miss exact-phrase repetition, URLs,
boilerplate, and local syntactic regularity that PPM captures cheaply.

## Compliance

- Each chunk is scored under `torch.no_grad()` strictly before any TTT update on that chunk
- Within a chunk, prefix-rank counts (`_dc_prank`) ensure a token never contributes to
  its own predicted probability
- Mixture weights for chunk c+1 are a function only of chunks 0..c losses
- All three statistical experts produce normalized probabilities over the SP8192 vocabulary
- The neural TTT update path is unchanged from PR #1493

## Hyperparameters

| Variable | Default |
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

## Base credit

Built on PR #1493 by clarkkev / dexhunter / abaybektursun / Robby955 / msisovic / X-Abhishek-X.
