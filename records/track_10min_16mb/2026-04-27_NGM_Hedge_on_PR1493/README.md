# NGM-Hedge: Normalized n-gram Hedge mixture on PR #1493 stack

val_bpb: TBD | 8xH100 SXM | brotli artifact ~16 MB

## Single change vs PR #1493

Three causal statistical experts (unigram, bigram, hash-trigram) are constructed at
evaluation time from already-scored validation tokens. Each expert is a normalized
distribution over the full SP8192 vocabulary. The neural model is a fourth expert.
Final per-token probability is an exponential-weighted mixture; the chunk weights
are updated only after the chunk is fully scored.

## Distributions

```
p_U_t(a) = ( C_t(a) + a0 ) / ( n_t + a0 V )
p_B_t(a) = ( C_t(x_{t-1}, a) + a1 p_U_t(a) ) / ( C_t(x_{t-1}) + a1 )
p_H_t(a) = ( C_t(h(x_{t-2}, x_{t-1}), a) + a2 p_B_t(a) ) / ( C_t(h(x_{t-2}, x_{t-1})) + a2 )
p_t(a)   = w_N p_N_t(a) + w_B p_B_t(a) + w_H p_H_t(a)
```

`h(.,.)` is a power-of-two hash bucket; collisions just yield a coarser context state.
The unigram is a smoothing prior for the bigram, and the bigram a smoothing prior for
the hash-trigram, so all three experts are normalized.

## Theoretical motivation

Hedge / exponential weights gives an O(log K / eta + eta C L^2) regret bound vs the
single best expert across chunks (K=3 here). The neural TTT model is one expert,
so the mixture is protected against catastrophic n-gram failure while letting the
n-gram experts win on repeated boilerplate, URLs, lists, tables, markup, and local
web-text burstiness that small neural models systematically miss.

## Compliance

- Each chunk is scored under `torch.no_grad()` strictly before any TTT update on that chunk
- Within a chunk, prefix-rank counts ensure a token never contributes to its own predicted probability
- Mixture weights for chunk c+1 are a function only of chunks 0..c losses
- All three statistical experts produce normalized probabilities over the SP8192 vocabulary
- The neural TTT update path is unchanged from PR #1493

## Hyperparameters

| Variable | Default |
|----------|---------|
| NGM_ENABLED | 1 |
| NGM_BUCKETS | 4096 |
| NGM_ALPHA0 | 0.1 |
| NGM_ALPHA1 | 16.0 |
| NGM_ALPHA2 | 32.0 |
| NGM_ETA | 0.35 |
| NGM_LOGW | "0,-4,-3" |

## Run

```bash
SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
NGM_ENABLED=1 NGM_BUCKETS=4096 NGM_ALPHA0=0.1 NGM_ALPHA1=16 NGM_ALPHA2=32 \
NGM_ETA=0.35 NGM_LOGW="0,-4,-3" \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Base credit

Built on PR #1493 by clarkkev / dexhunter / abaybektursun / Robby955 / msisovic / X-Abhishek-X.
