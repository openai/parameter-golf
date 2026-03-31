# BW6_Skipgram — Hypothesis

## One variable
`TRIGRAM=1` — enable trigram hashing into the existing bigram embedding table.

## Parent
`crawler/2026-03-29_BW5/` (champion: 1.18672385 BPB, 8.61MB)

## What changes
`BigramHashEmbedding.forward` accumulates a trigram hash on top of the bigram hash:
- Bigram: hash(t-1, t) → embed lookup
- Trigram: hash(t-2, t-1, t) → same embed table lookup, summed

**Zero extra parameters.** Same embedding table (`BIGRAM_VOCAB_SIZE=2048, BIGRAM_DIM=128`),
same projection, same scale. The trigram hash is just an additional index into the
existing 2048-slot table.

## Why
The crawler processes tokens in a sliding window with recurrent loops. Each loop
re-reads the residual stream. Bigram context (t-1, t) is already fed in at embedding
time. Adding trigram context (t-2, t-1, t) at zero parameter cost gives the crawler
richer n-gram signal at the input, potentially helping the recurrent loops compress
longer-range local patterns.

The neural SOTA (Rascal lineage) has this feature in its `BigramHashEmbedding`.
The crawler does not. This is a direct port.

## Expected effect
- Quality: positive. Zero-param enrichment of input features.
- Speed: neutral to negligible. One extra `embed()` lookup per forward pass.
- Size: neutral. No new parameters.

## Gate target
raw_bpb < BW5 control (same arm) at 2000 steps on 1 GPU.
Step avg should stay within ±2ms of BW5 baseline (~585ms on 1GPU with grad_accum=8).
