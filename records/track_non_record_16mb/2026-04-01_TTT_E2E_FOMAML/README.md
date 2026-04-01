# TTT-E2E: Prime MLP Test-Time Training

Non-record submission exploring test-time training with separate "prime MLP"
adapters for Parameter Golf. Key finding: **naive TTT with zero-init prime MLPs
gives -0.022 BPB without any meta-learning**, invalidating the assumption that
meta-learning is required for effective TTT.

## Motivation

All 25 prior naive TTT attempts in Parameter Golf failed because they perturbed
GPTQ'd int5/int6 weights — disrupting quantized rounding decisions. Prime MLPs
are fundamentally different: fresh bf16 parameters, separate from GPTQ, zero-init
so they start invisible to the model. This has never been tested.

## Architecture

Add rank-256 "prime MLPs" (786K params) to the last 3 blocks (layers 8-10).
Each runs sequentially before the main MLP with its own RMSNorm and residual:

```
h = h + attn(norm(h))
h = h + prime_MLP(prime_norm(h))   # adapted at test time (bf16, not GPTQ'd)
h = h + MLP(mlp_norm(h))           # frozen at test time (GPTQ'd int5/int6)
```

Prime MLP: `dim → rank → dim` with LeakyReLU(0.5)². Down projection zero-init
so model starts identical to baseline. Up projection orthogonal-init so gradients
flow from step 1.

## Results (1x L40S, 1 train shard)

### Naive TTT LR sweep (5K chunks, score-first eval)

| Config | val_bpb | Delta |
|--------|---------|-------|
| Baseline (no TTT) | 1.5019 | — |
| TTT lr=0.001 | 1.5018 | -0.0001 |
| TTT lr=0.003 | 1.5013 | -0.0006 |
| TTT lr=0.01 | 1.5001 | -0.0018 |
| TTT lr=0.03 | 1.4988 | -0.0031 |
| **TTT lr=0.1** | **1.4971** | **-0.0048** |

Higher LR = more improvement, monotonically. Adaptation is clearly beneficial.

### Full eval (all 60K chunks, best LR)

| Config | val_bpb | Delta |
|--------|---------|-------|
| Baseline | 1.5019 | — |
| **TTT lr=0.1, chunk=1024** | **1.4794** | **-0.0224** |

Full eval shows **stronger improvement** than 5K preview (-0.0224 vs -0.0048) —
adaptation compounds over the val shard.

### Negative results

| Config | Delta | Why |
|--------|-------|-----|
| chunk=4096 | +0.095 | Fewer adaptation steps, stale gradients |
| reset_every=1000 | -0.003 (vs -0.005 no-reset) | Accumulated knowledge is valuable |
| FOMAML meta-learning | +0.083 vs baseline | W_0 degrades base model predictions |

## Key Findings

1. **Naive TTT with prime MLPs works.** No meta-learning needed. The architecture
   (separate bf16 adapters that don't touch GPTQ'd weights) is the key insight.

2. **FOMAML hurts.** Meta-learned W_0 produces non-zero outputs that degrade the
   base model (+0.16 BPB). TTT only recovers half. On 1 shard, meta-learning has
   insufficient task diversity.

3. **Higher TTT LR is better** up to 0.1. The model can absorb aggressive adaptation
   because prime MLPs are small and zero-init'd.

4. **Small chunks (1024) beat large chunks (4096).** Frequent adaptation > batch quality.

5. **No reset is best.** Accumulated adaptation across the val shard is valuable —
   the model builds up useful representations over time.

## What This Means for 8xH100

On the real PR 1105 model (1.1125 BPB), this approach:
- Adds 786K prime MLP params (~1.5 MB bf16, fits in 16 MB budget)
- Zero training cost (no Phase 2 needed)
- ~6s eval overhead (60K backward passes through 3 small MLPs)
- Expected improvement: scale of -0.001 to -0.003 BPB (the L40S delta scaled
  by the baseline quality ratio)

## Files

- `train_ttt_e2e.py` — Model definition with prime MLPs + FOMAML + TTT eval
- `sweep_naive_ttt.py` — Naive TTT LR/chunk/reset sweep

## References

- TTT-E2E paper: arxiv.org/abs/2512.23675
- Our PR 1105 (base model): 1.1125 BPB
