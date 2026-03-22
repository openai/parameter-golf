# Leapfrog Experiment Results — 2026-03-22

Target: Beat PR #414 (1.1233 BPB, 15.55 MB)

## Results Summary

| Variant | Description | Sliding BPB (s64) | Size | Verdict |
|---------|-------------|-------------------|------|---------|
| v1 seed 1337 | TTT burst (2ep, 10% LR, before EMA) | **1.12319** | 15.68 MB | WINNER |
| v1 seed 42 | Same as above, different seed | 1.12397 | 16.37 MB | Over size |
| v1b seed 1337 | EMA-first, then burst (1ep, 5% LR, QAT) | 1.12624 | 15.97 MB | Worse BPB |
| v1c seed 1337 | Burst+QAT before EMA + 15 GPTQ percentiles | 1.12319 | 15.68 MB | Same as v1 |
| v2 seed 1337 | Self-distillation (50 steps, KL+CE) | 1.12328 | 15.62 MB | ~Tied with v1 |
| v4 seed 1337 | Burst + distill + train_seq_len=1024 | 1.22243 | 15.53 MB | BUST |

## Key Findings

1. **TTT burst before EMA works** — replaying 100 recent batches for 2 epochs at 10% LR, with EMA updates, then applying EMA. Gives ~0.0001 over baseline.

2. **Self-distillation matches burst** — using EMA as teacher with KL+CE loss lands in the same spot. Both approaches hit the same ceiling.

3. **Stacking burst + distill doesn't help** — the two techniques capture the same signal.

4. **EMA-first then burst is worse** — the burst needs to happen before EMA so EMA can smooth the sharpened weights.

5. **15 GPTQ percentiles = no gain over 5** — the original 5 percentiles already find near-optimal clips.

6. **train_seq_len=1024 is catastrophic** — only 6% more steps but massive quality loss. Partial RoPE extrapolation from 1024→2048 is not good enough.

7. **zlib vs zstd matters for size, not BPB** — same quantization, different compression. zstd-22 saves ~1.3MB.

## Submitted

PR #445: v1 seed 1337, 1.12319 BPB, 15.68 MB
