![frugendorff](https://github.com/user-attachments/assets/b99f0186-994a-43f3-9ba4-c7d1a5c055f7)

## The Frugendorff: Recursive Weight Sharing for Transformer Compression

A weight-sharing architecture where K unique transformer blocks are each applied N times in sequence, producing deeper effective networks from fewer stored parameters. The saved parameter budget is reinvested into wider MLP layers (4x expansion) that would otherwise exceed the artifact size limit.

This is not a SOTA submission. It is a research direction exploring whether recursive weight reuse can improve compute-per-byte efficiency in size-constrained language models.

## Result

- **val_bpb: 1.1478** (sliding window, stride 64)
- **Artifact: 15.19 MB** (under 16 MB limit)
- 8xH100 SXM, 600s, seed 1337

**Interpretation:** Competitive performance given the architectural compression — 6 stored blocks doing the work of 12. The primary value is demonstrating a viable direction for parameter efficiency, not chasing the leaderboard. Current results reflect a partially stabilized version of the architecture, not its theoretical ceiling.

## Architecture

The architectural contributions, separate from the training stack:

| Component | Detail |
|-----------|--------|
| Recursive blocks | 6 unique × 2 loops = 12 effective depth |
| MLP expansion | 4x (hidden 2560) — enabled by parameter savings from sharing |
| Loop position embeddings | QR-initialized orthogonal vectors, one per loop iteration |
| U-Net skip connections | Within each loop iteration |
| Activation | relu-squared |
| Attention | GQA (10 heads, 5 KV), XSA on last 2 blocks |
| Input conditioning | BigramHash (8192 buckets), SmearGate |
| Embeddings | Tied, with shared value embeddings (128d) |
| Parameters | 28.2M stored, behaving as 12-layer network |

## Training & Stabilization Stack

These are standard techniques applied to stabilize the recursive architecture during training:

| Technique | Purpose |
|-----------|---------|
| Muon (matrices) + AdamW (embeddings, scalars) | Optimizer split by parameter type |
| EMA (decay 0.9985) | Weight averaging for smoother convergence |
| Late QAT (int6 STE at LR scale < 0.15) | Quantization-aware fine-tuning in final phase |
| Training data replay (2 epochs, last 100 batches) | Sharpening on recent data before export |
| Self-distillation (EMA teacher, 50 steps) | Knowledge transfer from averaged weights |
| GPTQ int6 + zstd-22 | Final compression for artifact export |

## What We Understand So Far

400+ experiments across DGX Spark, G100, and 8xH100 clusters. Key findings:

**Weight sharing works as compression.** At matched parameter count, shared-wider (3×3, 864d) beats unique-narrow (9L, 512d) by 7.1% BPB. The mechanism is straightforward: fewer unique blocks means more parameter budget for width.

**Shared weights create gradient conflict.** Training all loops every step produces a "bandsaw" loss oscillation — the shared block receives contradictory gradient signals from different loop positions. This is a training dynamic unique to recursive architectures.

**Cadence training resolves the conflict.** Alternating fractal steps (all loops) with normalize steps (single clean pass) breaks the destructive interference. A 227-run automated sweep found cadence 4 (F/N/N/N) optimal. Normalize steps run ~10x faster than fractal steps.

**Test-time training gets an N× multiplier.** Updating shared weights during evaluation improves all loop iterations simultaneously. On a 3×4 Frugendorff, TTT yielded 0.032 BPB improvement (1.2217 → 1.1901 peak) — roughly 3× larger than typical TTT gains. The unsolved problem is drift: performance peaked at window 1400, then degraded.

**Quantization is the hard problem.** GPTQ catastrophically degrades shared models — pre-quant BPB of 1.37 balloons to 5.7+ post-quant. Quantization error on shared weights compounds multiplicatively across loops. The Frugendorff Squared (6×2, only 2 loops) survives because the compounding is minimal. Deeper sharing (3×4, 4×5) does not survive standard GPTQ.

## Experiment Summary

### Architecture search (368 automated runs, DGX Spark)

| Config | Best BPB | Notes |
|--------|----------|-------|
| 4×2, cadence 4, MLP 4x | **2.155** | Overall best (300-step proxy) |
| 5×2, cadence 4, MLP 4x | 2.185 | Close second |
| 6×1, no sharing | 2.196 | Non-shared baseline |
| 4×3, cadence 3, MLP 4x | 2.202 | More loops, diminishing returns |
| 8×2, cadence 3, MLP 4x | 2.329 | Too many unique blocks |

### Full-scale runs (H100)

| Config | Sliding BPB | Artifact | Notes |
|--------|------------|----------|-------|
| **6×2, 640d, MLP 4x** | **1.1478** | **15.15 MB** | Best overall (this submission) |
| 3×4, 960d + TTT | ~1.1901 peak | 14.3 MB | Unstable — drifts after w1400 |
| 3×4, 960d, stable | 1.2113 | 14.2 MB | Best without TTT |
| 2×4, 1024d | 1.2715 | 11.3 MB | Smallest artifact |
| 6×2, 512d (hybrid) | 1.1757 | 10.65 MB | Narrower variant |

### Quantization stress test (G100, single GPU)

| Loops | Pre-quant | Post-quant | Verdict |
|-------|-----------|------------|---------|
| 2 (this submission) | 1.1570 | 1.1716 | Survives |
| 3 | 1.3766 | 5.716 | Catastrophic |
| 4 | 1.4058 | 6.313 | Catastrophic |
| 5 | 1.4138 | 6.246 | Catastrophic |

## Next Directions

- **Stabilize deeper shared regimes** — reduce reliance on unique "flat" layers; the 6×2 config only shares modestly. The architecture should support 4×3 or 3×4 without quality collapse.
- **Loop-conditioned input embeddings** — make n-gram hash loop-aware so each iteration gets distinct conditioning, potentially eliminating gradient conflict without cadence.
- **Quantization strategies for shared weights** — loop-aware GPTQ calibration, selective precision (keep shared block in fp16), per-loop dequant offsets.
- **TTT drift stabilization** — find the drift gate sweet spot to sustain the N× leverage multiplier without degradation past window 1400.
- **Reclaim stabilization overhead** — current training stack includes significant stabilization machinery (EMA, replay, QAT). As the architecture matures, redirect that overhead back into model capacity.

## Compliance

No test-time training on validation data. Training replay and self-distillation operate on training data only. All evaluation follows score-first protocol per issue #402.
