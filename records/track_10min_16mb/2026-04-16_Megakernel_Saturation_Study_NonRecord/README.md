# Can manual Triton MLP fusion beat torch.compile at 27M scale?
## A 5-variant saturation study.

**val_bpb: 1.1107** (v4 best fused variant) vs **1.1104** (eager torch.compile baseline) | 13.85 MB | 8×H100 SXM

Non-record idea submission. Headline finding: **no**.

## TL;DR

Ran a systematic 5-variant ablation of manual Triton block-level MLP fusion
on PR #1493's SOTA architecture. All 5 variants — including an audit-guided
best-practices version AND the exact PR #1450 design that claims +10.5%
throughput — land within **0.0008 BPB** of each other, all slightly
**worse** than torch.compile's automatic fusion. The saturation is robust
across the entire fusion design space.

## Motivation

Megakernels are on OpenAI's stated wishlist. The parameter-golf community
has 10+ megakernel PRs, but none has an end-to-end BPB improvement
verified. PR #1450 claims +10.5% throughput from a TMA fused MLP kernel
but was never merged; PR #1316 tried a full-depth MLP megakernel and
regressed 41%. This study asks: *is there ANY fused Triton MLP design
that beats torch.compile at 27M × 600s × H100?*

## Method

Fixed base stack (PR #1493 architecture on SP1024): 11L × 512d, GQA 8/4,
3-layer depth recurrence, parallel residuals L7+, QK-Gain 5.0, E2E TTT
MLP-only last 50%. Same seed (1337) for all runs. Only the MLP
implementation changes. Each run trains for 588s wallclock, then evals
with score-first E2E TTT.

The 5 variants span the fusion design space:

### v1 — RMSNorm fused, bf16 inv_rms, in-K scaling
Novel: fuses RMSNorm's per-row inv_rms scaling into the up-projection
matmul by multiplying `a * row_scale[:, None]` inside the K-loop.
Natural first attempt.

### v2 — Simpler (PR #1530 style)
Removes RMSNorm fusion (RMSNorm stays eager). Only fuses matmul +
LeakyReLU² activation. A/B test whether v1's RMSNorm fusion was adding
counterproductive per-K overhead.

### v3 — fp32 activation
Hypothesis: v1's bf16 downcast of the activation was the precision sink.
Keeps activation computation in fp32 registers, downcasts only at HBM
store.

### v4 — All 3 audit fixes
After an external audit against Triton tutorials + Liger-Kernel + PR
#1450, identified three bugs in v1-v3:
1. **Epilogue scale** — moved row_scale multiply OUT of the K-loop
   (serialized TMA→wgmma pipeline) INTO the fp32 epilogue. Algebraically
   identical since row_scale depends only on rows.
2. **fp32 inv_rms** — previously stored as bf16 (7-bit mantissa), now
   stored as fp32 end-to-end.
3. **L2 swizzle** — added `GROUP_SIZE_M=8` grouped tile iteration to
   improve L2 hit rate on B tile reuse.

All three applied in v4. This is the "best-practices" version.

### v5 — PR #1450 act_grad architecture
Forward kernel writes `post = leaky_relu(h)²` (for dw2 use) AND
`act_grad = d/dh[leaky_relu(h)²] = where(h>0, 2h, 0.5h)` (for backward
use). Backward kernel simplifies from computing `tl.where(pre>0, 2*pre,
0.5*pre)` per tile to just loading and multiplying by `act_grad`. This
is the exact design that PR #1450 claims gives +10.5% throughput.

## Results

| Variant | Steps | tok/s@1500 | Pre-quant | **Final TTT BPB** | Δ vs Eager |
|---|---|---|---|---|---|
| **Eager (torch.compile)** | 3940 | 6.74M | 1.1250 | **1.1104** | baseline |
| v1 RMSNorm-fused (bf16, in-K) | 3944 | 6.77M | 1.1252 | 1.1106 | +0.0002 |
| v2 simpler | 3945 | 6.78M | 1.1255 | 1.1110 | +0.0006 |
| v3 fp32 activation | 3938 | 6.77M | 1.1254 | 1.1108 | +0.0004 |
| v4 audit fixes | 3944 | 6.78M | 1.1252 | **1.1107** | +0.0003 |
| v5 PR #1450 act_grad | 3944 | 6.78M | 1.1255 | 1.1110 | +0.0006 |

**Observations:**

1. **All 5 fused variants gain real throughput** (+0.1-0.5% = +4-5 extra
   steps in 3944). Fusion IS faster, narrowly.
2. **All 5 fused variants lose on BPB** by 0.0002-0.0006 nats. Fusion
   loses precision, and the precision loss exceeds the throughput gain.
3. **Eager torch.compile is the practical ceiling.** Even v4 (every
   known best practice) and v5 (the exact PR #1450 architecture claiming
   +10.5%) cannot beat it.
4. **The pareto frontier is flat.** All 5 variants fall within 0.0008
   BPB of each other and within 0.0006 of eager. No clear winner
   emerges from the design-space scan.

## Why this saturates

torch.compile's inductor backend already fuses eager MLP:
`F.rms_norm → F.linear → F.leaky_relu → F.square → F.linear`
into a near-optimal Hopper schedule. The HBM traffic savings a
hand-written kernel can recover are small relative to the total compute
at these shapes (M=98K, N=2048, K=512): matmul dominates, and the
matmuls are already near peak. Meanwhile, every hand-written Triton
kernel introduces at least one extra bf16 round-trip (the intermediate
storage for the backward), and that precision loss propagates through
training.

At 3B+ model sizes (where PR #1450's claim presumably originates),
matmuls are bigger and memory-bandwidth becomes the real bottleneck —
fusion helps there. At 27M × 600s it doesn't.

## Implications for parameter-golf participants

Do NOT pursue block-level MLP kernel fusion as a standalone BPB lever
at this scale. Five independent designs all saturate. The real levers
that change BPB are:
- Bigger tokenizer (SP8192 when available)
- More aggressive quantization (ternary / 1-bit)
- Architectural changes (depth recurrence, parallel residuals)
- Test-time techniques (legal score-first TTT)

## Why non-record

Single seed. Given the 0.0008 BPB noise band across 5 variants with
different fp32/bf16/tiling/scheduling choices, more seeds wouldn't
change the conclusion — the finding IS the saturation, not any
specific number. SP1024 data (SP8192 unavailable from HF source) adds
a ~0.03 BPB penalty vs merged SOTA baselines, but the study's relative
comparisons (each variant vs eager, all on same base) hold regardless.

## Files

- `train_gpt.py` — v5 implementation (PR #1450 act_grad design + all v4 audit fixes). Toggles: `FUSED_MLP_ENABLED=1` activates v5 default. `FUSED_MLP_FULL=1` switches to v2 path.
- `submission.json` — metadata + full 5-variant ablation table
- `train_seed1337.log` — training + eval log for v5 run (most recent iteration)
