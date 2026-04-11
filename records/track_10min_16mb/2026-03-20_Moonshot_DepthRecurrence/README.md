# 2026-03-20_Moonshot_DepthRecurrence

**Verdict: ❌ Catastrophic failure — do not retry at this scale**

Depth recurrence with Huginn-style eval-time loop scaling. Trained 3 unique blocks × 3 loops
(effective depth 9), then doubled loops to 6 at eval time. Both variants (U-Net skips and flat)
produced near-random output at eval depth.

---

## Results

| Variant | Train loops | Eval loops | Pre-quant val_bpb | Post-quant val_bpb |
|---------|------------|-----------|-------------------|--------------------|
| v1: with U-Net skips | 3 | 6 | ~1.29 | **4.34** |
| v2: flat (no skips) | 3 | 6 | 1.2934 | **5.5755** |

Both are effectively random output (uniform over 1024 tokens ≈ 10 BPB; 4–5 BPB means the model
is worse than random on the tokens it's most confident about).

Flat loops made it **worse** than skips: 5.58 vs 4.34.

For reference, the naive baseline scores 1.2244. A model outputting uniform random probabilities
would score ~6.9 BPB. These results sit between baseline and uniform random — not a degraded
model, a broken one.

---

## Configuration (v2 — flat loops)

```
NUM_LAYERS=3 NUM_LOOPS=3 FLAT_LOOPS=1
MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2
model_params: 7,608,344
step_avg: 46.5ms  steps: 12,903  seed: 1337
artifact: 5.85MB (int8+zstd-22)
```

v1 used U-Net skip connections between encoder and decoder halves of the loop stack.

---

## What Was Attempted

Huginn (arXiv:2502.05171) trains a 3.5B parameter model with shared blocks and scales
compute at inference by running more loops. The hypothesis was: train 3 shared 512-dim
blocks for 3 loops, then run 6 loops at eval to get "free" extra depth without increasing
artifact size.

**v1** (with skips): Used the existing U-Net skip weight architecture. During extra eval
loops the skip connections were disabled (skips only meaningful for the trained depth).

**v2** (flat): Removed skip connections entirely. Pure iterative refinement — each loop
applies the same 3 blocks, no special handling for different loop indices.

---

## Why It Fails

The blocks learn a **specific transformation for their position in the 3-loop stack**, not
a general iterative refinement operator. When run for 6 loops:

- The blocks expect to receive input that looks like "output of loop 2 block 3"
- After loop 3, the residual stream has the wrong statistical profile for the blocks
- Each additional loop compounds the distribution mismatch
- Output diverges toward noise

Flat loops (v2) are worse because without skip connections there is even less mechanism
to reset the residual stream to a recognizable state between loops.

**The fundamental issue is scale.** Huginn was validated at 3.5B parameters with ~390M
parameters per unique layer. At 7.6M total / ~2.5M per unique layer, the blocks lack
sufficient capacity to learn a transformation that is both:
1. A good language model at trained depth
2. A valid input distribution for re-application

These two constraints are compatible at large scale (the representation space is rich
enough) but not at 7.6M — the blocks overfit to being a specific-depth function.

---

## What Would Be Needed to Make This Work

1. **Much larger per-block capacity** — Huginn uses ~100M+ params per unique layer.
   At our artifact budget (~16MB int6+zstd), this is not achievable.

2. **Loop-conditioned blocks** — inject loop index as a learned embedding so blocks
   can behave differently at different depths. This adds parameters and complexity
   without the "free compute" benefit.

3. **Shorter train loops, more eval loops** — e.g., train 2 loops, eval 8. Reduces
   training quality to expand the gap. Unlikely to work at this scale for the same reason.

None of these are worth pursuing in the 10min/16MB track.

---

## Relationship to PR #167

PR #167 in the main competition repo also attempted depth recurrence at small scale.
This experiment was inspired by that approach and Huginn. Both failed — this is not
an implementation bug, it is a fundamental scale mismatch.

---

## Reproduction

```bash
git checkout int6-3xMLP-pr  # script lives on this branch (uses same base)
# v2 (flat):
NUM_LAYERS=3 NUM_LOOPS=3 FLAT_LOOPS=1 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_Int6_3xMLP/train_gpt.py
# At eval, set NUM_LOOPS=6 — results will be catastrophic
```

## Author

GitHub: [@mrdavtan](https://github.com/mrdavtan)
Date: 2026-03-20
