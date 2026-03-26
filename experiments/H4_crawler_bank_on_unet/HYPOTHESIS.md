# H4: Crawler Bank on U-Net Frame (GS v7)

## Question
Does a single shared block (crawler bank) at the U-Net bottleneck improve
BPB over equivalent unique layers? Is weight-shared depth worth the compute
trade-off when placed at the compression point of an encoder-decoder?

## Motivation
H1 proved recursion (C-step double-firing) is overhead. But the crawler
architecture still won at 1.1325 — its value comes from weight-shared depth.
The question is whether that concept belongs on a flat stack (4f+2cx2) or
at the bottleneck of a U-Net where it acts as a learned compression step.

GS v7 (1.1206 BPB, 15.56MB) is an 11L/512d U-Net: 5 encoder + 6 decoder
with skip connections. Adding a crawler bank at the bottleneck gives:

- **Free effective depth** at zero param/artifact cost
- The bottleneck is the information pinch point — weight sharing here
  forces the model to learn a reusable compression/decompression transform
- Each extra loop costs ~1% wallclock (~70 fewer steps in 600s)

## Architecture

**Control (A): GS v7 as-is**
```
11 unique layers: [enc0-enc4] → [dec0-dec5]
5 encoder + 6 decoder, skip connections
11 stored blocks, 11 effective depth
```

**Test (B): GS v7 + 1 crawler bank at bottleneck**
```
10 unique layers + 1 shared × 2 loops:
[enc0-enc4] → [crawler × 2] → [dec0-dec4]
11 stored blocks, 12 effective depth
Same param count. Extra compute from 1 additional forward pass.
```

Alternative: 9 unique + 1 shared × 3 = 12 effective from 10 stored.
Saves 1 block of params (~2.8M) that could widen dim or add trigram.

## Prediction
The crawler bank at the bottleneck will show a small improvement (0.001-0.003)
because the U-Net pinch point benefits from iterative refinement more than
arbitrary depth positions. The weight sharing regularizes the bottleneck
representation, forcing it to be reusable across passes.

If the extra effective depth doesn't overcome the step count loss (~1-2%
fewer steps), the crawler bank is not worth it.

## Scale
0.25 (150s wallclock, TTT/distill OFF)

## Arms
| Arm | Config | Effective depth | Stored blocks |
|-----|--------|----------------|---------------|
| A (control) | GS v7 11L | 11 | 11 |
| B (crawler bank) | GS v7 10L + 1 shared × 2 | 12 | 11 |

## Diagnostic Focus
1. val_bpb at matched wall-clock
2. Steps achieved (B will get slightly fewer)
3. Quant gap — does the shared block produce harder-to-quantize activations?

## Results (2026-03-24, 8xH100 SXM, 0.25 scale)

| Arm | Steps | step_avg | val@500 | val@1000 | val@1500 | post_ema | sliding_bpb | artifact |
|-----|-------|----------|---------|----------|----------|----------|-------------|----------|
| A (control) | 1,744 | 86ms | 1.3980 | 1.3071 | 1.2475 | 1.2308 | **1.2145** | 14.54MB |
| B (crawler) | 1,507 | 99ms | 1.3958 | 1.2936 | 1.2318 | 1.2506 | 1.2371 | 14.08MB |

## Status
COMPLETED — crawler bank at U-Net bottleneck is retired for 10-minute track.

## Verdict

**REFUTED.** Crawler bank loses by 0.023 sliding BPB despite better per-step learning.

Per-step learning IS better with the crawler bank:
- +0.002 at step 500, +0.014 at step 1000, +0.016 at step 1500
- Weight sharing at the bottleneck genuinely improves per-step quality

But in a wallclock-limited competition:
- 15% slower per step → 14% fewer total steps (1507 vs 1744)
- post_ema 0.020 worse (EMA struggles with shared block dynamics)
- Quantization 0.023 worse (shared block activations harder for GPTQ)
- The step count + quant penalty overwhelms the per-step advantage

Artifact is 0.46MB smaller with crawler (weight sharing compresses well).
This is the only advantage, and it doesn't help when BPB is worse.

**Conclusion: in wallclock-limited training, steps beat tricks.** The crawler
concept is a genuine regularizer but its compute cost exceeds its benefit.
The GS v7 U-Net without modifications remains the strongest frame.
