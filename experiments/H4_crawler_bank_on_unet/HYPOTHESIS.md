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

## Verdict
_To be filled after runs complete._

| Arm | Steps | val@500 | final_val | post_ema | sliding_bpb | quant_gap |
|-----|-------|---------|-----------|----------|-------------|-----------|
| A | | | | | | |
| B | | | | | | |

## Status
READY — needs code modification to GS v7 training script.
