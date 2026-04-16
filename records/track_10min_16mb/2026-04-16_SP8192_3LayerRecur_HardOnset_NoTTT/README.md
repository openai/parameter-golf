# Record: SP8192 + 3-Layer Recurrence + Hard Onset + No TTT

**val_bpb: 1.0858** (seed 1337) | **~15.86 MB** | 8×H100 SXM, 600s | No TTT

## Results (8×H100 80GB SXM)

| Seed | Steps | Pre-quant BPB | Roundtrip BPB | **Sliding BPB** | Artifact |
|------|-------|---------------|---------------|-----------------|----------|
| 1337 | 5,247 | 1.0854 | 1.1027 | **1.0858** | 15,857,563 |

## Key Features

### 3-Layer Recurrence with Hard Onset

Layers 3, 4, 5 repeat as virtual layers [0,1,2,3,4,5,**3,4,5**,6,7,8,9,10], activated at step 3000 via hard gate. This is the D0 recipe from the recurrence stack (PR #1394 lineage) without homotopy smoothing.

### No Val Pauses

`VAL_LOSS_EVERY=99999` eliminates mid-training validation, reclaiming ~216s of wallclock for gradient steps. This alone yields ~600 extra training steps vs the standard recipe.

### No TTT

This submission does not use test-time training. The improvement is purely from training dynamics.

### GPTQ int6 + Brotli

Standard GPTQ with `SDCLIP_K=12.85`, 64 calibration batches, brotli compression. Quant tax (fp32 → int6 sliding) = +0.00046 BPB.

## Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 11 (+ 3 recurrent) |
| Model dim | 512 |
| Heads / KV heads | 8 / 4 |
| MLP mult | 4× |
| Vocab | 8192 (SP BPE) |
| Activation | LeakyReLU(0.5)² |
| EMA decay | 0.9965 |
| Recurrence | Layers 3,4,5, onset step 3000 |
| QK Gain | 5.25 |
| Logit softcap | 30.0 |

## Training Config

| Parameter | Value |
|-----------|-------|
| Batch tokens | 786,432 |
| Sequence length | 2048 |
| Warmup steps | 20 |
| Warmdown frac | 0.72 |
| Max wallclock | 600s |
| Optimizer | Muon (momentum 0.99) + Adam (embed/scalar) |
| Matrix LR | 0.022 |
| Embed LR | 0.6 |
| Weight decay | 0.095 (Muon + Embed) |
| Grad clip | 0.3 |

## Comparison to D0 Baseline

D0 baseline (same recipe, same codebase): int6 sliding = 1.09022. This submission improves by **-0.0044 BPB**, primarily from eliminating val pauses (+607 extra training steps within the 600s wallclock).
