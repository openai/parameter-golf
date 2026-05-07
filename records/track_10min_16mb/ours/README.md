# Non-record: QAT & EMA Negative Results on SOTA Stack

**val_bpb: 1.1426** (baseline reproduction) | 8xH100 SXM, 600s

## Summary

This submission documents a **negative result**: adding Quantization-Aware Training (QAT) and Exponential Moving Average (EMA) to the current SOTA stack (PR #180) **hurts performance** due to throughput loss. The techniques close the quantization gap and improve compression, but the lost training steps outweigh the benefits within the 10-minute budget.

## Approach

Built on the #1 SOTA (PR #180, thwu1, 1.1428 BPB) with two additions:

### QAT with Straight-Through Estimator (STE)
- Per-layer fake quantization during forward pass
- Matching target precision: int5 (clip=15) for MLP, int6 (clip=31) for attention
- Configurable warmup delay (`QAT_WARMUP_STEPS`)
- Implementation: STE in `CastedLinear.forward()` — forward sees quantized weights, backward sees full precision

### EMA (Exponential Moving Average)
- Shadow copy of weights updated every step with decay 0.9999
- Applied before SWA for final eval
- Configurable start step and decay rate

## Results (8xH100 SXM, 600s)

| Config | Steps | val_bpb | Artifact | Delta |
|--------|-------|---------|----------|-------|
| Baseline (PR #180 repro) | 6,684 | **1.1426** | 15.99 MB | — |
| + QAT (warmup=500) | 6,143 | 1.1473 | 15.69 MB | +0.005 (worse) |
| + QAT + EMA | 4,546 | 1.1606 | 16.89 MB | +0.018 (worse) |

## Key Findings

### QAT: Better compression, worse score
- **-8% throughput** (6,143 vs 6,684 steps) due to per-forward fake quantization overhead
- **Smaller artifact** (15.69 MB vs 15.99 MB) — QAT does produce more compressible weights
- **Net negative**: the 541 lost training steps matter more than closing the quantization gap
- The SOTA stack already has SWA + magnitude pruning which partially addresses quantization, reducing QAT's marginal value

### EMA: Catastrophic throughput loss
- **-32% throughput** (4,546 vs 6,684 steps) due to `.cpu().clone()` every step
- **Over 16MB artifact** (16.89 MB) — EMA-averaged weights compress worse
- Not viable in the current form within the 10-minute budget

### Implication
Techniques that trade training steps for inference/compression quality are likely counterproductive in this challenge. The 10-minute wallclock constraint makes every step precious — ~89ms/step on 8xH100 means each lost step is ~89ms of irreplaceable training.

## Potential fixes (not implemented)
- **Late QAT**: apply QAT only in the last 1000-2000 steps instead of from step 500
- **In-place EMA**: keep EMA on GPU to avoid CPU transfer overhead
- **Periodic QAT**: fake-quantize every K steps instead of every forward pass

## Hardware

8x NVIDIA H100 80GB HBM3 SXM (RunPod on-demand). ~89ms/step baseline.

## Command

```bash
# Baseline (reproduces SOTA)
RUN_ID=ablation_A_baseline QAT_ENABLED=0 EMA_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# QAT
RUN_ID=ablation_B_qat QAT_ENABLED=1 QAT_WARMUP_STEPS=500 EMA_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# QAT + EMA
RUN_ID=ablation_C_qat_ema QAT_ENABLED=1 QAT_WARMUP_STEPS=500 \
EMA_ENABLED=1 EMA_DECAY=0.9999 EMA_START_STEP=500 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included files

- `train_gpt.py` — SOTA #1 fork with QAT + EMA + MPS fallback
- `ablation_A_baseline.txt` — Baseline reproduction log
- `ablation_B_qat.txt` — QAT ablation log
- `ablation_C_qat_ema.txt` — QAT + EMA ablation log
- `submission.json`
