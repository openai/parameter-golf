# 11L GPTQ-lite + Self-Distillation TTT

**val_bpb: 1.1260** (sliding window, stride=64) | **15.99 MB** | 8xH100 SXM, 600s

## Architecture

Built on PR #374's SOTA stack with two novel post-training optimizations.

- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3x MLP expansion with relu-squared activation
- Efficient Partial XSA on last 4 layers
- Partial RoPE (16/64 dims) + NTK-aware scaling
- LN Scale Factor 1/sqrt(layer_idx+1)
- U-Net skip connections (5 encoder, 6 decoder)
- SmearGate + BigramHash (2048 buckets, dim=128)
- Shared Value Embedding (dim=128, layers 9,10)
- FlashAttention 3 (Hopper)
- Orthogonal init with proj scaling
- Tight SWA (scale<0.2, every 50 steps, 12 checkpoints)
- Late QAT (STE int6 at lr_scale<0.1)
- EMA not used (Tight SWA instead)

## Novel Contributions

### 1. GPTQ-lite: Per-Layer Optimal Clip Percentile Search

Standard int6 quantization uses a fixed clipping strategy (row-wise amax). GPTQ-lite searches 5 clip percentiles per weight matrix (1.0, 0.999, 0.995, 0.99, 0.98) and selects the one minimizing reconstruction error. This reduces quantization degradation at zero compute cost during training.

### 2. Self-Distillation TTT (Eval-Time Adaptation)

Post-training KL-divergence adaptation on validation data. A frozen teacher (snapshot of the trained model) guides the student's adaptation, preserving XSA attention patterns that hard-label TTT disrupts (as documented in PR #303's negative interaction study). Temperature=2.0, freeze first 4 blocks, 2 epochs SGD (lr=0.001).

Result: SDTTT was slightly negative (-0.0003 bpb) in this run. The KL constraint may be too strong at T=2.0. Included for completeness and future tuning.

## Training

- Muon optimizer (matrices): lr=0.025, momentum=0.99 (warmup 0.92 over 1500 steps), WD=0.04
- AdamW (embeddings): lr=0.035, (scalars): lr=0.025, WD=0.04
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 3000 iters (wallclock-based)
- Tight SWA: every 50 steps when scale<0.2 (12 checkpoints)
- Late QAT: STE int6 when LR scale<0.1

## Results

| Metric | Value |
|--------|-------|
| Steps | 6,701 |
| Step avg | 89.55ms |
| Pre-quant val_bpb | 1.1429 |
| Post-SWA val_bpb | 1.1428 |
| Post-SDTTT val_bpb | 1.1431 |
| Int6 roundtrip val_bpb | 1.1497 |
| **Sliding window val_bpb (s64)** | **1.1260** |
| Artifact size | 15,989,300 bytes |
| Peak memory | 20,680 MiB/GPU |

## Run

```bash
SDTTT_ENABLED=1 SDTTT_EPOCHS=2 SDTTT_LR=0.001 SDTTT_TEMPERATURE=2.0 \
SDTTT_FREEZE_BLOCKS=4 GPTQ_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All other hyperparameters use PR #374 defaults (NUM_LAYERS=11, XSA_LAST_N=4, SWA_ENABLED=1, etc.).

## Code

- Full source and experiment history: https://github.com/dannywillowliu-uchi/parameter-golf-entry
