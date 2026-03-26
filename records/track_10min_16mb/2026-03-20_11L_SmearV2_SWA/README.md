# 11L Int6 QAT + Per-Dim SmearGate + SWA + Decoupled WD 0.038

**Mean val_bpb: 1.1480** (3 seeds) | **Best: 1.1453** (seed 1337) | **Artifact: 15.31 MiB**

Built on the modded-nanogpt baseline with techniques from PR #60 (@notapplica), PR #81 (@samacqua), and independent experimentation.

## Self-Verification (3 seeds)

| Seed | val_loss | val_bpb | Steps | ms/step | SWA ckpts |
|------|----------|---------|-------|---------|-----------|
| **1337** | 1.9339 | **1.1453** | 8052 | 74.49 | 30 |
| 7 | 1.9400 | 1.1490 | 8040 | 74.34 | 29 |
| 42 | 1.9413 | 1.1498 | 7772 | 77.17 | 27 |

**Mean: 1.1480** | Std: 0.0024 | Inter-seed variance: 0.0045

Improvement over prior SOTA (PR #60, 1.1748 BPB): **-0.0268 BPB / -0.0452 nats** (one-sided t-test: t=16.04, p < 0.001).

Note: Seed 42 ran on hot GPUs (no cooldown between runs), resulting in ~300 fewer training steps due to thermal throttling. Seeds 1337 and 7 ran on cold GPUs with 120s cooldowns.

## Pre-Quant vs Post-Quant Breakdown

| Stage | val_bpb (seed 1337) |
|-------|---------------------|
| Pre-quant (bf16) | ~1.130 |
| Post-quant int6 roundtrip | 1.1453 |
| Quantization penalty | ~0.015 |

Sliding window evaluation (stride=64) recovers ~0.03 BPB over standard fixed-window eval by scoring every token position with full 2048-token context.

## Ablation Summary

Starting from the 9L baseline (PR #60, 1.1748 BPB), each technique was added incrementally. Approximate BPB contributions measured on single-seed 8xH100 runs:

| Change | val_bpb | Delta |
|--------|---------|-------|
| 9L int8 baseline (PR #60) | 1.1748 | -- |
| + Int6 QAT (6-bit STE) | ~1.168 | -0.007 |
| + 11 layers (fits under 16MB with int6) | ~1.158 | -0.010 |
| + MLP 3x expansion | ~1.155 | -0.003 |
| + Per-dim SmearGate | ~1.153 | -0.002 |
| + Orthogonal init | ~1.152 | -0.001 |
| + Muon WD 0.038 | ~1.149 | -0.003 |
| + SWA (every 50, last 50%) | ~1.148 | -0.001 |
| + Sliding window eval (stride=64) | **1.1453** | -0.003 |

## Key Techniques

1. **11 Transformer Layers.** The single biggest gain comes from depth. Int6 compression makes 11 layers fit under 16MB where int8 only allows 9-10. Each additional layer adds ~0.005 BPB improvement at this model width, with diminishing returns beyond 11 layers due to the time budget (74ms/step at 11L leaves ~8000 steps in 10 minutes).

2. **Int6 Quantization-Aware Training (QAT).** Straight-through estimator (STE) injects fake int6 quantization during the forward pass: `w_dq = round(clamp(w/scale, -31, 31)) * scale`, with gradients flowing through via `w + (w_dq - w).detach()`. Per-row symmetric quantization with 6-bit clipping. Active from step 0 (no warmup needed — tested warmup at 15-75% with no improvement). The QAT overhead is ~5% wall-clock time, well worth the int6 compression savings.

3. **Int6-in-Int8 Compression.** Int6 values (-32 to 31) stored in int8 containers rather than bit-packed. Bit-packing destroys byte alignment and zstd can't compress it. Int8 containers with restricted values compress ~35% with zstd-22 (payload ratio 3.86x), achieving better final size than bit-packing.

4. **Per-Dim SmearGate.** Learned per-dimension gate `sigmoid(Parameter(dim))` blending current and previous token embeddings, zero-initialized (starts at sigmoid(0)=0.5 blend). More expressive than scalar gating while adding only 512 parameters. Provides a cheap bigram-like signal at the embedding level. Requires orthogonal initialization to work properly (without OrthoInit, SmearGate can hurt BPB by +0.003 per mrdavtan's ablations in PR #212).

5. **Stochastic Weight Averaging (SWA).** Checkpoint averaging every 50 steps over the last 50% of training (~29-30 checkpoints on 8000-step runs). SWA produces smoother weights that sit in flatter loss basins, which quantize significantly better — the pre-quant to post-quant penalty drops from ~0.025 to ~0.015 with SWA. Accumulation done in fp32 (bf16 causes catastrophic precision loss).

6. **Decoupled Muon Weight Decay (0.038).** Applied to matrix parameters via the Muon optimizer. Swept from 0.01 to 0.05; 0.038 was optimal. Weight decay regularizes magnitudes, directly improving int6 quantization quality by keeping weights closer to zero where the 6-bit grid is densest. Combined with SWA, this is the primary defense against quantization degradation.

7. **Sliding Window Evaluation** (stride=64, batch=32 sequences). Full 2048-token context scoring for every token position. At stride=64, each token gets scored with nearly maximum left-context, recovering ~0.03 BPB over fixed-window eval. Evaluation time: ~280s per seed (included in the 10-minute budget when stride > 0).

8. **FP16 Tied Embedding Passthrough.** Token embeddings stored in fp16 rather than int6, since they're shared with the output projection and quantization noise here has outsized impact on all logits. Costs ~2KB extra but avoids ~0.005 BPB quantization penalty on embeddings.

## Architecture

- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA), MLP mult 3x (1536 hidden)
- 26.5M parameters, tied embeddings, vocab 1024 (sp1024 BPE)
- Training: seq_len=2048, batch=524,288 tokens, ~8000 steps in 600s
- U-Net skip connections with learned skip weights
- RMSNorm, RoPE (base 10000), logit softcap 30.0
- Orthogonal initialization, ReLU-squared MLP activation

## Training Config

| Parameter | Value |
|-----------|-------|
| Matrix LR (Muon) | 0.02 |
| Scalar LR (Adam) | 0.02 |
| Tied Embed LR (Adam) | 0.03 |
| Muon Momentum | 0.99 (warmup 0.92 -> 0.99 over 1500 steps) |
| Muon Weight Decay | 0.038 |
| Warmdown | 3000 steps (wallclock-aware) |
| QAT | int6 STE, active from step 0 |
| SWA | every 50 steps, start at 50% of training |
| Grad Clip | 0.3 |

## Artifact

| Component | Size |
|-----------|------|
| Model (int6 + zstd-22) | 15,992,249 bytes |
| Code | 84,488 bytes |
| **Total** | **16,076,737 bytes** |

## Included Files

| File | Purpose |
|------|---------|
| `README.md` | This document |
| `submission.json` | Metadata, scores, seed results |
| `train_gpt.py` | Training script (single file, all code) |
| `train_seed1337.log` | Full training log for seed 1337 (best) |
| `train_seed7.log` | Full training log for seed 7 |
| `train_seed42.log` | Full training log for seed 42 |

## Command

```bash
NCCL_NVLS_ENABLE=0 \
RUN_ID=v5_11L_smearv2 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 NUM_HEADS=8 NUM_KV_HEADS=4 MODEL_DIM=512 \
NUM_LAYERS=11 MLP_MULT=3 \
QAT=1 QUANT_BITS=6 FP16_EMBED=1 LATE_K_LAYERS=0 \
EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 \
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 MUON_WEIGHT_DECAY=0.038 \
SMEAR_GATE=1 BIGRAM_HASH=0 \
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288 \
WARMDOWN_STEPS=3000 \
MAX_WALLCLOCK_SECONDS=600 \
SWA_ENABLED=1 SWA_START_FRAC=0.5 SWA_EVERY=50 TTT_ENABLED=0 \
TRAIN_LOG_EVERY=50 VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
