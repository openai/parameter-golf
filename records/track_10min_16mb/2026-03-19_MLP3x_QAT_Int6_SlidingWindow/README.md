# 11L MLP3x + WD=0.04 + Int6 QAT + zstd-22 + Sliding Window Eval

## Summary

11-layer transformer with 3x MLP expansion, int6 quantization-aware training, decoupled weight decay (0.04), zstd-22 compression, and sliding window evaluation. This achieves **val_bpb = 1.1502** (mean across 3 seeds).

### Key Changes from Baseline

1. **11 transformer layers** (vs 9 baseline) — more effective depth, funded by aggressive int6 compression
2. **Wider MLP (MLP_MULT=3)** — 3x expansion (hidden=1536), more capacity per layer
3. **Decoupled weight decay (0.04)** — on both Muon and AdamW, keeps weights small and quantization-friendly
4. **QAT int6** — STE fake-quantize simulates int6 noise during training
5. **Int6 quantization on all block weights** (layers 0-10)
6. **FP16 tied embedding export** — preserves embedding/output head quality
7. **zstd-22 compression** — saves ~1.5MB vs zlib, critical for fitting 11L MLP3x under 16MB
8. **Sliding window evaluation (stride=64)** — ~0.034 BPB free improvement
9. **Higher Muon momentum (0.99)** with warmup from 0.92 over 1500 steps
10. **Lower learning rates**: MATRIX_LR=0.025, SCALAR_LR=0.025, TIED_EMBED_LR=0.035

### Architecture

- 11 transformer blocks, 512 model dim, 8 attention heads, 4 KV heads
- GQA attention with RoPE, ReLU² MLP (**3x** expansion)
- Tied embeddings with 1024 BPE vocabulary
- U-Net skip connections (5 encoder + 6 decoder layers)
- 26.5M parameters, ~15.4MB compressed artifact (zstd-22)

## Multi-Seed Results (3 seeds, p << 0.001)

| Seed | slide_loss (nats) | slide_bpb | rt_bpb | Artifact |
|---|---|---|---|---|
| 1337 | 1.94265607 | 1.15055135 | 1.18484075 | 15,360,260 |
| 42 | 1.94207795 | 1.15020896 | 1.18456681 | 15,556,813 |
| 123 | 1.94121940 | 1.14970047 | 1.18421993 | 15,365,293 |
| **Mean** | **1.94198447** | **1.15015359** | **1.18454250** | **15,427,455** |
| **Std** | **0.00072288** | | | |

- **Mean improvement: 0.1307 nats** over baseline
- **t-statistic: 313.20** (df=2, p << 0.001)
- All 3 artifacts under 16MB
- Sliding window eval takes ~88s on 8xH100 (under 10-min eval budget)

## Hardware

All runs on 8×H100 SXM (RunPod). ~10,070 training steps at ~59.6ms/step in 600s.

## How to Run

Requires `zstandard` package (`pip install zstandard`).

```bash
RUN_ID=submission \
SEED=1337 \
NUM_LAYERS=11 \
MLP_MULT=3 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
FP16_EMBED_EXPORT=1 \
INT6_LAYER_START=0 \
INT6_LAYER_END=10 \
QAT_ENABLED=1 \
QAT_INT6=1 \
MUON_WEIGHT_DECAY=0.04 \
ADAM_WEIGHT_DECAY=0.04 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 \
USE_ZSTD=1 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
