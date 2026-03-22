# 7L + BigramHash Projection + Batch Scaling + Systematic HyperOpt

**val_bpb: 1.2417** | 15.5 MB artifact | 1x H100 SXM, 10 minutes

## Approach

Systematic hyperparameter optimization across 111 experiments, starting from the baseline architecture and progressively optimizing for single-GPU throughput-constrained training. Key contribution: demonstrating how batch size, learning rate, and weight decay must co-scale with GPU speed and step count.

## Architecture

- 7 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 4x MLP expansion with relu^2 activation
- BigramHash 8192 buckets with **dim=128 linear projection** (saves 3.7MB vs full-dim)
- RoPE with base=50000 (optimized for 2048 context)
- Overtone embedding init + phase-transition residual mixing
- FP16 tied embeddings + logit softcap=30

## Training

- **Muon optimizer** with decoupled weight decay
- matrix_lr=0.035, scalar_lr=0.035, embed_lr=0.09
- **TRAIN_BATCH_TOKENS=131072** (2x default — key H100 optimization)
- GRAD_ACCUM_STEPS=1 (no accumulation needed)
- WARMDOWN_ITERS=3500 (entire training in warmdown)
- Muon WD=0.025, grad_clip=0.3
- 4,781 steps in 600s (~126ms/step on 1x H100 SXM)

## Key Findings from 111 Experiments

### Hyperparameter Scaling Laws (Novel)

1. **LR scales inversely with step count**: 2k steps → LR=0.06, 5k steps → LR=0.035, 8k steps → LR=0.03
2. **WD scales inversely with step count**: 2k steps → WD=0.04, 5k steps → WD=0.025, 8k steps → WD=0.02
3. **Batch size is a major lever on fast GPUs**: 131K tokens (2x default) gave -0.024 BPB. Each step sees 2x data, compensating for fewer total steps.
4. **WD improves compression**: Higher WD → smaller weight magnitudes → better int8+zlib ratio (15.5MB vs 17.2MB for identical model at WD=0.01)

### Architecture Findings

5. **Optimal depth scales with GPU speed**: RTX 5070 Ti (135ms/step) → 5 layers optimal. H100 SXM (74-126ms/step) → 7 layers optimal.
6. **BigramHash dim=128 projection**: Saves 3.7MB artifact space (10.8MB vs 14.5MB) with <0.002 BPB quality loss. Enables deeper models within 16MB.
7. **relu^2 > SwiGLU** in throughput-limited regime (fewer FLOPs per token)
8. **4x MLP > 3x MLP** when artifact budget allows

### Negative Results

9. SmearGate: throughput penalty outweighs quality gain on single GPU
10. EMA: catastrophic for short training (<5k steps) — averages in noisy early weights
11. SWA: no benefit at <7k steps
12. Orthogonal init: hurts at <5k steps (overwrites beneficial Kaiming + overtone init)
13. Label smoothing: even 0.01 hurts — model needs all signal
14. Magnitude pruning: barely helps int8+zlib compression (zeros don't compress much better after int8)

## Improvement Trajectory

| Stage | val_bpb | Improvement |
|-------|---------|-------------|
| Baseline (9L, 5070 Ti) | 1.4691 | — |
| SOTA arch + higher LRs | 1.4613 | -0.008 |
| Layer optimization (5L) | 1.3997 | -0.069 |
| BigramHash + RoPE 50k | 1.3873 | -0.082 |
| 10 min training | 1.3380 | -0.131 |
| H100 + 6L + LR tuning | 1.2701 | -0.199 |
| H100 + 7L + 131K batch | **1.2417** | **-0.227** |

## Reproducibility

```bash
cd records/track_non_record_16mb/2026-03-22_1xH100_HyperOpt_BigramProj/
NUM_LAYERS=7 MLP_MULT=4 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 \
EVAL_STRIDE=512 TRAIN_BATCH_TOKENS=131072 GRAD_ACCUM_STEPS=1 \
MAX_WALLCLOCK_SECONDS=600 WARMDOWN_ITERS=3500 \
MATRIX_LR=0.035 SCALAR_LR=0.035 TIED_EMBED_LR=0.09 \
MUON_WD=0.025 GRAD_CLIP_NORM=0.3 QK_GAIN_INIT=1.0 \
BIGRAM_BUCKETS=8192 BIGRAM_DIM=128 ROPE_BASE=50000 \
LOGIT_SOFTCAP=30 MUON_BACKEND_STEPS=5 \
python train_gpt.py
```

## Notes

This is a **non-record submission** run on 1x H100 SXM (not 8xH100). The primary contribution is the systematic exploration of hyperparameter scaling laws for throughput-constrained training, and the BigramHash projection technique for artifact size reduction. We expect significant further improvement when scaling to 8xH100 (8x throughput = deeper models + more steps + INT6 QAT + SWA).

## Hardware

- 1x NVIDIA H100 SXM 80GB (Vast.ai, Iowa)
- Training time: 600s (10 min wallclock)
- Peak VRAM: 3,298 MiB
- Step time: ~126ms average
- Total steps: 4,781
