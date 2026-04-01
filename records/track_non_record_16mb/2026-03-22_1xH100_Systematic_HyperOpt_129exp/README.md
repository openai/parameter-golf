# 8L + BigramHash Projection + Batch Scaling + Systematic HyperOpt (129 Experiments)

**val_bpb: 1.2392** | 15.9 MB artifact | 1x H100 SXM, 10 minutes

## Approach

Systematic hyperparameter optimization across 129 experiments ($19.47 total compute), starting from the baseline architecture and progressively optimizing for single-GPU throughput-constrained training. Key contributions: (1) demonstrating how batch size, learning rate, and weight decay must co-scale with GPU speed, model depth, and step count, (2) BigramHash with dimensionality projection for artifact size reduction, (3) weight decay as an artifact compression control knob.

## Run Command

```bash
cd records/track_non_record_16mb/2026-03-22_1xH100_Systematic_HyperOpt_129exp/
NUM_LAYERS=8 MLP_MULT=4 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 \
EVAL_STRIDE=512 TRAIN_BATCH_TOKENS=131072 GRAD_ACCUM_STEPS=1 \
MAX_WALLCLOCK_SECONDS=600 WARMDOWN_ITERS=3000 \
MATRIX_LR=0.03 SCALAR_LR=0.03 TIED_EMBED_LR=0.08 \
MUON_WD=0.048 GRAD_CLIP_NORM=0.3 QK_GAIN_INIT=1.0 \
BIGRAM_BUCKETS=12288 BIGRAM_DIM=128 ROPE_BASE=50000 \
LOGIT_SOFTCAP=30 MUON_BACKEND_STEPS=5 \
python train_gpt.py
```

## Architecture

- 8 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 4x MLP expansion with relu^2 activation
- BigramHash 12288 buckets with **dim=128 linear projection** (saves 3.7MB vs full-dim)
- RoPE with base=50000 (optimized for 2048 context)
- Overtone embedding init + phase-transition residual mixing
- FP16 tied embeddings + logit softcap=30

## Training

- **Muon optimizer** with decoupled weight decay
- matrix_lr=0.03, scalar_lr=0.03, embed_lr=0.08
- **TRAIN_BATCH_TOKENS=131072** (2x default -- key H100 optimization)
- GRAD_ACCUM_STEPS=1 (no accumulation needed)
- WARMDOWN_ITERS=3000
- Muon WD=0.048, grad_clip=0.3
- 4,486 steps in 600s (~134ms/step on 1x H100 SXM)

## Key Findings from 129 Experiments

### Hyperparameter Scaling Laws

1. **LR scales inversely with step count**: 2k steps -> LR=0.06, 4.5k steps -> LR=0.03, 8k steps -> LR=0.025
2. **WD controls artifact size linearly**: WD=0.03 -> 17.0MB, WD=0.04 -> 16.3MB, WD=0.05 -> 15.7MB, WD=0.06 -> 15.2MB (8L model)
3. **Optimal depth scales with GPU speed**: RTX 5070 Ti (135ms/step) -> 5 layers. H100 SXM (74-134ms/step) -> 7-8 layers.
4. **Batch size is a major lever on fast GPUs**: 131K tokens (2x default) gave -0.027 BPB vs 65K. Each step sees 2x data, compensating for fewer total steps.

### Architecture Findings

5. **BigramHash dim=128 projection**: Saves 3.7MB artifact space with <0.002 BPB quality loss. Enables deeper models within 16MB.
6. **relu^2 > SwiGLU** in throughput-limited regime (fewer FLOPs per token)
7. **4x MLP > 3x MLP** when artifact budget allows (9L 3x = worse than 8L 4x)
8. **BIGRAM_BUCKETS=12288 > 8192**: Marginal improvement (0.0003 BPB)

### Negative Results (saves others time)

9. SmearGate: throughput penalty outweighs quality gain on single GPU
10. EMA: catastrophic for short training (<5k steps)
11. SWA: no benefit at <7k steps (needs 30k+)
12. Orthogonal init: hurts at <5k steps
13. Label smoothing: even 0.01 hurts
14. Magnitude pruning: barely helps int8+zlib compression
15. zstd-22: inconsistent vs zlib-9 (sometimes worse due to different weight distributions)
16. All alt activations (leaky_relu^2, x*abs(x), mish^2): all worse than relu^2
17. 96K batch: worse than both 65K and 131K (awkward middle ground)

## Improvement Trajectory

| Stage | val_bpb | Improvement |
|-------|---------|-------------|
| Baseline (9L, 5070 Ti) | 1.4691 | -- |
| SOTA arch + higher LRs | 1.4613 | -0.008 |
| Layer optimization (5L) | 1.3997 | -0.069 |
| BigramHash + RoPE 50k | 1.3873 | -0.082 |
| 10 min training | 1.3380 | -0.131 |
| H100 + 6L + LR tuning | 1.2701 | -0.199 |
| H100 + 7L + 131K batch | 1.2417 | -0.227 |
| **H100 + 8L + BIGRAM 12288** | **1.2392** | **-0.230** |

## Notes

This is a **non-record submission** run on 1x H100 SXM (not 8xH100). The primary contribution is the systematic exploration of hyperparameter scaling laws for throughput-constrained training across 129 experiments, and the BigramHash projection technique for artifact size reduction. With 8xH100 compute, we plan to scale to 10-11 layers with INT6 mixed quantization, Test-Time Training (TTT), and SWA at 30k+ steps.

## Hardware

- 1x NVIDIA H100 SXM 80GB (Vast.ai, Iowa)
- Training time: 600s (10 min wallclock)
- Peak VRAM: 3,259 MiB
- Step time: ~134ms average
- Total steps: 4,486
- Total compute cost: $19.47 across 129 experiments
