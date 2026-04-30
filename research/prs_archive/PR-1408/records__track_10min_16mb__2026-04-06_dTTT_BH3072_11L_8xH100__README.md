# dTTT + BigramHash 3072×112

**val_bpb: 1.0800** (3-seed mean, std 0.0002) | ~15.9 MB | 8×H100 SXM

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | Steps | step_avg | post-dTTT bpb | post-GPTQ bpb | sliding window bpb | Artifact |
|------|-------|----------|--------------|---------------|-------------------|----------|
| 1337 | 6916 | 86.77ms | 1.0940 | 1.09941 | **1.08017** | 15,873,363 |
| 42   | 6909 | 86.86ms | 1.0939 | 1.09926 | **1.07980** | 15,895,227 |
| 2025 | 6907 | 86.88ms | 1.0939 | 1.09938 | **1.08018** | 15,865,471 |
| **Mean** | ~6911 | 86.84ms | 1.0939 | 1.09935 | **1.0800 (std 0.0002)** | |

Delta vs PR #1351 (1.0807, best prior open PR): **−0.0007 bpb**

## What Changed

Builds directly on PR #1351 (Discriminative TTT) with one modification:

1. **BigramHash 3072×112** (up from 2048×128 in PR #1351). More expressive n-gram context features — 3072×112 follows PR #1019 and PR #1405 best practices for the current architecture.

All other hyperparameters are identical to PR #1351: dTTT 10 epochs, AdamW LR=0.0005, freeze=0, per-block LR scaling 0.3×→1.0×, cosine decay, GPTQ int6 damp=0.005, QK_GAIN=5.0, WARMDOWN=4000, XSA all-layers, ROPE_DIMS=16.

## Discriminative TTT (from PR #1351)

Pre-quantization AdamW TTT with per-block adaptive learning rates. Each transformer block receives a learning rate scaled by its depth:
- Block 0 (earliest): 0.3× base LR
- Block 10 (latest): 1.0× base LR
- Intermediate: linear interpolation

This concentrates adaptation capacity in later (more context-specific) blocks while preserving learned general features in earlier blocks.

## Pipeline

| Phase | Time |
|-------|------|
| Training (8×H100, 600s wall) | ~6910 steps, 86.8ms/step |
| Pre-quant dTTT (10 epochs, AdamW) | ~186s |
| GPTQ int6 quantization + lzma | ~23s |
| Sliding window eval (stride=64) | ~100s |

## Compliance (Track A — Fixed Predictor)

- No eval-time adaptation of any kind
- Score-first ordering: standard autoregressive sliding-window eval
- No n-gram cache or external data at eval
- Single left-to-right pass

## Architecture Stack

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV GQA) |
| XSA | All 11 layers |
| BigramHash | **3072×112** (projected to 512d) |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| VE | Dim=128, layers 9–10 |
| Weight avg | SWA every 50 steps |
| QK Gain | 5.0 (learnable scalar init) |
| Quantization | Full Hessian GPTQ int6, damp=0.005, lzma |
| Optimizer | Parallel Muon, WARMDOWN=4000 |
| **Pre-quant TTT** | **AdamW, 10ep, freeze=0, per-block LR 0.3×–1.0×** |

## Run Command

```bash
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 \
ETLB_ENABLED=0 \
TTT_ENABLED=1 TTT_LR=0.0005 TTT_EPOCHS=10 TTT_BATCH_SEQS=32 \
TTT_FREEZE_BLOCKS=0 TTT_GRAD_CLIP=1.0 TTT_COSINE_DECAY=1 \
XSA_LAST_N=11 EMA_ENABLED=0 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=4000 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
QK_GAIN_INIT=5.0 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- Discriminative TTT (per-block adaptive LR): PR #1351 by @Christopher-Lee-McClendon and @MatoTeziTanka
- Full Hessian GPTQ + XSA-all + BigramHash: PR #1019
- BigramHash 3072×112 config: PR #1405 by @jfprincz
- Base architecture: community stack via PR #1019
