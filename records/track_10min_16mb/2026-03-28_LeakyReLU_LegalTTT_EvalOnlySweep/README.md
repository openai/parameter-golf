# LeakyReLU² + 4ep Legal TTT + Parallel Muon

**val_bpb: 1.1189** (3-seed mean, std 0.0006; best 1.11835341) | **~15.88 MB** | 8xH100 SXM

## Results

| Seed | step_avg | steps | Post-EMA bpb | **Post-TTT bpb** | TTT time | Artifact |
|------|----------|-------|--------------|------------------|----------|----------|
| 1337 | 83.9ms | 7,000 | 1.1370 | **1.11903472** | 548.2s | 15,869,615 |
| 42 | 84.0ms | 7,000 | 1.1374 | **1.11944510** | 541.6s | 15,866,975 |
| 2025 | 83.8ms | 7,000 | 1.1362 | **1.11835341** | 545.4s | 15,882,595 |
| **Mean** | **83.9ms** | **7,000** | **1.1369** | **1.1189 (std 0.0006)** | **~545s** | |

## What Changed

This submission is a minimal improvement on `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`:

- `TTT_LR`: `0.002 -> 0.0025`
- `TTT_EPOCHS`: `3 -> 4`
- Added `SKIP_PRE_TTT_EVALS=1` to skip diagnostic int6 roundtrip + sliding-window evals before TTT, keeping total eval under the 10 minute limit
- Added `EVAL_ONLY_MODEL_PATH` / `EVAL_ONLY_INT6_PATH` for fast TTT sweeps without retraining

The underlying training stack is unchanged: 11 layers, LeakyReLU(0.5)^2, BigramHash(1536), XSA in the last 4 layers, partial RoPE, VE128, EMA, tight SWA, GPTQ-lite int6, and Parameter Banking + Parallel Muon.

## Legal TTT Protocol

Backward-looking, score-first TTT following PR #461's framework:

1. Validation tokens are split into 1,893 non-overlapping 32K-token chunks
2. For each chunk:
   - **SCORE**: Sliding-window eval under `torch.inference_mode()`
   - **TRAIN**: SGD(lr=0.0025, momentum=0.9) on the already-scored chunk for 4 epochs, all blocks unfrozen, cosine LR decay, grad clip 1.0
3. The last chunk is scored but never trained on
4. Chunk N is scored by a model adapted only on chunks 0..N-1

## Timing Budget

| Phase | Time |
|-------|------|
| Training | 600s |
| Legal TTT eval | 541.6s - 548.2s |
| Total eval wall time | ~550s |

## Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.0025 TTT_EPOCHS=4 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
SKIP_PRE_TTT_EVALS=1 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=2025 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Sweep Notes

Eval-only sweeps on the seed 2025 checkpoint:

| TTT_LR | TTT_EPOCHS | GPUs | legal_ttt_exact |
|--------|------------|------|-----------------|
| 0.0015 | 3 | 8 | 1.11909194 |
| 0.0020 | 3 | 8 | 1.11902385 |
| 0.0022 | 3 | 4 | 1.11896226 |
| 0.0025 | 3 | 4 | 1.11887529 |
| 0.0028 | 3 | 4 | 1.11879319 |
| 0.0025 | 4 | 4 | 1.11866320 |
| 0.0025 | 4 | 8 | 1.11866592 |
| 0.0025 | 4 + skip pre-evals | 8 | 1.11866707 |
| 0.0025 | 4 + skip pre-evals + full retrain | 8 | **1.11835341** |

## Credits

- **Base submission**: [2026-03-23_LeakyReLU_LegalTTT_ParallelMuon](../2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md) by @abaybektursun
- **LeakyReLU² activation**: PR #493 by @parinzee, PR #518 by @sofiabod
- **Optimizer (Parameter Banking + Parallel Muon)**: PR #399 by @abaybektursun
- **TTT recipe**: PR #461 by @Christopher-Lee-McClendon
- **Base model**: PR #414 by @signalrush
