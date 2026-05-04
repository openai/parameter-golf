# 11L Int6 QAT + SmearGate + SWA(0.4) + WD=0.04

## Summary

**Mean val_bpb = 1.1488** (3-seed verified)

Builds on @baudrillardsgh0st's technique stack (PR #194). Two hyperparameter changes informed by systematic analysis across 30+ experiments:

1. **Muon weight decay 0.04** (vs 0.038) — improves int6 quantization quality
2. **SWA start fraction 0.4** (vs 0.5) — captures more checkpoint diversity for smoother weight averaging

| Seed | val_loss | val_bpb | Steps | ms/step | Artifact |
|------|----------|---------|-------|---------|----------|
| **42** | 1.9387 | **1.1482** | 8,393 | 71.79 | 15,177,275 |
| 7 | 1.9399 | 1.1489 | 8,380 | 71.59 | 15,203,710 |
| 1337 | 1.9407 | 1.1494 | 8,390 | 71.51 | 15,331,034 |
| **Mean** | **1.9398** | **1.1488** | | | |

**Std: 0.0006** — 4× tighter than SOTA's 0.0024.

## Key finding

SWA_START_FRAC=0.4 captures ~33% more checkpoints than 0.5, producing smoother weight distributions that survive int6 quantization better. Combined with WD=0.04 (which keeps weight magnitudes smaller), the quantization gap is minimized.

## Architecture

- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- 3× MLP (1536 hidden), vocab 1024, seq len 2048
- Tied embeddings (FP16 passthrough), RoPE, logit softcapping (30.0)
- Per-dim SmearGate, U-Net skip connections

## Training

- Muon optimizer: LR=0.02, momentum=0.99, WD=0.04
- AdamW for embeddings: LR=0.03
- Warmdown: 3000 iterations
- SWA: every 50 steps from 40% of training (~30 checkpoints averaged)
- Int6 QAT with STE throughout training
- Sliding window eval: stride=64, batch_seqs=32

## Compression

- Int6 per-row symmetric quantization in int8 containers
- FP16 passthrough for tied embeddings
- zstd level 22 compression
- All artifacts under 16MB (max 15.3MB)

## Experiment context

This submission is the result of 30+ experiments across 1×H100 and 8×H100 configurations, including:
- Warmdown schedule sweeps (150-700 iterations)
- Learning rate tuning (0.02-0.06)
- Progressive quantization variants (Int8→Int7→Int6, reverse, wave patterns — all worse than constant Int6)
- Int5-MLP asymmetric quantization (worse due to torch.compile overhead)
- Auxiliary multi-token prediction heads (marginal benefit)
- Architecture sweeps (9-12 layers, 384-512 dim)

Total compute cost: ~$100 RunPod spend across all experiments.

## Run command

```bash
NCCL_NVLS_ENABLE=0 \
VOCAB_SIZE=1024 NUM_HEADS=8 NUM_KV_HEADS=4 MODEL_DIM=512 \
NUM_LAYERS=11 MLP_MULT=3 \
QAT=1 QUANT_BITS=6 FP16_EMBED=1 \
EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 \
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 MUON_WEIGHT_DECAY=0.04 \
SMEAR_GATE=1 BIGRAM_HASH=0 \
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288 \
WARMDOWN_ITERS=3000 MAX_WALLCLOCK_SECONDS=600 \
SWA_ENABLED=1 SWA_START_FRAC=0.4 SWA_EVERY=50 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Submission checklist

- [x] 3-seed verification (mean val_bpb=1.1488, std=0.0006)
- [x] All artifacts < 16MB (max 15.3MB)
- [x] Wallclock < 600s on 8×H100
- [x] Train logs included (3 seeds)
- [x] Reproducible train_gpt.py included
- [x] submission.json with metadata
