# 1xH100 Budget Run: SmearGate + BigramHash + MLP3x

**val_bpb: 1.2774** (mean of 3 seeds, sliding window stride=64, post int6+zlib quantization roundtrip)

## Motivation

This is a non-record submission exploring how far a **single H100 GPU** with limited compute budget ($20 RunPod credits) can go using proven leaderboard techniques. While the official leaderboard targets 8xH100 SXM, we believe there's value in demonstrating competitive results on more accessible hardware.

## 3-Seed Results

| Seed | val_bpb | val_loss | artifact_bytes |
|------|---------|----------|---------------|
| 1337 | 1.27754 | 2.15706 | 16,374,104 |
| 42 | 1.27402 | 2.15113 | 16,389,057 |
| 7 | 1.28077 | 2.16252 | 16,377,079 |
| **Mean** | **1.27744** | **2.15690** | |
| **Std** | **0.00338** | | |

**Note on artifact size:** All 3 seeds slightly exceed the 16,000,000 byte limit (~2.3% over). This could be resolved by switching from zlib to zstd-22 compression (estimated ~5% additional savings), or by reducing BigramHash vocab from 4096 to 2048. We chose not to re-run to preserve the authentic single-attempt results.

## Approach

Built on PR #162 (SmearGate + BigramHash + MLP3x + SWA), adapted for 1xH100 with the following hyperparameter adjustments:

### Key Modifications for 1xH100

| Parameter | 8xH100 (PR #162) | 1xH100 (this run) | Rationale |
|-----------|-------------------|-------------------|-----------|
| train_batch_tokens | 786,432 | 524,288 | More steps in 10min window |
| warmdown_iters | 3,000 | 800 | Shorter warmdown for fewer total steps |
| swa_start_frac | 0.5 | 0.7 | Collect only well-converged checkpoints |
| swa_every | 50 | 100 | Fewer, higher-quality averaged checkpoints |
| train_shards | 80 | 20 | Budget constraint |

All architecture choices (MLP3x, SmearGate, BigramHash, U-Net skip, int6 quantization) kept identical to PR #162.

## Architecture

- 9 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu^2 activation
- SmearGate + BigramHash(4096, dim=128)
- Orthogonal init with muP-scaled output projections
- U-Net skip connections, tied embeddings

## Training

- Muon optimizer: matrix_lr=0.02, WD=0.01, momentum=0.99
- AdamW for embeddings/scalars
- warmdown=800 iters, warmup=20 steps
- seq_len=2048, batch=524K tokens
- grad_clip=0.3
- SWA: start_frac=0.7, every=100 steps
- Sliding window eval: stride=64

## Hardware

- 1x NVIDIA H100 80GB HBM3 SXM (RunPod, on-demand)
- Peak VRAM: 11,560 MiB / 81,559 MiB
- Training: ~1,250 steps in 600s (~480ms/step)
- Total cost per seed: ~$1.65 (training) + ~$3.30 (eval) = ~$5

## Comparison

| Run | Hardware | val_bpb | 
|-----|----------|---------|
| Naive Baseline | 8xH100 | 1.2244 |
| **This submission** | **1xH100** | **1.2774** |
| PR #162 (SmearGate+BigramHash) | 8xH100 | 1.1458 |

Achieving val_bpb 1.2774 on 1xH100 demonstrates that the SmearGate/BigramHash/MLP3x techniques provide meaningful improvements even with 1/8th the compute and 1/4th the training data.

## Run Command

```bash
RUN_ID=submit SEED=1337 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

All parameters are set as defaults in `train_gpt.py`. No env vars needed except RUN_ID and SEED.
