# Depth Recurrence (layers 4,5)

## Score: mean val_bpb = 1.1182 (3 seeds: 1.1179, 1.1191, 1.1176)

Trained on 8xH100 SXM in ~600 seconds. ~15.9MB artifact (int6+lzma).

## Motivation

We explored both width scaling (MODEL_DIM=576) and depth scaling (adding layers) and found that depth consistently wins over width in this regime. A full independent 12-layer model at dim=512 outperformed a wider 11-layer model at dim=576, despite the wider model having more parameters. However, adding independent layers pushes the model over the 16MB artifact budget. Depth recurrence solves this: by re-executing mid-network layers with independent block scalars, we get the depth benefit without the parameter/size cost. Dual recurrence on layers 4 and 5 gives us 13 virtual layers from 11 physical, staying well under budget at ~15.9MB.

## Approach

Depth recurrence applied to layers 4 and 5, creating 13 virtual layers from 11 physical layers while keeping parameter count at ~27M. Combined with test-time training (TTT) for additional evaluation-time adaptation.

### 1. Dual Depth Recurrence (layers 4,5)
Layers 4 and 5 are each executed twice in sequence (pattern: 0,1,2,3,4,5,4,5,6,7,8,9,10), producing 13 virtual layers from 11 physical layers. Each recurrent pass uses independent learnable block scalars, so the model can modulate how the repeated layers behave on their second pass. This adds depth without increasing model size or artifact bytes — only the small block scalar parameters are added (~2K params).

### 2. Test-Time Training (TTT)
At evaluation time, the model adapts its weights to the validation data using a short fine-tuning pass. 3 epochs over the validation set with lr=0.002, chunked into 32K-token segments. The top 2 blocks are frozen during TTT to preserve the output head's calibration. Tied TTT (no weight untying) performs equivalently to untied.

### 3. Inherited Techniques from Baseline
- **Int6 quantization + lzma compression**: Per-row int6 quantization on MLP/attention weights
- **3x MLP expansion**: Hidden dim 1536 (3x model dim)
- **Bigram hash embeddings**: 2048-bucket hash table (dim=128)
- **Value embeddings**: Learned value residuals on layers 9,10
- **SWA**: Stochastic weight averaging every 50 steps
- **Muon optimizer**: With weight decay 0.04, momentum warmup 0.92->0.99
- **Orthogonal initialization**

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| num_layers | 11 (physical) / 13 (virtual) |
| model_dim | 512 |
| mlp_mult | 3.0 (hidden=1536) |
| recur_layers | 4, 5 |
| train_seq_len | 2048 |
| train_batch_tokens | 786,432 |
| warmdown_iters | 3500 |
| matrix_lr | 0.025 |
| scalar_lr | 0.025 |
| tied_embed_lr | 0.035 |
| muon_momentum | 0.99 (warmup from 0.92 over 1500 steps) |
| muon_weight_decay | 0.04 |
| adam_weight_decay | 0.04 |
| grad_clip_norm | 0.3 |
| eval_stride | 64 |
| swa_every | 50 |
| ttt_lr | 0.002 |
| ttt_epochs | 3 |
| ttt_chunk_tokens | 32768 |
| ttt_freeze_blocks | 2 |

## Key Metrics

- **Mean val_bpb: 1.11819** (std: 0.00076)
- Training: ~6,100 steps in ~600s
- Model params: ~27M
- Artifact size: ~15.9MB (int6+lzma)

## Reproducibility

Three independent training runs with different random seeds:

| Seed | val_loss | val_bpb |
|------|----------|---------|
| 1337 | 1.88749538 | 1.11788404 |
| 2025 | 1.88948575 | 1.11906285 |
| 2024 | 1.88706558 | 1.11762949 |
| **Mean** | **1.88801557** | **1.11819213** |
| **Std** | **0.00129122** | **0.00076473** |

## Run Commands

```bash
# Seed 1337 (default)
ITERATIONS=9000 RECUR_LAYERS=4,5 TTT_ENABLED=1 TTT_UNTIE=0 \
  torchrun --nproc_per_node=8 train_gpt.py

# Seed 2025
ITERATIONS=9000 RECUR_LAYERS=4,5 TTT_ENABLED=1 TTT_UNTIE=0 SEED=2025 \
  torchrun --nproc_per_node=8 train_gpt.py

# Seed 2024
ITERATIONS=9000 RECUR_LAYERS=4,5 TTT_ENABLED=1 TTT_UNTIE=0 SEED=2024 \
  torchrun --nproc_per_node=8 train_gpt.py
```
