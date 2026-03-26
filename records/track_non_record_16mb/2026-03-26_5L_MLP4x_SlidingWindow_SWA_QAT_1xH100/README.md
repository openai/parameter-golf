# Non-Record: 5L MLP4x + SlidingWindow + SWA + QAT (1xH100)

## Score: val_bpb = 1.3827 (post-quant, sliding window eval)

Trained on 1xH100 80GB in 300 seconds (5-minute budget). 14.6MB artifact (int8+zlib). This is a **non-record submission** demonstrating an autonomous AI-driven exploration of 15+ experiments using the autoresearch framework.

## Key Discovery: Width > Depth

The single most impactful finding: **5 layers with MLP 4x expansion (hidden=2048) significantly outperforms deeper, narrower architectures**. Going from 6L MLP3x (val_bpb 1.505) to 5L MLP4x (val_bpb 1.417) gave a -0.088 bpb improvement -- the largest single gain in our exploration.

This is surprising because the baseline and SOTA both favor 9-10 layers with MLP 2-3x. On a single GPU with limited compute, the wider MLP captures more per-token information per step, which compensates for having fewer layers.

## Approach

Seven techniques stacked on the baseline architecture:

### 1. 5-Layer MLP 4x Architecture
5 transformer blocks with MLP expansion factor 4x (hidden=2048). Trades depth for width. Model dim=512, 8 attention heads, 4 KV heads (GQA). U-Net skip connections between encoder/decoder halves.

### 2. BigramHash Embedding (4096 buckets, dim=128)
Hash table mapping adjacent token pairs to learned embeddings via `(prev_token * 92821 + curr_token) % 4096`. Projected to model dim. Adds ~589K parameters for lightweight bigram context.

### 3. SmearGate
Learned per-dimension gate blending each token with the previous token's embedding. Adds ~512 parameters. Complements BigramHash with a soft blending signal.

### 4. Orthogonal Weight Initialization
All weight matrices initialized with `orthogonal_()`. Zero-init for output projections. Matches Muon optimizer's orthogonalization geometry.

### 5. QAT (Quantization-Aware Training)
Int8 fake-quantization during training forward pass with Straight-Through Estimator (STE). Model learns quantization-robust weights, reducing the quant gap to 0.0004 bpb.

### 6. Stochastic Weight Averaging (SWA)
Average 18 checkpoints collected every 50 steps during the warmdown phase (last 50% of training). Produces smoother weight distributions that quantize better.

### 7. Sliding Window Evaluation (stride=64)
Every token scored with near-full context (960+ tokens). Free -0.034 bpb improvement over standard chunked evaluation.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| num_layers | 5 |
| model_dim | 512 |
| mlp_mult | 4.0 (hidden=2048) |
| num_heads | 8 |
| num_kv_heads | 4 |
| train_seq_len | 1024 |
| train_batch_tokens | 131,072 |
| matrix_lr (Muon) | 0.03 |
| embed_lr (Adam) | 0.06 |
| weight_decay | 0.04 |
| muon_momentum | 0.99 (warmup from 0.92 over 500 steps) |
| warmdown_frac | 0.5 |
| grad_clip_norm | 1.0 |
| swa_every | 50 |
| eval_stride | 64 |
| logit_softcap | 30.0 |
| bigram_buckets | 4096 |

## Key Metrics

- **val_bpb** (post-quant): **1.382720**
- **val_bpb** (pre-quant): 1.382291
- **quant_gap**: 0.000428
- **artifact_bytes**: 14,621,542 (1.4MB headroom under 16MB)
- **model_params**: 15.5M
- **training_steps**: 4,573
- **training_time**: 300s (5 min)
- **eval_time**: 236s (sliding window)
- **peak_vram**: 13,339 MB
- **GPU**: 1xH100 80GB HBM3

## Full Experiment Log (Autonomous AI Exploration)

All experiments run autonomously by Claude Code using the autoresearch framework on `autoresearch/runpod` branch.

| # | Description | val_bpb | Delta | Status |
|---|------------|---------|-------|--------|
| 01 | 6L MLP3x + BigramHash + SmearGate + OrthoInit + SWA + QAT | 1.505 | -- | baseline |
| 02 | MATRIX_LR=0.03 (from 0.02) | 1.479 | -0.026 | keep |
| 03 | BigramHash 8192 buckets | 1.535 | +0.056 | discard |
| 04 | WARMDOWN_FRAC=0.3 | 1.592 | +0.113 | discard |
| 05 | WARMDOWN_FRAC=0.7 | 1.539 | +0.060 | discard |
| 06 | MATRIX_LR=0.04 | 1.521 | +0.042 | discard |
| 07 | EMBED_LR=0.08 | 1.486 | +0.007 | discard |
| 08 | 7L MLP3x | 1.493 | +0.014 | discard (>16MB) |
| 09 | 7L dim480 MLP3x | 1.512 | +0.033 | discard |
| 10 | WD=0.06 + SWA/25 | 1.493 | +0.014 | discard |
| 11 | **5L MLP4x** | **1.417** | **-0.062** | **keep** |
| 12 | N-gram mixing + LeakyReLU(0.5)^2 | 1.434 | +0.017 | discard |
| 13 | **Sliding window eval (stride=64)** | **1.383** | **-0.034** | **keep (best)** |
| 14 | MATRIX_LR=0.02 | 1.409 | +0.026 | discard |
| 15 | MATRIX_LR=0.025 | 1.391 | +0.008 | discard |
| 16 | GRAD_CLIP=0.3 | 1.444 | +0.061 | discard |

**Total improvement: -0.122 bpb** (1.505 -> 1.383)

## Methodology: Autonomous AI Experimentation

This submission was produced using **autoresearch**, an autonomous AI research framework where Claude Code iterates on `train_pgolf.py`:
1. Agent proposes a change (architecture, hyperparameter, technique)
2. Commits and runs the experiment (5-min fixed budget)
3. If val_bpb improves: keep (advance branch)
4. If worse: discard (git reset)
5. Loop until stopped

The full experiment history is preserved on the `autoresearch/runpod` branch with individual commits for each experiment.

## Hardware Note

All experiments ran on a single H100 80GB with a 5-minute wallclock cap. With 8xH100 and 10-minute budget (16x more compute), the architectural discoveries (MLP4x, BigramHash, SmearGate, sliding window) should transfer and yield significantly better results.

## Next Steps (with compute credit)

1. Scale to 8xH100 with 10-minute budget
2. Increase batch size to 786K tokens for better gradient estimates
3. Train at seq_len=2048 for longer context
4. Apply int6/int5 quantization to fit more layers (10L+)
5. Run 3 seeds for statistical significance
6. Target sub-1.14 val_bpb (current SOTA)
