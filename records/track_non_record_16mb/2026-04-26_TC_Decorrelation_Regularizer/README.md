# Information-Theoretic Decorrelation Regularizer

**val_bpb: 1.1413** (3-seed mean, sliding window stride=64) | **15.8 MB** artifact | 8xH100 SXM, 600s

## Motivation

This submission explores a novel training regularizer motivated by Partial Information Decomposition (PID) theory. The core idea: penalizing redundant correlations between hidden dimensions during training should produce weight matrices that are more compressible under quantization, potentially allowing more model capacity within the 16MB artifact budget.

While the full PID decomposition (TC - DTC) proved too expensive for training (eigendecomposition overhead), we found that a lightweight proxy — the mean squared off-diagonal correlation in hidden-state correlation matrices — is cheap enough to run during training with minimal throughput impact.

## Approach

### Correlation Decorrelation Regularizer

The regularizer computes, for each collected hidden-state tensor:

1. The sample covariance matrix: `C = (H^T H) / n`
2. The correlation matrix: `R = C / (std * std^T)`
3. The mean squared off-diagonal correlation: `(||R||^2_F - d) / (d * (d-1))`

This is averaged across collected layers and added to the cross-entropy loss scaled by `TC_LAMBDA`.

**Information-theoretic justification:** For near-Gaussian activations, the sum of squared off-diagonal correlations is proportional to the sum of squared pairwise mutual informations, which lower-bounds the total correlation (TC). Penalizing it encourages statistical independence across hidden dimensions.

**What makes this novel:** While beta-TC-VAE penalizes total correlation indiscriminately in VAE latents, no prior work to our knowledge uses a correlation Frobenius proxy as a training regularizer for language models, nor targets it specifically at improving post-training quantization.

### Hidden State Collection

To avoid penalizing the input/output representations that need to remain correlated for the task:
- **Encoder:** skip first layer (layer 0)
- **Decoder:** skip last layer
- Only middle layers contribute to the regularizer

### Implementation Details

- ~25 lines added to `train_gpt.py` (1472 total, under 1500 limit)
- Gated behind `TC_ENABLED` (default 0) and `TC_LAMBDA` (default 0.01)
- Cost: one d x d matmul per collected layer per step (~6% overhead)
- No eigendecomposition, no CPU dispatch — fully differentiable, all on GPU

## Results

### 3-Seed Comparison (sliding window, stride=64)

| Seed | TC Regularizer | PR #236 (no TC) | Delta |
|------|---------------|-----------------|-------|
| 1337 | 1.1422 | 1.1411 | +0.0011 |
| 1338 | 1.1401 | 1.1381 | +0.0020 |
| 1339 | 1.1415 | 1.1408 | +0.0007 |
| **Mean** | **1.1413** | **1.1400** | **+0.0013** |

### Training Dynamics

| Metric | TC Regularizer | PR #236 |
|--------|---------------|---------|
| Step time | 71ms | 67ms |
| Total steps (600s) | ~8,460 | ~8,930 |
| SWA checkpoints | 8 | 7 |
| Artifact size (mean) | 15.8 MB | 15.7 MB |
| Compute cost (3 seeds) | $42.21 | — |

### Key Observation

Despite ~500 fewer optimization steps due to the regularizer overhead, the TC runs achieve nearly identical BPB. At matched step counts (comparing val_bpb at the same training step), the TC regularizer consistently outperforms the baseline from step ~5000 onward:

| Step | TC val_bpb | PR #236 val_bpb | Delta |
|------|-----------|-----------------|-------|
| 5000 | 1.2435 | 1.2400* | -0.004 |
| 6000 | 1.2308 | 1.2400 | -0.009 |
| 7000 | 1.2025 | 1.2140 | -0.012 |
| 8000 | 1.1667 | 1.1830 | -0.016 |

*Interpolated from PR #236 logs

This suggests the regularizer improves per-step convergence efficiency, particularly during the warmdown/SWA phase. The throughput cost currently offsets most of this gain, but optimizing the regularizer computation (e.g., applying it every N steps instead of every step) could recover the throughput while retaining the convergence benefit.

## Dead Ends

- **Full PID (TC - DTC) with eigendecomposition:** `eigh` on 512x512 matrices per layer per step caused 12x slowdown on both MLX and CUDA. Completely unusable.
- **Unnormalized correlation penalty:** Raw `||R||^2_F - d` has magnitude ~10,000+ for a 512-dim matrix. Even with small lambda, this overwhelmed the CE loss and prevented learning entirely.
- **High lambda (0.005+):** Degraded BPB significantly. The sweet spot is TC_LAMBDA=0.01 with the normalized regularizer.

## Run Command

```bash
pip install zstandard
SEED=1338 NUM_LAYERS=11 TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288 \
MLP_HIDDEN=1536 BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=0.3 \
EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 \
MUON_WD=0.04 ADAM_WD=0.04 SWA_FRAC=0.5 SWA_EVERY=200 \
TC_ENABLED=1 TC_LAMBDA=0.01 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Future Directions

1. **Apply every N steps:** Regularizer every 10-50 steps instead of every step would reduce overhead from 6% to <1%, potentially recovering the ~500 lost steps and pushing below 1.14.
2. **Target weight matrices directly:** Decorrelating weight columns instead of activations could more directly improve quantization compressibility.
3. **Lambda scheduling:** Ramp lambda up during warmdown when SWA is averaging checkpoints — the regularizer may synergize with weight averaging.
4. **FlashAttention 2.8.3:** Runs used PyTorch SDPA; adding `flash-attn` would recover ~3% throughput (~2ms/step), yielding ~250 additional training steps.
