# EBLS: Empirical Bayes Layer Sharing (Non-Record Submission)

**Author:** Robby Sneiderman ([@Robby955](https://github.com/Robby955))

**BPB:** 1.3441 (post-quantization) | 1.2105 (pre-quantization, beats 1.2244 baseline)

This is a non-record submission exploring a novel architecture direction: using James-Stein shrinkage estimators to learn optimal layer-sharing patterns in compressed transformers.

## Approach

Three shared transformer blocks are each applied 3 times (9 effective layers), with per-virtual-layer LoRA deviations (rank 8) gated by learned shrinkage factors:

```
W_effective[i] = W_shared + gamma_i * A_i @ B_i
```

where `gamma_i = sigmoid(logit_i)` is optimized jointly with model weights. A regularization penalty `lambda * sum(gamma_i)` encourages sharing unless deviation genuinely helps — analogous to the James-Stein estimator shrinking individual estimates toward the grand mean.

## Key Findings

### 1. MLP-vs-Attention Sharing Asymmetry

After training on 8xH100 (4572 steps), the learned gammas show:

| Component | Gamma Range | Interpretation |
|-----------|------------|----------------|
| MLP (all layers) | 0.0000 | Fully shared — identical computation across depth |
| Attention (layers 0-2) | 0.001-0.005 | Trace specialization in early layers only |
| Attention (layers 3-8) | 0.0000 | Fully shared |

**MLP weights converge to exact sharing.** The model discovers through gradient optimization that feedforward computation does not need to vary with depth under compression constraints. This provides empirical evidence for hard-sharing decisions made by intuition in other submissions.

### 2. Quantization Error Amplification in Depth-Recurrent Architectures

EBLS reveals a fundamental limitation of shared-block architectures: quantization error compounds multiplicatively through repeated application. We observe a 0.19 BPB gap between `torch.compile` (fused kernels) and eager-mode evaluation — not from quantization, but from floating-point numerical differences amplified across 15 passes through 5 shared blocks. This gap exists even without QAT and persists regardless of quantization scheme.

This finding has implications beyond this challenge: any architecture using weight sharing with depth recurrence (Universal Transformer, ALBERT-style) will exhibit amplified sensitivity to numerical precision.

### 3. LoRA Rank Threshold for Specialization

At rank 8, all gammas converge to ~0 (no specialization needed). At rank 16, gammas reach 0.01-0.05 — the model uses the additional capacity for mild deviation. This suggests an interesting capacity-sharing tradeoff: lower LoRA rank forces the model to decide more aggressively between sharing and specialization.

## Architecture Details

- 1024-dim, 16 heads, 4 KV heads, mlp_mult=3
- BigramHash(10240 buckets, 128-dim), SmearGate
- Int6 STE QAT, zstd-22 compression
- SWA (9 checkpoints), Muon optimizer (WD=0.04)
- Orthogonal initialization

## Why Not Competitive

The 1024-dim model trains at 131ms/step (vs 43ms baseline), limiting total steps to ~4500 in 10 minutes vs ~13,000 for the baseline. Combined with the quantization amplification gap, post-quant BPB (1.34) falls short of competitive entries despite pre-quant BPB (1.21) beating the baseline.

## Reproducing

```bash
# 8xH100 SXM, 10-minute wallclock
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Full Writeup

For the statistical foundations connecting James-Stein shrinkage to neural network parameter sharing, see the companion repository: [github.com/Robby955/parameter-golf-ebls](https://github.com/Robby955/parameter-golf-ebls)
