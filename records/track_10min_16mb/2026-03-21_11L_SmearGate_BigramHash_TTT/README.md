# 11L SmearGate + BigramHash(10240) + Causal TTT + Mixed Int5/Int6 + SWA

**val_bpb: pending** (awaiting RunPod credits for 8×H100 runs)

## Approach

This submission combines the strongest proven techniques from top leaderboard entries into a unified architecture targeting sub-1.135 val_bpb.

## Run Command

```bash
# Setup (once)
bash prepare.sh

# Train + evaluate
bash eval/eval.sh

# With specific seed
SEED=42 bash eval/eval.sh
```

## Key Techniques

### Architecture: 11-Layer GPT
- 11 layers (vs 10 in current top1), 512 dim, 8 heads, 4 KV heads (GQA)
- U-Net skip connections with learned weights
- ReLU² MLP activation, tied embeddings, logit softcap=30

### SmearGate (from PR #162)
- Learned sigmoid gate blending each token embedding with the previous position
- ~512 parameters, near-zero computational cost
- Captures local context at the embedding level

### BigramHash(10240) (from PR #162, #180)
- Hash consecutive token pairs: `(prev * 31 + curr) % 10240`
- 128-dim embedding → project to 512 via learned linear
- Explicit bigram statistics available to every layer

### Causal Test-Time Training (from PR #267, #281)
- At eval time, process validation data chunk-by-chunk
- Score each chunk FIRST (causal), then fine-tune on it
- No future information leakage — each token scored before model sees it
- SGD with lr=0.002, momentum=0.9, 1 step per chunk
- Expected gain: 3-5 millibpb

### Mixed Int5/Int6 Quantization (from PR #180)
- Int5 for MLP weights (most compressible)
- Int6 for attention/bigram weights (precision-sensitive)
- FP16 for tied embeddings and last-layer key projections
- 3% magnitude pruning for better zlib compressibility

### Z-Loss Regularization
- `1e-4 * logsumexp(logits)²` for logit stability
- Prevents logit explosion during training

### SWA (Stochastic Weight Averaging)
- Snapshot every 50 steps from 40% of training onward
- Average at end for flatter minima

## Training Hyperparameters
- Muon optimizer: matrix_lr=0.025, momentum warmup 0.92→0.99 over 1500 steps
- AdamW: tied_embed_lr=0.035, scalar_lr=0.025
- weight_decay=0.042, grad_clip=0.3
- warmdown=3000 iters, max_wallclock=591s
- batch=524K tokens, seq_len=1024

## Expected Results

Based on ablation data from PRs #162, #180, #267, #281:

| Technique | Expected gain |
|-----------|--------------|
| 11th layer | -0.001 to -0.002 |
| TTT | -0.003 to -0.005 |
| Better WD/LR tuning | -0.001 |
| Cumulative from 1.1428 | **~1.133-1.137** |

## Status

**Pending compute credits.** Submission code is complete and syntax-validated. Awaiting RunPod 8×H100 allocation for 3-seed validation runs.

Built on PR #162 by @unnir and PR #180 by @thwu1.
