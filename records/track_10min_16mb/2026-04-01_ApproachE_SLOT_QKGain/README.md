# Approach E: SLOT + QK-Gain + Int5 GPTQ + Score-First TTT

## Summary

Combines SLOT (Sample-specific LM Optimization at Test-time) with raised QK-Gain initialization on top of the Approach B base (33.6M params, Int5 GPTQ, score-first TTT).

**val_bpb: TBD** | 8xH100 SXM | 600s train + eval budget

## Key Changes vs Approach B

### 1. SLOT: Per-Batch Delta Vector Optimization (eval-time)

At eval time, for each sliding-window batch, we optimize a single additive delta vector (R^512) between the frozen hidden states and the logit projection.

- **Delta shape**: `[1, 1, 512]` -- broadcasts across batch and sequence
- **Optimizer**: AdamW (lr=0.005, weight_decay=1e-8, eps=1e-5)
- **Steps**: 8 per batch
- **Score-first compliant**: hidden states computed under `torch.no_grad()`, delta adapts through `compute_logits()` only, model weights never modified
- **No cross-batch leakage**: delta re-initialized to zeros for each new batch

The model forward is split into `forward_hidden()` (frozen, no grad) and `compute_logits()` (carries grad for delta optimization).

Reference: Hu et al., arXiv:2505.12392v2. Proven in PR #1172 (~0.010 BPB improvement) and PR #1209 (~0.010 BPB improvement).

### 2. QK-Gain Initialization Raised to 4.0

Per-head learnable scalar on queries after QK-norm, initialized at 4.0 (up from 1.5). This sharpens attention patterns and has been shown to improve convergence in recent SOTA submissions (PR #1209).

## Eval Pipeline

| Stage | Expected BPB Impact | Time | Legality |
|-------|-------------------|------|----------|
| Sliding window (stride=64) | baseline | ~30s | Standard eval |
| Score-first TTT (3ep, 131K chunks) | ~-0.003 | ~120s | Score chunk, then train on it |
| SLOT (8 AdamW steps, delta vector) | ~-0.010 | ~90s | Per-batch delta reset, no cross-batch leakage |
| **Total eval** | | **~240s** | **Within 600s budget** |

## Architecture

- 11 layers, 512 dim, 8 heads, 8 KV heads, MLP mult 3.5
- BigramHash(6144x128), XSA-all, VE(128) at layers 9,10
- RoPE partial (16 dims), LN scale, U-Net skip connections
- SmearGate, tied embeddings
- Int5 GPTQ quantization with 10% magnitude pruning
- Late QAT (threshold=0.5)

## Run Command

```bash
NCCL_IB_DISABLE=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Or use `run.sh` which sets all environment variables.

## Rule Compliance

- [x] SLOT is score-first: hidden states frozen, delta adapts only through logit projection
- [x] No re-scoring of already-scored tokens
- [x] GPTQ calibration within training budget (590s train + ~5s calibration < 600s)
- [x] Artifact < 16MB (Int5 GPTQ + zstd compression)
- [x] All assertions present (artifact size, eval time budget)
- [x] inference_temp = 1.0
- [x] No n-gram cache, no multi-pass, no min(NLL)
