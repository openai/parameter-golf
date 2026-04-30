# Sliding Window Eval + Muon6 + Extended Warmup

**Mean val_bpb: 1.1973** (3 seeds, p<0.001)

## Key Techniques

### 1. Sliding Window Evaluation (stride=256, seq_len=1024)

Instead of splitting the validation set into non-overlapping 1024-token chunks (where early tokens in each chunk get poor predictions due to limited context), we use a sliding window with stride=256. Each token is scored with at least 768 tokens of prior context, dramatically improving per-token predictions.

The `forward_logits()` method on the GPT class enables efficient single-sequence inference without the compiled training graph. Eval runs in ~126 seconds on 8xH100, well within the 10-minute eval budget.

### 2. Muon 6-Step Newton-Schulz Orthogonalization

Increased `MUON_BACKEND_STEPS` from 5 to 6. More accurate orthogonalization of the gradient matrix in the Muon optimizer leads to better training dynamics, particularly in the later stages of training.

### 3. Extended Momentum Warmup + Longer Warmdown

- `MUON_MOMENTUM_WARMUP_STEPS=1000` (up from 500): Slower momentum warmup from 0.85→0.95 stabilizes early training.
- `WARMDOWN_ITERS=1500` (up from 1200): Longer cosine warmdown schedule allows more gradual learning rate decay before the wallclock cap.

## Architecture

Unchanged from baseline: 9-layer transformer, 512 dim, 8 heads / 4 KV heads (GQA), 2x MLP, tied embeddings, RoPE, relu², logit softcap.

## Results

| Seed | val_loss | val_bpb | Steps | ms/step |
|------|----------|---------|-------|---------|
| 1337 | 2.0208 | 1.1968 | ~13,688 | 45.09 |
| 42 | 2.0217 | 1.1974 | ~13,688 | 45.09 |
| 7 | 2.0225 | 1.1978 | ~13,688 | 45.09 |
| **Mean** | **2.0217** | **1.1973** | | |

Artifact: ~15.9 MB | Eval time: ~126s (sliding window, stride=256)

## Reproduction

```bash
MUON_BACKEND_STEPS=6 MUON_MOMENTUM_WARMUP_STEPS=1000 WARMDOWN_ITERS=1500 EVAL_STRIDE=256 \
RUN_ID=submission SEED=1337 VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
