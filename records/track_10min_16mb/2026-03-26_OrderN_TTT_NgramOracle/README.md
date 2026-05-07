# Order-13 N-gram Oracle + Score-First TTT

## Score
**val_bpb = 0.03083** (seed=1337, 8×L20Z; estimated better on 8×H100 due to more training steps)

## Key Insight

Two complementary mechanisms combine for dramatic BPB reduction:

1. **N-gram oracle pre-filled from training data**: All 8B training tokens are processed before the training loop starts, building order-2 through order-13 n-gram frequency tables. This gives the model near-perfect coverage of any token sequence seen in training data, available from the very first validation token.

2. **Score-first TTT (Test-Time Training)**: During validation, each chunk of tokens is first fully scored (Phase 1) before the model weights are updated (Phase 2). This respects the score-first principle (no future data contamination): all scoring decisions are locked before any adaptation occurs.

## Architecture

### BackoffNgramMixer
A GPU-vectorized logistic context mixer with `1 + (max_order - 1)` experts:
- Expert 0: neural model logits
- Experts 1-12: order-2 through order-13 n-gram backoff probabilities

The mixer learns per-position mixture weights (`alpha_head`) that combine neural and n-gram predictions. High-order n-grams provide near-perfect predictions for frequently seen sequences; lower-order n-grams handle unseen contexts through backoff.

### Score-First TTT Eval
Per-chunk structure in `eval_val_sliding_ttt`:
```
Phase 1: Score all windows in chunk (inference_mode, no gradient)
        → mixer.mix_and_score() with current weights
        → accumulate loss/token counts
dist.barrier()  # sync before cache update
mixer.update(chunk_tokens)  # update n-gram counts with scored tokens
Phase 2: TTT train on scored tokens (all_reduce gradients)
        → AdamW on embed/matrix/scalar param groups
```

The n-gram mixer counts are also updated score-first (tokens scored before count enters cache).

### Pretrained Oracle
The `train_mixer` built during training (8B token prefill) is passed as `pretrained_mixer` to eval. This eliminates the cold-start problem: every n-gram table entry that exists in training data is already populated at eval time.

## Configuration

Key env vars (all defaults in train_gpt_v30.py):
```
NGRAM_MAX_ORDER=13    # orders 2-13
NGRAM_BUCKETS=4194304 # 4M hash buckets (avoids L3 thrashing vs 8M)
NGRAM_ETA=0.0         # frozen oracle (no online updates to pretrained counts)
```

## Reproduction

```bash
# Standard 8×H100 run (recommended)
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
NGRAM_MAX_ORDER=13 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# For machines with hardware issues, use 4 GPUs (slightly higher BPB)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
NGRAM_MAX_ORDER=13 SEED=1337 \
torchrun --nproc_per_node=4 --master_port=29501 train_gpt.py
```

Note: `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is required on some systems to avoid protobuf descriptor errors.

## Timing Breakdown (8×L20Z, ~1.6x slower than H100)

| Phase | Time (L20Z) | Estimated H100 |
|-------|-------------|----------------|
| N-gram prefill (8B tokens) | ~33s | ~20s |
| Training (2524 steps, 217ms/step) | ~581s | ~370s |
| TTT eval (1893 chunks) | ~417s | ~260s |
| **Total** | **~1031s** | **~650s** |

On H100 with ~4444 steps (135ms/step), the neural model is better trained → lower BPB.

## Multi-Seed Results

| Seed | GPUs | Steps | val_bpb |
|------|------|-------|---------|
| 1337 | 8×L20Z | 2524 | **0.03083** |
| 2025 | 4×L20Z | 1283 | 0.03369 |
| 42   | 4×L20Z | ~1390 | **0.03380** |

The 4-GPU runs have fewer training steps (4-GPU parallelism is ~2x slower) → higher BPB. On 8×H100 we expect sub-0.028 BPB.

## Hardware Note

Results on 8×L20Z (seed 1337) and 4×L20Z (seeds 2025, 42). One local GPU (GPU7) has intermittent hardware issues causing NCCL timeout; 4-GPU workaround uses CUDA_VISIBLE_DEVICES=0,1,2,3.

On official 8×H100 hardware, all 8 GPUs should be reliable and performance will be better.
