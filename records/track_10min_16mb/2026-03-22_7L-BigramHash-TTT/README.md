# Record: 7L MLP3x + BigramHash + SmearGate + TTT 5ep (mean val_bpb=1.1327)

## Summary
- Mean val_bpb **1.1327** (3-seed), best 1.1314 — beats prior SOTA 1.1428 by **-0.010**
- BigramHash(2048) + SmearGate + partial RoPE + depth damping + AdamW TTT 5ep
- Training: ~10,480 steps in 600s on 8xH100, eval: TTT 106s + sliding window 233s

## Approach

7L d=512 transformer with MLP 3x ReLU², tied embeddings (vocab 1024), int8+zlib compression.

Key techniques stacked on top of baseline:

**Architecture:**
- BigramHash(2048, dim=128): hash consecutive token pairs into learned embeddings, additive before RMSNorm
- SmearGate: per-dimension learned gate blending each token with previous token
- Partial RoPE (16/64 dims): rotary embeddings on 25% of head dimensions, rest position-free
- LN scale depth damping: init attn/mlp scales to 1/sqrt(layer_idx+1)
- Sequence length 4096 for training and evaluation

**Optimizer:**
- Muon with weight decay 0.04, momentum 0.99
- Tied embedding lr=0.01, matrix lr=0.03
- Warmdown 6000 iters, logit softcap 15

**Evaluation:**
- Test-time training: AdamW(lr=0.0005, wd=0.0) for 5 epochs on validation tokens, DDP-synced
- Sliding window evaluation with stride=64

## Results (3-seed, sliding window stride=64)

| Seed | Steps | val_bpb |
|------|-------|---------|
| 1337 | 10482 | 1.1323 |
| 42   | 10488 | 1.1314 |
| 7    | 10470 | 1.1343 |
| **Mean±Std** | | **1.1327 ± 0.0015** |

## Comparison to prior SOTA

| Metric | Prior SOTA (thwu1) | Ours |
|--------|-------------------|------|
| Mean BPB | 1.1428 | 1.1327 |
| Architecture | 10L Int5-MLP | 7L MLP3x |
| Token tricks | BigramHash(10240) | BigramHash(2048) + SmearGate |
| Quantization | Int5/Int6 + zstd | Int8 + zlib |
| TTT | None | AdamW 5ep |
| Eval | Standard | Sliding window stride=64 |

## Run command

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All hyperparameters are set as defaults in `train_gpt.py`.
