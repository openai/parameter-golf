# Record: 0.1582 BPB — Learned Mixer Head + No TTT + Matrix LR 0.03

**val_bpb = 0.1582** (seed 42, additional seeds pending) | **15.59 MB** | 8xH100 SXM | **No TTT**

## Results

| Seed | Steps | ms/step | Sliding BPB | **Mixer BPB** | Artifact |
|------|-------|---------|-------------|---------------|----------|
| 42 | 5,300 | 113 | 1.1396 | **0.1582** | 15,590,944 |

## Two Key Changes from PR #834

1. **MATRIX_LR=0.03** (was 0.025) — discovered through systematic screening of 79+ experiments
2. **TTT_EPOCHS=0** — completely removes test-time training. Result is clean, fully legal, no gradient updates on val data

Despite removing TTT, our result (0.1582) **beats** PR #834's original (0.1663 with TTT enabled). The higher matrix LR produces a better-trained model that the learned mixing head can leverage more effectively.

## Architecture (from PR #834)

- 11L, 512d, MHA 8/8, MLP 3.5x, LeakyReLU(0.5)²
- **Learned mixer head**: `Linear(512 → 7)` predicts per-token mixing weights for neural model + n-gram orders 2-7
- **Frozen n-gram oracle**: bigram/trigram/...7-gram tables precomputed from training data, used as lookup during training
- Mixed int5/int6 quantization + GPTQ + zstd, EMA(0.997), CROWN-Q penalty

## Eval: Learned Multi-Expert Mixing (NO TTT)

- Score-first backward-looking n-gram cache (orders 2-7)
- Model-predicted mixing weights (not fixed alpha — learned during training)
- Each token gets its own expert weights based on transformer hidden state
- **515s eval time** (within 600s budget, no TTT overhead)

## Reproduction

```bash
MATRIX_LR=0.03 TTT_EPOCHS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Legality

- No TTT (zero gradient updates on validation data)
- N-gram cache is backward-looking (score-first, cache updated after scoring)
- Learned mixing head trained on training data only (frozen oracle)
- Single-pass evaluation

## Based On

- PR #834: Learned Multi-Expert Gate + Frozen Oracle architecture
- Our systematic hyperparameter screening (79+ experiments, MATRIX_LR=0.03 discovery)
