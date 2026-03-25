# Record: Multi-order N-gram Backoff + Entropy-Adaptive Alpha (val_bpb=0.9674)

**val_bpb = 0.9674** (3-seed mean, std=0.0006) | **~15.99 MB** | **No TTT**

## 3-Seed Results

| Seed | val_bpb | val_loss | Submission Size | Quantization |
|------|---------|----------|-----------------|--------------|
| 1337 | 0.96679 | 1.63238 | 15,994,366 B | int6+zstd-16 |
| 42   | 0.96703 | 1.63278 | 15,996,585 B | int6+zstd-16 |
| 7    | 0.96825 | 1.63485 | 15,988,201 B | int6+zstd-16 |
| **Mean** | **0.96736** | **1.63334** | — | — |
| **Std** | **0.00063** | — | — | — |

## Architecture

- **Model**: 11 layers, 512 dim, GQA 8H/4KV, MLP 3x expansion
- **Activation**: LeakyReLU(0.5)² (squared LeakyReLU with negative slope 0.5)
- **Attention**: XSA-all with last_n=11 (cross-sequence attention across all layers)
- **Residual**: Value Residual + Gated Attention
- **Embeddings**: SmearGate, BigramHash(4096), Partial RoPE (16/64), tied embeddings
- **Normalization**: LN Scale (learnable scale only, no bias)
- **EMA**: decay=0.997
- **Optimizer**: Muon (momentum=0.99, warmup from 0.92 over 1500 steps)
- **LR**: matrix=0.025, scalar=0.025, tied_embed=0.035

## Quantization

int6 per-row quantization + zstd-16 compression. Auto-downgrade fallback to int5 for select layers is available but was **not triggered** for any seed in this run.

## N-gram Eval Cache

The n-gram cache is an **eval-time only** technique that interpolates LM logits with n-gram statistics collected during evaluation.

### Multi-order Backoff (orders 2–7)

Instead of a single fixed n-gram order, we maintain counts for orders 2 through 7. At each position, we attempt the highest order first (7-gram). If the context has no match (count < min_count=2), we cascade down to the next lower order until a match is found or we exhaust all orders. This dramatically improves coverage compared to a fixed high-order model.

### Entropy-Adaptive Alpha

Instead of a fixed interpolation weight, alpha adapts based on the model's own entropy:

```
alpha = ent_base + ent_range * sigmoid(2 * (H - 4.0))
      = 0.05    + 0.55      * sigmoid(2 * (H - 4.0))
```

- **Low entropy** (model is confident): alpha → 0.05, trust the LM
- **High entropy** (model is uncertain): alpha → 0.60, trust the n-gram cache

### Compliance

- **Score-first, backward-looking**: n-gram counts are built from previously scored tokens only
- **No oracle selection**: alpha depends solely on the model's own output distribution (entropy), never on ground-truth labels
- **No cross-GPU sync**: each GPU maintains its own independent cache (4M buckets)

## Ablation

| Configuration | val_bpb | Delta |
|---------------|---------|-------|
| No n-gram cache (neural only) | 1.1271 | baseline |
| Fixed alpha=0.40, order=7, no backoff | 1.0336 | -0.0935 |
| Multi-order backoff (2-7) + fixed alpha=0.40 | 0.9825 | -0.1446 |
| **Multi-order backoff (2-7) + entropy-adaptive** | **0.9674** | **-0.1597** |

Entropy-adaptive alpha improves over fixed alpha by **0.0151 bpb**.

## Reproduction

```bash
cd records/track_10min_16mb/2026-03-26_Backoff_Entropy_Adaptive

# Symlink data directory
ln -sf ../../../data data

# Training (seed 1337)
SEED=1337 TTT_ENABLED=0 CANON_LAST_N=0 SWA_ENABLED=0 \
  MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
  MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
  XSA_LAST_N=11 LEAKY_RELU=1 MAX_WALLCLOCK_SECONDS=600 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

# Eval-only with n-gram cache (uses saved model)
EVAL_ONLY="$(pwd)/final_model.pt" ITERATIONS=0 \
  TTT_ENABLED=0 CANON_LAST_N=0 SWA_ENABLED=0 \
  MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
  XSA_LAST_N=11 LEAKY_RELU=1 MAX_WALLCLOCK_SECONDS=600 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

Built on [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) and the following PRs:
- PR #315 (GQA + RoPE)
- PR #609 (XSA)
- PR #493 (Value Residual)
- PR #518 (Gated Attention)
- PR #413 (LeakyReLU²)
- PR #674 (SmearGate + BigramHash)
- PR #702 (Multi-order backoff concept)
