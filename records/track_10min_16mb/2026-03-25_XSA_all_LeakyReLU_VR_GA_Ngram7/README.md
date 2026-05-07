# Record: 11L XSA-all + LeakyReLU(0.5)^2 + VR + GA + 7-gram cache (val_bpb=1.0337)

**3-seed mean val_bpb = 1.0337** (std=0.0010) | **~15.99 MB** | No TTT

## Summary

Non-TTT submission combining XSA on all 11 layers with LeakyReLU(0.5)^2 activation, Value Residual, Gated Attention, and a 7-gram backward-looking eval cache (alpha=0.40, fixed mixing). Achieves 1.0337 mean BPB across 3 seeds on 8xH100 SXM within 600s wallclock.

## 3-Seed Results (8xH100 SXM, 600s wallclock)

| Seed | Steps | Quant | Size (bytes) | Sliding BPB (s=64) |
|------|-------|-------|-------------|---------------------|
| 1337 | 5589 | int6 zstd-16 | 15,990,221 | 1.0329 |
| 42 | ~5589 | int6 zstd-17 | 15,982,903 | 1.0334 |
| 7 | ~5589 | int6 zstd-16 | 15,992,378 | 1.0349 |
| **Mean** | | | | **1.0337** |
| **Std** | | | | **0.0010** |

## Architecture

- 11 transformer layers, 512d, 8H/4KV (GQA), MLP 3x
- **LeakyReLU(0.5)^2**: `leaky_relu(x, 0.5).square()` replaces ReLU^2. Preserves negative gradient flow at zero overhead.
- **XSA on all 11 layers**: Exclusive Self-Attention removes self-position bias in all layers.
- **Value Residual (VR)**: Layer 0 V output mixed into subsequent layers via learned sigmoid gates.
- **Gated Attention (GA)**: Per-head sigmoid gates on attention output.
- SmearGate + OrthoInit, BigramHash(4096), U-Net skip connections
- Partial RoPE (16/64 dims), LN Scale, EMA(0.997)
- Int6 per-row quantization + zstd compression

## 7-gram Backward-Looking Eval Cache

During sliding-window evaluation, a token-level n-gram cache adjusts the model's next-token predictions using observed n-gram statistics from previously scored tokens.

### How it works

1. As evaluation proceeds left-to-right through the validation set, completed (already-scored) tokens are added to an n-gram frequency table.
2. For each new position, the cache looks up all n-gram contexts (orders 1 through 7) ending at the current position using only backward (already-scored) context.
3. The n-gram distribution is mixed with the model's softmax output: `p_final = (1 - alpha) * p_model + alpha * p_ngram`, with a fixed alpha=0.40.
4. The mixed distribution is used to compute the loss for that position.

### Compliance notes

- **Score-first**: Each token is scored by the model *before* it enters the n-gram table. The cache only uses tokens that have already been scored — it never looks ahead.
- **Fixed alpha**: The mixing weight alpha=0.40 is a fixed hyperparameter baked into the submission code, not tuned per-sample or per-position at eval time.
- **No oracle selection**: There is no selection among multiple cache configurations at eval time. The same alpha and order are used for every token.
- **Deterministic**: Given the same model weights and validation data, the cache produces identical results regardless of hardware or random seeds.
- **No additional parameters**: The n-gram cache adds zero learned parameters. It is a purely statistical post-processing step built from the evaluation data stream.

## Training Config

```bash
ITERATIONS=20000 (wallclock-capped at ~5589 steps)
WARMDOWN_ITERS=3000  MAX_WALLCLOCK_SECONDS=600
MATRIX_LR=0.025  SCALAR_LR=0.025  TIED_EMBED_LR=0.035
MUON_MOMENTUM=0.99  MUON_MOMENTUM_WARMUP_START=0.92  MUON_MOMENTUM_WARMUP_STEPS=1500
XSA_LAST_N=11  LEAKY_RELU=1  TTT_ENABLED=0  CANON_LAST_N=0  SWA_ENABLED=0
# N-gram cache (eval-time only):
NGRAM_CACHE=1  NGRAM_ALPHA=0.40  NGRAM_ORDER=7
```

## Reproduction

```bash
# Download data
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

# Train (seed 1337, 8xH100)
SEED=1337 XSA_LAST_N=11 LEAKY_RELU=1 WARMDOWN_ITERS=3000 \
  MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
  MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
  TTT_ENABLED=0 CANON_LAST_N=0 SWA_ENABLED=0 MAX_WALLCLOCK_SECONDS=600 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- Base architecture: modded-nanogpt
- XSA-all: PR #609
- LeakyReLU^2: PR #493, #518
- Value Residual: PR #413 (arXiv:2410.17897)
- Gated Attention: NeurIPS 2025, arXiv:2505.06708
