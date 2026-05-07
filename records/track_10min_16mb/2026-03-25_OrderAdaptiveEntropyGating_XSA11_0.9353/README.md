# Order-Adaptive Entropy Gating + XSA-All

**val_bpb: 0.9370** (n-gram7 sliding window, stride=64, 3-seed mean, std=0.0003) | **~15.9 MB** artifact | 8xH100 SXM, 600s

Built on PR #753 with two improvements: XSA extended to all layers and order-adaptive entropy gating for n-gram eval.

## Results (8xH100 80GB SXM)

| Seed | Steps | Sliding s64 BPB | N-gram7 s64 BPB | Artifact |
|------|-------|-----------------|-----------------|----------|
| 1337 | 6,783 | 1.1225 | 0.9372 | 15,828,199 |
| 42 | 6,783 | 1.1219 | 0.9372 | 15,923,891 |
| 2025 | 6,776 | 1.1223 | 0.9367 | 15,964,115 |
| **Mean** | | **1.1222 (±0.0003)** | **0.9370 (±0.0003)** | |

| Metric | Value |
|--------|-------|
| Step avg | ~88.5ms |
| Training time | 600s |
| **Total submission size (seed 1337)** | **15,828,199 bytes** |

## Key Innovation: Order-Adaptive Entropy Gating

Standard n-gram eval uses a single `entropy_center` threshold to decide when to trust the n-gram cache over the transformer. This treats all n-gram orders equally -- but a 7-gram match ("the United States of America") is far more informative than a 2-gram match ("of the").

**Order-adaptive entropy gating** assigns a different entropy threshold per n-gram order:

```
ent_center_n = entropy_center - slope * (matched_order - min_order)
```

With `entropy_center=3.0` and `slope=0.25`:
- **7-gram match**: threshold = 3.0 - 0.25*(7-2) = **1.75** (trust even at moderate model confidence)
- **5-gram match**: threshold = 3.0 - 0.25*(5-2) = **2.25**
- **3-gram match**: threshold = 3.0 - 0.25*(3-2) = **2.75**
- **2-gram match**: threshold = 3.0 - 0.25*(2-2) = **3.00** (only trust when model is very uncertain)

The intuition: high-order n-grams capture specific multi-word patterns that are almost certainly correct. Low-order n-grams are noisy frequency estimates that should only override the transformer when it has no idea what comes next.

### Implementation

Three changes to the n-gram eval loop (all eval-time only, no training changes):

1. **Track matched order per token**: During multi-order backoff (7→6→5→...→2), record which order actually matched for each token position.

2. **Compute order-aware entropy center**: Replace the scalar `entropy_center` with a per-token center that depends on the matched n-gram order.

3. **Use order-aware center in sigmoid gate**: The mixing weight `alpha` between transformer and n-gram predictions uses the order-specific threshold instead of the global one.

```python
# Standard (single threshold for all orders)
alpha_i = alpha_max * sigmoid((entropy_i - ent_center) / temp)

# Order-adaptive (threshold varies by matched n-gram order)
ent_center_i = ent_center - slope * (matched_order_i - min_order)
alpha_i = alpha_max * sigmoid((entropy_i - ent_center_i) / temp)
```

**Score-first legality**: The matched order comes from the n-gram cache (built from already-scored tokens only). The entropy comes from the model's own logits. No future tokens are used.

### Ablation

| Configuration | N-gram7 BPB | Delta vs PR #753 baseline |
|--------------|------------|--------------------------|
| PR #753 baseline (XSA_LAST_N=4, ent_center=4.0) | 0.9618 | -- |
| + XSA-all (XSA_LAST_N=11) + entropy_center=3.0 | 0.9416 | -0.0202 |
| + **Order-adaptive gating (slope=0.25)** | **0.9353** | **-0.0265** |

## Changes from PR #753

| | PR #753 | This |
|---|---|---|
| N-gram7 BPB | 0.9618 | **0.9353** |
| Sliding BPB (no n-gram) | 1.1193 | 1.1195 |
| XSA layers | Last 4 (XSA_LAST_N=4) | **All 11 (XSA_LAST_N=11)** |
| Entropy center | 4.0 | **3.0** |
| Order-adaptive gating | No | **Yes (slope=0.25)** |
| Artifact size | ~15.83 MB | ~15.83 MB |
| Training | Identical | Identical |

## Architecture (carried from PR #753)

- 11 transformer layers (512d, 8 heads, 4 KV heads)
- MLP 3x (1536 hidden) with LeakyReLU(0.5)^2 activation
- Cross-Self-Attention (XSA) with learned memory keys/values
- Partial RoPE (16/64 dims)
- LN Scale (1/sqrt(layer+1))
- Value Embedding (VE128) on layers 9-10
- Bigram Hash Embedding (1536 buckets)
- EMA(0.997) + SWA(every 50 steps)
- GPTQ int6 quantization + lzma compression
- Parameter Banking + Parallel Muon optimizer
- Late QAT (threshold=0.15)
- Multi-order n-gram eval with hashed backoff (orders 2-7)
- Shard ordering for training data
- DTG (Dynamic Token Gating)

## Configuration

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=11 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
NGRAM_EVAL_ORDER=7 NGRAM_EVAL_ALPHA=0.3 NGRAM_EVAL_MIN_COUNT=2 \
NGRAM_EVAL_BUCKETS=4194304 NGRAM_EVAL_ENTROPY_CENTER=3.0 \
NGRAM_EVAL_ORDER_ADAPTIVE=1 NGRAM_EVAL_ORDER_ENT_SLOPE=0.25 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Legality

- **Score-first n-gram cache**: Cache updated ONLY after scoring each sliding window batch. Tokens are never used before being evaluated.
- **Order-adaptive gating uses only model entropy and cache statistics**: The matched n-gram order comes from already-scored token patterns. The entropy is computed from the model's own logits. No ground truth tokens are accessed for the mixing decision.
- **No TTT**: This submission does not use test-time training.
- **Training time**: 600s (within 10-minute cap).
- **Artifact size**: 15,828,199 – 15,964,115 bytes across seeds (all within 16,000,000 byte cap).

## Credits

- **Base model + n-gram eval + GPTQ + full training stack**: PR #753 by @152334H (Podracing II)
- **XSA**: PR #430 by @sahiee-dev (extended from last-4 to all layers)
- **LeakyReLU^2**: PR #493 by @parinzee
- **Parameter Banking + Parallel Muon**: PR #399 by @abaybektursun
- **Order-adaptive entropy gating**: This submission

## Included Files

- `train_gpt.py` -- full training + quantization + n-gram evaluation script
- `train.log` -- training log from seed 1337
- `submission.json` -- leaderboard metadata
- `README.md` -- this file
