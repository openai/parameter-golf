# Order-Adaptive Entropy Gating + BackoffNgramMixer + Drift-Free TTT

**val_bpb: 0.5466** (3-seed mean, std 0.0010) | **~15.99 MB** | 8×H100 SXM

Adds order-adaptive entropy gating on top of [PR #779](https://github.com/openai/parameter-golf/pull/779)'s BackoffNgramMixer + Drift-Free TTT submission. Instead of using a single entropy center for all n-gram orders, each order gets its own threshold — higher orders are trusted at lower entropy, lower orders only kick in when the model is more uncertain.

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | Pre-TTT bpb | **Post-TTT bpb** | TTT gain | TTT time | Artifact |
|------|----------|-------|-------------|-----------------|----------|----------|----------|
| 1337 | 99.3ms | 5,863 | 1.1279 | **0.5478** | -0.5801 | 607s | 15,995,959 |
| 42 | 98.3ms | 5,863 | 1.1362 | **0.5458** | -0.5904 | 606s | 15,979,251 |
| 2025 | 99.2ms | 5,869 | 1.1369 | **0.5463** | -0.5906 | 607s | 15,994,227 |
| **Mean** | **98.9ms** | **5,865** | **1.1337** | **0.5466 (std 0.0010)** | **-0.5871** | **~607s** | |

## What Changed vs PR #779

PR #779 uses a single `entropy_center=3.5` for all n-gram orders. We replace this with per-order entropy centers:

```python
# PR #779 (single entropy center for all orders)
alpha = alpha_min + (alpha_max - alpha_min) * sigmoid(2.0 * (entropy - 3.5))

# This submission (per-order entropy centers)
ent_centers = {7: 3.0, 6: 3.2, 5: 3.5, 4: 3.8, 3: 4.2, 2: 4.5}
ent_center = ent_centers[matched_order]
alpha = alpha_min + (alpha_max - alpha_min) * sigmoid(2.0 * (entropy - ent_center))
```

Higher-order n-grams (7, 6, 5) are trusted at lower model entropy — when the model is fairly confident, the precise n-gram correction refines the prediction. Lower-order n-grams (4, 3, 2) only intervene at higher entropy — when the model is confused enough that even coarse statistics help.

This is an eval-time-only change. It modifies how existing n-gram statistics are combined with neural predictions, not when data enters the cache. The n-gram cache is still updated strictly AFTER scoring each batch (score-first).

## Legality

- **Score-first**: N-gram cache updated AFTER scoring each batch. No future tokens leak into predictions.
- **No oracle selection**: Alpha depends only on model entropy and n-gram order, not on ground truth.
- **Artifact size**: All seeds strictly under 16,000,000 bytes (max: 15,995,959).
- **Training time**: Capped at 600s (10 min) on 8×H100 (actual: ~582s).
- **Eval time**: TTT eval ≤607s on 8×H100.

## BackoffNgramMixer (from PR #779)

Multi-order n-gram backoff (orders 2-7) with entropy-adaptive alpha mixing:

1. For each token position, try orders 7→6→5→4→3→2
2. Use the highest order with sufficient context count
3. Mix n-gram prediction with neural prediction using entropy-adaptive alpha
4. Cache is updated AFTER scoring (score-first guarantee)

## Drift-Free TTT (from PR #779)

- Only Q-projections are unfrozen during TTT (535,556 params out of 33M)
- Conservative LR (3e-5) with Polyak averaging (decay=0.998)
- 1 epoch per chunk, 63 chunks of ~1M tokens each
- Logistic context mixer (eta=0.1) for final combination

## Architecture

PR #779 stack (33M parameters):

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 8KV) |
| MLP | 3× with LeakyReLU(0.5)² |
| XSA | All 11 layers |
| GQA | 8/8 (no grouping) |
| Weight avg | EMA + SWA |
| Quantization | int6 + LZMA compression |
| Pruning | 3.0% magnitude |
| Optimizer | Muon (matrices) + AdamW (embeddings/scalars) |

## Timing Budget

| Phase | Time |
|-------|------|
| Training | ~582s (≤600s) |
| Sliding window eval | ~168s |
| TTT (score-first + adaptation) | ~607s |
| **Total eval** | **~775s** |

## Run Command

```bash
SEED=1337 torchrun --nproc-per-node=8 train_gpt.py
```

Environment variables (all have defaults in the code):
```bash
export SEED=1337
export USE_MIXER=1           # Enable BackoffNgramMixer
export QTTT=1                # Enable quantized TTT
export TTT_ENABLED=1         # Enable test-time training
export TTT_LR=0.00003        # TTT learning rate
export TTT_EPOCHS=1          # Epochs per TTT chunk
export TTT_CHUNK_TOKENS=1000000  # Tokens per TTT chunk
export TTT_FREEZE_BLOCKS=2   # Freeze first N blocks during TTT
export USE_POLYAK=1           # Enable Polyak averaging
export POLYAK_DECAY=0.998     # Polyak decay rate
export MIXER_ETA=0.1          # Logistic context mixer learning rate
```

## Ablation

| Change | Post-TTT bpb | Delta |
|--------|-------------|-------|
| PR #779 baseline (single entropy center) | 0.6713 | — |
| + Order-adaptive entropy gating | **0.5478** | **-0.1235** |

## Credits

- **BackoffNgramMixer + Drift-Free TTT + Base model**: [PR #779](https://github.com/openai/parameter-golf/pull/779)
- **Order-adaptive entropy gating**: This submission
