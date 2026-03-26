# Record: Phrase Cache + Order-Adaptive N-gram + Regime Tracker + TTT (val_bpb=0.1003)

**val_bpb: 0.1003** (3-seed mean) | **~15.7 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM)

| Seed | Pre-TTT BPB | **Post-TTT BPB** | Artifact | Train time | Eval time |
|------|-------------|-----------------|----------|------------|-----------|
| 1337 | 1.1287 | **0.1003** | 15.74 MB | 582s | 592.4s |
| 42 | 1.1277 | **0.1002** | 15.59 MB | 582s | 593.3s |
| 7 | 1.1249 | **0.1003** | 15.73 MB | 582s | 590.0s |
| **Mean** | 1.1271 | **0.1003** | | | |

## Key Techniques

### 1. Long Phrase Cache (eval-time, novel)
Variable-length suffix matcher that complements the fixed-order n-gram cache. Probes at lengths [48, 36, 28, 20, 16] using rolling hashes. When a 48-token suffix matches previously scored text, it's almost certainly an exact copy (boilerplate, markup, legal text) — gets alpha near 0.99.

Exploits the massive verbatim repetition in web text that fixed-order n-grams (even order 9) miss: cookie banners, navigation menus, code headers, list structures, copyright notices.

### 2. Order-Adaptive Entropy Gating (eval-time)
Per-order entropy thresholds and alpha multipliers:
- High-order matches (order 9): low entropy threshold, 2.0× multiplier
- Low-order matches (order 2): high entropy threshold, 0.3× multiplier
- Continuous sigmoid interpolation between orders

### 3. Online Regime Tracker (eval-time, novel)
Detects text regime (boilerplate vs prose vs code) from cheap scored-token features: n-gram match rate, average match order, token diversity. Modulates alpha multiplier [0.7×, 1.5×] by detected regime.

### 4. Multi-Order N-gram Backoff Cache (eval-time)
Hashed count tables for orders 2-9 with 4M buckets each. Full-chunk cache sharing across all 8 GPU ranks. Entropy-adaptive base alpha (alpha_high=0.95). min_count=1.

### 5. Score-First TTT (eval-time)
2-epoch AdamW TTT with Polyak EMA (decay=0.998), byte-weighted loss, adaptive cosine LR. Freeze first 2 blocks.

### 6. Adaptive Temperature + Online Logit Calibration (eval-time)
Per-token temperature=0.85 on logits. Momentum-EMA logit bias correction.

### 7. 5-Expert Hedge Mixer + CROWN-Q + GPTQ int5 (training-time)
GPU-vectorized Hedge mixer. Quantization-aware penalty. 5% magnitude pruning. zstd level 22.

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 8KV) |
| MLP | 3.5x with LeakyReLU(0.5)^2 |
| BigramHash | 6144 (dim=128) |
| XSA | All 11 layers (ws=8) |
| VE128 | Layers 9-10 |
| Quantization | Full GPTQ int5 + zstd level 22 |
| Pruning | 5% magnitude |
| CROWN-Q | lambda=0.01 during warmdown |
| TTT | AdamW lr=0.0001, 2 epochs, 131K chunks, Polyak 0.998 |
| N-gram cache | Orders 2-9, 4M buckets, order-adaptive gating |
| Phrase cache | Probes [48, 36, 28, 20, 16], 4M buckets |
| Regime tracker | Window=4096, alpha mult [0.7, 1.5] |
| Temperature | 0.85 (adaptive per-token) |

## Compliance

| Constraint | Limit | Actual | Status |
|-----------|-------|--------|--------|
| Train time | 600s | 582s | Pass |
| Eval time | 600s | 593.3s (worst seed) | Pass |
| Artifact size | 16,000,000 bytes | 15,737,937 (worst seed) | Pass |
| No pre-scoring training | — | Score-first TTT + backward-looking caches | Pass |
| GPTQ in training budget | — | 1.8s within 18s reserve | Pass |
| Single-pass scoring | — | Each token scored exactly once, no rescoring | Pass |

## Reproduction

```bash
cd records/track_10min_16mb/2026-03-26_PhraseCache_OrderAdaptiveNgram_RegimeTracker_0.1003
DATA_PATH=../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 MAX_WALLCLOCK_SECONDS=600 \
USE_MIXER=1 MIXER_ETA=0.1 \
TTT_EPOCHS=2 TTT_FREEZE_BLOCKS=2 \
TTT_LR=0.0001 TTT_CHUNK_TOKENS=131072 \
ADAPTIVE_LR=1 ADAPTIVE_LR_MAX=3.0 \
EVAL_STRIDE=64 \
CROWN_Q_LAMBDA=0.01 \
USE_NGRAM_CACHE=1 \
NGRAM_EVAL_ORDER=9 NGRAM_ALPHA_HIGH=0.95 NGRAM_EVAL_MIN_COUNT=1 \
USE_LOGIT_CAL=1 \
TTT_TEMPERATURE=0.85 \
PRUNE_PCT=0.05 \
NGRAM_ORDER_ADAPTIVE=1 \
USE_REGIME_TRACKER=1 \
USE_PHRASE_CACHE=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
