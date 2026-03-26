# Record: Order-Adaptive N-gram + TTT + Hedge Mixer (val_bpb=0.2071)

**val_bpb: 0.2071** (3-seed mean) | **~15.5 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM)

| Seed | Pre-TTT BPB | **Post-TTT BPB** | Artifact | Train time | Eval time |
|------|-------------|-----------------|----------|------------|-----------|
| 1337 | 1.1243 | **0.2071** | 15.35 MB | 582s | 496s |
| 42 | 1.1266 | **0.2074** | 15.60 MB | 582s | 496s |
| 7 | 1.1249 | **0.2069** | 15.59 MB | 582s | 495s |
| **Mean** | 1.1253 | **0.2071** | | | |

## Key Techniques

### 1. Order-Adaptive Entropy Gating (eval-time)
Per-order entropy thresholds and alpha multipliers for the n-gram cache:
- High-order matches (order 9): low entropy threshold (trust even when model is fairly confident), 2.0× alpha multiplier
- Low-order matches (order 2): high entropy threshold (only trust when model is confused), 0.3× alpha multiplier
- Continuous sigmoid interpolation between orders

### 2. Multi-Order N-gram Backoff Cache (eval-time)
Hashed count tables for orders 2-9 with 4M buckets each. Multi-order backoff: try longest match first, fall back to shorter. Entropy-adaptive base alpha (alpha_high=0.95). min_count=1.

### 3. Full-Chunk Cache Sharing (eval-time)
All 8 GPU ranks update their n-gram caches with the FULL chunk of scored tokens (not just their own windows). This gives 8× more n-gram data per rank.

### 4. Score-First TTT (eval-time)
4-epoch AdamW TTT with Polyak EMA (decay=0.998), byte-weighted loss, adaptive cosine LR. Freeze first 2 blocks, unfreeze last 9 + norms/scales.

### 5. Adaptive Temperature Sharpening (eval-time)
Per-token temperature=0.85 on logits. The model is under-confident after quantization; sharpening concentrates probability on correct tokens.

### 6. Online Logit Calibration (eval-time)
Momentum-EMA tracker of per-token empirical frequency vs model predicted probability. Applies log-ratio correction to logits to fix systematic over/under-prediction.

### 7. 5-Expert Hedge Mixer (eval-time)
GPU-vectorized logistic context mixing: neural, unigram, bigram, trigram, and entropy experts with Hedge/multiplicative-weights updates.

### 8. CROWN-Q + GPTQ int5 (training-time)
Quantization-aware penalty during warmdown + GPTQ with 5% magnitude pruning + zstd level 22 compression.

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
| TTT | AdamW lr=0.0001, 4 epochs, 131K chunks, Polyak 0.998 |
| Mixer | 5-expert Hedge (eta=0.1) |
| N-gram cache | Orders 2-9, 4M buckets, order-adaptive gating |
| Temperature | 0.85 (adaptive per-token) |
| Training reserve | 18s |
| Eval stride | 64 |

## Compliance

| Constraint | Limit | Actual | Status |
|-----------|-------|--------|--------|
| Train time | 600s | 582s | Pass |
| Eval time | 600s | 496s | Pass |
| Artifact size | 16,000,000 bytes | 15,600,361 (worst seed) | Pass |
| No pre-scoring training | — | Score-first TTT + backward-looking cache | Pass |
| GPTQ in training budget | — | 1.8s within 18s reserve | Pass |

## Reproduction

```bash
cd submission-curr-2026-03-26
DATA_PATH=../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 MAX_WALLCLOCK_SECONDS=600 \
USE_MIXER=1 MIXER_ETA=0.1 \
TTT_EPOCHS=4 TTT_FREEZE_BLOCKS=2 \
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
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
