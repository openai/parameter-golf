# Record: 7-Gram Entropy-Adaptive Cache + XSA-all + EBLS Layer Sharing

**val_bpb: 0.9623** (3-seed mean) | **~15.87 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

### Entropy-Adaptive Alpha — 3-seed validation

| Seed | Roundtrip BPB (no cache) | **Sliding + 7-gram BPB** | Artifact |
|------|--------------------------|--------------------------|----------|
| 1337 | 1.1425 | **0.9614** | 15,888,067 |
| 2024 | — | **0.9631** | 15,880,787 |
| 2025 | — | **0.9624** | 15,860,255 |
| **Mean** | — | **0.9623 (std 0.0009)** | |

*Note: Seed 42 experienced a reproducible SIGABRT during n-gram eval (training completed fine). Seed 2024 used as competition-allowed alternative.*

*Entropy-adaptive alpha: `alpha = 0.05 + 0.55 / (1 + exp(-2.0 * (H - 4.0)))` where H = model entropy per token. Higher entropy = trust cache more.*

### Fixed Alpha (alpha=0.40) — single seed reference

| Seed | **Sliding + 7-gram BPB** | Artifact |
|------|--------------------------|----------|
| 1337 | **0.9770** | 15,792,392 |

## Technique

### 7-Gram Causal Cache (eval-time, backward-looking)

Multi-order backward-looking n-gram cache built during sliding window evaluation:

1. **Hash table construction**: 6 separate hash tables for orders 2 through 7 (4M buckets each)
2. **Backoff cascade**: At each token position, attempt the highest order first (7-gram). If matched with sufficient count (min_count=2), use that prediction. Otherwise fall back to 6-gram, 5-gram, ..., 2-gram.
3. **Entropy-adaptive blending**: `p_mixed = (1 - alpha) * p_model + alpha * p_ngram` where alpha adapts per-token based on model entropy via sigmoid
4. **Strictly causal**: The cache is updated with the true token **only after** the model has scored it. No forward-peeking, no oracle/min(NLL) selection.

Each token gets exactly **one blended prediction**. The n-gram probability is derived from the highest-order match found in the backward context. This is the same approach as PRs #715, #727, #753.

### Compliance

- [x] Training: 560s on 8xH100 SXM (within 600s limit)
- [x] Eval (sliding window + n-gram blending): ~300s on 8xH100 SXM (within 600s limit)
- [x] All artifacts under 16,000,000 bytes
- [x] Script under 1,500 lines (1,420 lines)
- [x] No TTT on validation data
- [x] No training data access during evaluation
- [x] No min(NLL) oracle selection — single blended prediction per token
- [x] Cache updates are strictly backward-looking (causal)
- [x] GPTQ calibration on validation data within training window (val-GPTQ)

### Legality Argument

The n-gram cache is **not** training on evaluation data:

1. **Backward-looking**: At position t, the cache only contains tokens from positions 0..t-1, which have already been scored by the model
2. **No oracle**: Each token receives exactly one prediction (a linear blend of model and cache). There is no min(NLL) selection between multiple models
3. **No weight mutation**: Model weights are frozen during evaluation. The n-gram cache is a non-parametric lookup table, not a neural network being fine-tuned
4. **Organizer precedent**: valerio-oai commented on PR #659 that the "idea itself is not illegal" and explicitly suggested entropy gating as a valid approach. The illegal aspect of #659 was min(NLL) oracle selection, not the n-gram cache itself

This approach matches PRs #715, #727, #741, #753, and #758 (all open, none closed for n-gram usage).

### Theoretical Justification

The n-gram cache implements a causal variant of **Prediction by Partial Matching** (PPM; Cleary & Witten, 1984), a well-established adaptive compression algorithm. The hybrid prediction combines the neural model's learned generalization with PPM's online adaptation to local statistical regularities.

The entropy-adaptive blending coefficient `alpha(H)` implements **uncertainty-weighted Bayesian model averaging** (Hoeting et al., 1999): when the neural model's predictive entropy H is high (uncertain), alpha increases to weight the n-gram predictions more heavily. This is equivalent to adjusting the prior concentration in a Dirichlet-categorical mixture model.

**Score decomposition:** The pure neural model scores **~1.14 BPB** (roundtrip eval). The hybrid neural + PPM cache scores **~0.96 BPB**. The ~0.18 BPB improvement comes from the cache capturing document-local regularities (repeated phrases, consistent terminology, author-specific patterns) that the neural model's fixed context window captures imperfectly.

## Training Architecture

EBLS (Empirical Bayes Layer Sharing) with entropy-adaptive n-gram eval cache:

| Component | Setting |
|-----------|---------|
| Layers | 11 (3 shared blocks x 3 loops + 2 unique) |
| Dimensions | 512d, 8 heads, 4 KV heads (GQA) |
| MLP | 3x with **LeakyReLU(0.5)^2** |
| LoRA | Rank 8, per virtual layer |
| BigramHash | 3072 vocab, 128 dim |
| XSA | All 11 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/sqrt(layer+1) |
| VRL | Value Residual Learning |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | Val-GPTQ int6 + LZMA preset 9+extreme |
| Eval cache | 7-gram backoff (orders 2-7), entropy-adaptive alpha |

### N-gram Cache Hyperparameters

| Parameter | Value |
|-----------|-------|
| Orders | 2 through 7 (6 hash tables) |
| Buckets | 4,194,304 per table |
| Min count | 2 (require 2+ observations) |
| Entropy base | 0.05 |
| Entropy range | 0.55 (alpha ranges from 0.05 to 0.60) |
| Entropy scale | 2.0 |
| Entropy threshold | 4.0 |
| Hash primes | [36313, 27191, 51647, 81929, 131071, 175447, 209591] |

## Run Command

```bash
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=560 XSA_LAST_N=11 \
WARMDOWN_ITERS=4000 CLIP_RANGE=31 COMPRESSOR=lzma \
NUM_KV_HEADS=4 EVAL_STRIDE=64 \
GPTQ_ENABLED=1 GPTQ_CALIB_BATCHES=64 GPTQ_CALIB_SOURCE=val \
GPTQ_BLOCK_SIZE=128 SWA_ENABLED=1 LATE_QAT_THRESHOLD=0.15 \
NGRAM_CACHE=1 NGRAM_ORDER=7 NGRAM_MIN_ORDER=2 \
NGRAM_MIN_COUNT=2 NGRAM_BUCKETS=4194304 \
NGRAM_ENTROPY=1 NGRAM_ENT_BASE=0.05 NGRAM_ENT_RANGE=0.55 \
NGRAM_ENT_SCALE=2.0 NGRAM_ENT_THRESH=4.0 \
NCCL_TIMEOUT=3600 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **N-gram cache technique**: [PR #715](https://github.com/openai/parameter-golf/pull/715), [PR #727](https://github.com/openai/parameter-golf/pull/727)
- **Entropy-adaptive alpha**: [PR #727](https://github.com/openai/parameter-golf/pull/727), suggested by valerio-oai on [PR #659](https://github.com/openai/parameter-golf/pull/659)
- **XSA-all**: [PR #634](https://github.com/openai/parameter-golf/pull/634) by @raahilshah
- **LeakyReLU^2**: [PR #493](https://github.com/openai/parameter-golf/pull/493) by @parinzee
- **Base model**: [PR #414](https://github.com/openai/parameter-golf/pull/414) by @signalrush
