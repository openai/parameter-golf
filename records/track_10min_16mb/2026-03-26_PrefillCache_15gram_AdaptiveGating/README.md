# Record: Distributed Prefill + Order-Adaptive Entropy Gating + 15-Gram Backoff + EBLS

**val_bpb: 0.4374** (3-seed mean, std 0.0003) | **~15.99 MB** | 8xH100 SXM

## Motivation

My background is in Bayesian statistics, so when I first saw the parameter golf challenge I immediately thought about it through the lens of shrinkage estimation — how do you get the most out of a parameter budget when you know some parameters carry redundant information? That's where EBLS came from: treat shared transformer blocks as a prior, and let per-layer LoRA deviations act as empirical Bayes corrections. The name "BayesGPT" stuck from there.

The n-gram cache story was more of a debugging accident. When I first got the multi-order backoff working (building on the great work from @deanbrr, @lukacf, @Asukabot0 and others), I noticed my 8-GPU eval scores were way worse than expected. Turns out ranks 1-7 were starting with empty caches — they'd never seen the tokens before their assigned window. The fix was obvious once I saw it: pre-fill each rank's hash tables with all preceding positions before scoring begins. That single change dropped BPB from 0.96 to 0.65.

The order-adaptive gating came from thinking about what the entropy threshold *should* be for different n-gram orders. A 15-gram match is almost certainly right — the model saw that exact 15-token sequence before. A bigram match could be noise. So higher orders should be trusted at lower entropy. @travispchen had a similar idea in PR #798; I extended it with a continuous interpolation across all 14 orders.

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

### 3-seed validation

| Seed | **Sliding + 15-gram BPB** | Artifact bytes |
|------|---------------------------|----------------|
| 1337 | **0.43706735** | 15,994,785 |
| 2024 | **0.43738561** | 15,949,881 |
| 2025 | **0.43768394** | 15,992,965 |
| **Mean** | **0.4374 (std 0.0003)** | |

### How we got here (ablation)

Each row adds one thing on top of the previous:

| Config | BPB | Delta | What changed |
|--------|-----|-------|--------------|
| Neural model only (no cache) | 1.1425 | — | EBLS baseline after GPTQ |
| + 7-gram backoff + prefill | 0.6565 | -0.486 | Cache + distributed prefill |
| + extend to 15-gram | 0.6189 | -0.038 | More context helps |
| + order-adaptive gating | **0.4374** | -0.182 | Trust high orders more |

The -0.181 from adaptive gating was the biggest single improvement. Uniform thresholds waste most of the high-order matches by being too conservative.

## What's novel here

### 1. Distributed Cache Pre-fill

The problem: when evaluating on 8 GPUs with sliding windows, each rank processes a contiguous chunk of token positions. Rank 0 gets the first ~1/8, rank 7 gets the last ~1/8. Without pre-fill, rank 7 starts scoring with an empty cache — it has no n-gram statistics from the first 7/8 of the data. This is a massive handicap.

The fix: before scoring begins, each rank pre-populates its n-gram hash tables with ALL token positions preceding its window, using vectorized numpy. No NCCL needed — each rank independently reads the validation tokens and hashes them.

```python
# Pseudocode: rank k fills tables for positions 0..start_of_rank_k
for order in range(min_order, max_order+1):
    ctx_hash = hash(val_tokens[pos-order+1:pos])
    full_hash = hash(ctx_hash, val_tokens[pos])
    ctx_tables[order][ctx_hash % buckets] += 1
    full_tables[order][full_hash % buckets] += 1
```

This gives **mathematically identical** results to single-GPU sequential evaluation. It's purely a distributed implementation detail. Pre-fill takes ~22-164s depending on rank (rank 7 has the most to fill).

**Impact**: 0.96 BPB without prefill → 0.65 BPB with prefill (-0.31 BPB).

### 2. Order-Adaptive Entropy Gating

Standard entropy gating uses one threshold for all n-gram orders. But a 15-gram match and a bigram match are very different signals — the 15-gram is almost certainly correct while the bigram could easily be wrong.

Our approach: per-order thresholds that interpolate linearly from aggressive (center=2.5 for 15-grams) to conservative (center=4.5 for bigrams):

```
alpha(order, H) = base + range * sigmoid(scale * (H - center(order)))
center(order) = 4.5 - (order - 2) / (15 - 2) * (4.5 - 2.5)
```

Inspired by @travispchen's per-order thresholds in PR #798. We generalize to continuous interpolation across all 14 orders.

**Impact**: 0.6189 BPB (uniform threshold) → 0.4374 BPB (-0.181 BPB).

## Technical details

### N-gram cache (eval-time, backward-looking only)

Multi-order backoff cache built during sliding window evaluation. Builds on the framework from @lukacf (PR #702), @Asukabot0 (PR #727), @hypery11 (PR #788).

1. **14 hash tables** for orders 2-15 (4M buckets each)
2. **Backoff**: try highest order first, fall back on miss (need min_count=2)
3. **Adaptive blending**: alpha varies by order and model entropy (see above)
4. **Strictly causal**: cache updated with true token only *after* the model scores it
5. **Distributed pre-fill**: each rank pre-populates from preceding positions

### Training architecture (EBLS)

The underlying model uses Empirical Bayes Layer Sharing — 3 shared transformer blocks looped 3x for 9 effective layers + 2 unique layers = 11 total. Per-virtual-layer LoRA (rank 8) provides the deviation from the shared prior. This saves enough parameters to fit everything in 16MB with int6 GPTQ + LZMA.

| Component | Setting |
|-----------|---------|
| Layers | 11 (3 shared x 3 loops + 2 unique) |
| Dims | 512d, 8 heads, 4 KV heads (GQA) |
| MLP | 3x with LeakyReLU(0.5)^2 |
| LoRA | Rank 8, per virtual layer |
| Quantization | Val-GPTQ int6 + LZMA preset 9+extreme |
| Weight avg | EMA(0.997) + SWA(every 50 steps) |
| XSA | All 11 layers |

### N-gram hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| Orders | 2-15 | Diminishing returns past 15 |
| Buckets | 4,194,304 | Fits in memory, low collision rate |
| Min count | 2 | Require repeated observation |
| Entropy base/range | 0.05 / 0.55 | alpha ranges 0.05-0.60 |
| Entropy scale | 2.0 | Sigmoid steepness |
| Threshold (bigrams) | 4.5 | Conservative — only when model confused |
| Threshold (15-grams) | 2.5 | Aggressive — trust long matches |

## Compliance

- [x] Training: 560s on 8xH100 (within 600s)
- [x] Eval: ~330s on 8xH100 (within 600s)
- [x] Artifacts under 16,000,000 bytes (max: 15,992,965)
- [x] Script: 1,451 lines (under 1,500)
- [x] No TTT on validation data
- [x] No training data access during eval
- [x] No oracle/min(NLL) selection — single blended prediction per token
- [x] Cache is strictly backward-looking (causal)
- [x] GPTQ calibration on val data within training window
- [x] Pre-fill only uses val_tokens[0..pos-1] — no future data

### Why pre-fill is legal

The pre-fill is an implementation optimization, not a new information source:

1. It produces **identical** n-gram tables as single-GPU sequential eval
2. At scored position p, cache contains only positions 0..p-1
3. Each token gets exactly one prediction (no oracle selection)
4. Model weights are frozen — no TTT
5. @valerio-oai [confirmed on PR #659](https://github.com/openai/parameter-golf/pull/659#issuecomment-2753280311) that n-gram caching "is not illegal" and suggested entropy-based gating as the legal path

## Run command

```bash
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=560 XSA_LAST_N=11 \
WARMDOWN_ITERS=4000 CLIP_RANGE=31 COMPRESSOR=lzma \
NUM_KV_HEADS=4 EVAL_STRIDE=64 \
GPTQ_ENABLED=1 GPTQ_CALIB_BATCHES=64 GPTQ_CALIB_SOURCE=val \
GPTQ_BLOCK_SIZE=128 SWA_ENABLED=1 LATE_QAT_THRESHOLD=0.15 \
NGRAM_CACHE=1 NGRAM_ORDER=15 NGRAM_MIN_ORDER=2 \
NGRAM_MIN_COUNT=2 NGRAM_BUCKETS=4194304 \
NGRAM_ENTROPY=1 NGRAM_ENT_BASE=0.05 NGRAM_ENT_RANGE=0.55 \
NGRAM_ENT_SCALE=2.0 NGRAM_ENT_THRESH=4.5 \
NGRAM_ENT_ADAPT=1 NGRAM_ENT_THRESH_LO=2.5 \
NCCL_TIMEOUT=3600 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits and acknowledgments

This builds on a ton of community work. The n-gram eval cache idea has been iterated on by many people and I want to make sure everyone gets proper credit:

- @deanbrr ([PR #659](https://github.com/openai/parameter-golf/pull/659)) — original n-gram cache concept (closed due to oracle gate, but the core idea started everything)
- @valerio-oai ([comment on #659](https://github.com/openai/parameter-golf/pull/659#issuecomment-2753280311)) — suggested entropy-based gating as the legal alternative
- @newjordan ([PR #674](https://github.com/openai/parameter-golf/pull/674)) — first legal implementation with fixed-alpha mixing
- @lukacf ([PR #702](https://github.com/openai/parameter-golf/pull/702)) — multi-order backoff + entropy-adaptive sigmoid formula (huge contribution)
- @Asukabot0 ([PR #727](https://github.com/openai/parameter-golf/pull/727)) — scaled to 7-gram, first sub-1.0 BPB
- @hypery11 ([PR #788](https://github.com/openai/parameter-golf/pull/788)) — 9-gram extension
- @travispchen ([PR #798](https://github.com/openai/parameter-golf/pull/798)) — per-order entropy thresholds (directly inspired our adaptive gating)
- @raahilshah ([PR #634](https://github.com/openai/parameter-golf/pull/634)) — XSA on all layers
- @parinzee ([PR #493](https://github.com/openai/parameter-golf/pull/493)) — LeakyReLU(0.5)^2
- @signalrush ([PR #414](https://github.com/openai/parameter-golf/pull/414)) — base GPTQ-lite + EMA + warmdown stack

**Our novel contributions**: distributed cache pre-fill, 15-gram extension, order-adaptive entropy gating with continuous interpolation, and the EBLS training architecture.
