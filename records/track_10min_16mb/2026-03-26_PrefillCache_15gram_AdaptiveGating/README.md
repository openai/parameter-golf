# Record: Empirical Bayes N-gram Mixing -- 0.2292 BPB

**val_bpb: 0.22923** (3-seed mean, std 0.000005) | **~14.9 MB** | 8xH100 SXM

## Motivation

My background is in Bayesian statistics, so when I first saw the parameter golf challenge I immediately thought about it through the lens of shrinkage estimation. That's where EBLS came from: treat shared transformer blocks as a prior, and let per-layer LoRA deviations act as empirical Bayes corrections.

The n-gram cache story was more of a debugging accident. When I first got the multi-order backoff working (building on the great work from @deanbrr, @lukacf, @Asukabot0 and others), I noticed my 8-GPU eval scores were way worse than expected. Turns out ranks 1-7 were starting with empty caches. The fix was obvious once I saw it: pre-fill each rank's hash tables with all preceding positions. That single change dropped BPB from 0.96 to 0.65.

My previous submission (0.2880) used hand-tuned per-order multipliers to mix the n-gram cache with the neural model. It worked but felt wrong: 14 parameters with no theoretical justification, each needing manual tuning. I asked whether there was a principled way to combine a neural prior with count evidence, and the answer turned out to be textbook Bayesian statistics.

## The Dirichlet smoothing formula

```
p(token) = (ngram_count + c * neural_prob) / (total + c)
```

This is the Dirichlet-Multinomial posterior predictive. The neural model acts as an informative Dirichlet prior, n-gram counts provide the multinomial likelihood, and the concentration `c` controls how much to trust sparse evidence vs the neural model. Applied recursively from bigram to 15-gram, where each order's smoothed estimate becomes the next order's prior.

A single global concentration (c=5.0) replaces the 14 hand-tuned multipliers and improves BPB from 0.2880 to 0.2292. I was surprised that one parameter does better than fourteen.

This is the Dirichlet special case (discount=0) of the Pitman-Yor hierarchy (Teh, 2006), using a neural LM as the base measure rather than the traditional uniform/unigram prior (MacKay & Peto, 1995). It is a sibling to Kneser-Ney smoothing, not a generalization.

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

### 3-seed validation

| Seed | Steps | Train (s) | Roundtrip BPB | **Sliding + 15-gram BPB** | Artifact bytes |
|------|-------|-----------|---------------|---------------------------|----------------|
| 1337 | 3,430 | 560 | 1.1770 | **0.22922259** | 14,845,997 |
| 2024 | 3,430 | 560 | 1.1801 | **0.22923179** | 14,860,181 |
| 2025 | 3,430 | 560 | 1.1727 | **0.22922912** | 14,846,933 |
| **Mean** | | | | **0.22923 (std 0.000005)** | |

### Ablation chain

Each row adds one thing on top of the previous:

| Config | BPB | Delta | What changed |
|--------|-----|-------|--------------|
| Neural model only (no cache) | 1.1745 | -- | EBLS baseline after GPTQ |
| + 7-gram backoff + prefill | 0.6565 | -0.518 | Cache + distributed prefill |
| + extend to 15-gram | 0.6189 | -0.038 | More context helps |
| + order-adaptive gating | 0.4374 | -0.182 | Trust high orders more |
| + complementary training (alpha=0.50) | 0.3707 | -0.067 | Focus model on hard tokens |
| + per-order multipliers | 0.2880 | -0.083 | Boost high orders, suppress bigrams |
| **+ Dirichlet smoothing (c=5.0)** | **0.2292** | **-0.059** | **Replace multipliers with Bayesian posterior** |

### Concentration sweep (seed 1337)

| c | BPB | Note |
|---|-----|------|
| 0.5 | 0.2783 | Too little prior, sparse counts dominate |
| 1.0 | 0.2518 | Moderate |
| 2.0 | 0.2349 | Improving |
| 5.0 | 0.2292 | Diminishing returns (best observed) |
| 10.0 | 0.2416 | Over-smoothed |

## What's in this submission

### 1. Dirichlet-Multinomial Smoothing (new in this update)

Instead of per-order multipliers, use the Bayesian posterior predictive:

```
p_k(token) = (count_k + c * p_{k-1}(token)) / (total_k + c)
```

where p_0 is the neural model's softmax output. Applied recursively from order 2 to order 15. Concentration c=5.0 for all orders.

**Impact**: 0.2880 -> 0.2292 (-0.059 BPB) while removing 14 tuned parameters.

### 2. Complementary Training (from @pentxayc PR #803)

Downweight loss on n-gram-predictable tokens so the neural model specializes where caching can't help. `COMP_ALPHA=0.50`, orders 2-5, 200-step warmup.

**Impact**: 0.4374 -> 0.3707 (-0.067 BPB).

### 3. Distributed Cache Pre-fill (our contribution)

Each GPU rank pre-populates 15-gram hash tables from all preceding validation positions before scoring. Gives mathematically identical results to single-GPU sequential evaluation.

**Impact**: 0.96 -> 0.65 BPB (-0.31 BPB).

## Training architecture (EBLS)

3 shared transformer blocks looped 3x for 9 effective layers + 2 unique layers = 11 total. Per-virtual-layer LoRA (rank 8) provides deviation from the shared prior.

| Component | Setting |
|-----------|---------|
| Layers | 11 (3 shared x 3 loops + 2 unique) |
| Dims | 512d, 8 heads, 4 KV heads (GQA) |
| MLP | 3x with LeakyReLU(0.5)^2 |
| LoRA | Rank 8, per virtual layer |
| Quantization | Val-GPTQ int6 + LZMA preset 9+extreme |
| Weight avg | EMA(0.997) + SWA(every 50 steps) |
| XSA | All 11 layers |
| VRL | Layers 1-10 |
| Params | 27,124,848 |

## Compliance

- [x] Training: 560s on 8xH100 (within 600s)
- [x] Eval: 366s max across seeds (within 600s)
- [x] Artifacts under 16,000,000 bytes (max: 14,860,181)
- [x] No training data accessed during evaluation
- [x] No oracle/min(NLL) selection
- [x] Cache is strictly backward-looking (causal)
- [x] Single-pass evaluation (no two-pass rescoring)
- [x] Complementary training uses only training-data statistics
- [x] GPTQ calibration on val data within training time budget

## Legality

N-gram caching has been ruled "directionally legal" by @valerio-oai. Our implementation is strictly backward-looking, score-first, single-pass, no training data at eval time.

We also maintain a separate neural-only submission (PR #734, 1.1198 BPB) that uses no n-gram techniques.

## Run command

```bash
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=560 XSA_LAST_N=11 \
WARMDOWN_ITERS=4000 CLIP_RANGE=31 COMPRESSOR=lzma \
NUM_KV_HEADS=4 EVAL_STRIDE=64 \
GPTQ_ENABLED=1 GPTQ_CALIB_BATCHES=64 GPTQ_CALIB_SOURCE=val \
GPTQ_BLOCK_SIZE=128 SWA_ENABLED=1 LATE_QAT_THRESHOLD=0.15 \
COMP_ENABLED=1 COMP_ALPHA=0.50 COMP_ORDER=5 COMP_WARMUP=200 COMP_MIN_COUNT=3 \
NGRAM_CACHE=1 NGRAM_ORDER=15 NGRAM_MIN_ORDER=2 \
NGRAM_BUCKETS=4194304 NGRAM_DIRICHLET=1 NGRAM_CONCENTRATION=5.0 \
NCCL_TIMEOUT=3600 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

This builds on a lot of community work:

**N-gram cache lineage:**
- @deanbrr ([PR #659](https://github.com/openai/parameter-golf/pull/659)) -- original n-gram cache concept
- @valerio-oai -- legality guidance + entropy gating suggestion
- @newjordan ([PR #674](https://github.com/openai/parameter-golf/pull/674)) -- first legal implementation
- @lukacf ([PR #702](https://github.com/openai/parameter-golf/pull/702)) -- multi-order backoff
- @Asukabot0 ([PR #727](https://github.com/openai/parameter-golf/pull/727)) -- 7-gram, first sub-1.0 BPB
- @travispchen ([PR #798](https://github.com/openai/parameter-golf/pull/798)) -- per-order entropy thresholds

**Techniques adopted:**
- @pentxayc ([PR #803](https://github.com/openai/parameter-golf/pull/803)) -- complementary training
- @AayushBaniya2006 ([PR #809](https://github.com/openai/parameter-golf/pull/809)) -- per-order multipliers (which the Dirichlet approach now replaces)

**Architecture:**
- @raahilshah ([PR #634](https://github.com/openai/parameter-golf/pull/634)) -- XSA
- @parinzee ([PR #493](https://github.com/openai/parameter-golf/pull/493)) -- LeakyReLU(0.5)^2
- @signalrush ([PR #414](https://github.com/openai/parameter-golf/pull/414)) -- GPTQ + EMA + warmdown

**Our contributions**: distributed cache pre-fill, Dirichlet-Multinomial smoothing with neural base measure, and the EBLS training architecture.

Feedback welcome.
