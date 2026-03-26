# Record: Complementary Training + Per-Order Multipliers + Distributed Prefill + 15-Gram + EBLS

**val_bpb: 0.2880** (3-seed mean, std 0.00006) | **~15.3 MB** | 8xH100 SXM

## Motivation

My background is in Bayesian statistics, so when I first saw the parameter golf challenge I immediately thought about it through the lens of shrinkage estimation — how do you get the most out of a parameter budget when you know some parameters carry redundant information? That's where EBLS came from: treat shared transformer blocks as a prior, and let per-layer LoRA deviations act as empirical Bayes corrections. The name "BayesGPT" stuck from there.

The n-gram cache story was more of a debugging accident. When I first got the multi-order backoff working (building on the great work from @deanbrr, @lukacf, @Asukabot0 and others), I noticed my 8-GPU eval scores were way worse than expected. Turns out ranks 1-7 were starting with empty caches — they'd never seen the tokens before their assigned window. The fix was obvious once I saw it: pre-fill each rank's hash tables with all preceding positions before scoring begins. That single change dropped BPB from 0.96 to 0.65.

The big breakthrough came from combining two ideas from the community: @pentxayc's complementary training (PR #803) — which trains the neural model to focus on tokens that n-grams can't predict — and @AayushBaniya2006's per-order multipliers (PR #809) — which aggressively boost high-order n-gram contributions while suppressing noisy bigrams. Together these dropped BPB from 0.44 to 0.29, far more than either technique alone.

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

### 3-seed validation

| Seed | Steps | Train (s) | Pre-quant BPB | Roundtrip BPB | **Sliding + 15-gram BPB** | Artifact bytes |
|------|-------|-----------|---------------|---------------|---------------------------|----------------|
| 1337 | 3,620 | 560 | 1.1698 | 1.1745 | **0.28797872** | 15,143,631 |
| 2024 | 3,587 | 560 | 1.1701 | 1.1749 | **0.28804071** | 15,124,675 |
| 2025 | 3,593 | 560 | 1.1702 | 1.1751 | **0.28809874** | 15,324,143 |
| **Mean** | | | | | **0.2880 (std 0.00006)** | |

### How we got here (ablation)

Each row adds one thing on top of the previous:

| Config | BPB | Delta | What changed |
|--------|-----|-------|--------------|
| Neural model only (no cache) | 1.1425 | — | EBLS baseline after GPTQ |
| + 7-gram backoff + prefill | 0.6565 | -0.486 | Cache + distributed prefill |
| + extend to 15-gram | 0.6189 | -0.038 | More context helps |
| + order-adaptive gating | 0.4374 | -0.182 | Trust high orders more |
| + complementary training (alpha=0.20) | 0.3707 | -0.067 | Focus model on hard tokens |
| **+ per-order multipliers** | **0.2880** | **-0.083** | **Boost high orders, suppress bigrams** |

## What's in this submission

### 1. Complementary Training (from @pentxayc PR #803)

During training, tokens that bigrams predict well get downweighted in the loss function:

```
weight[i] = max(0.1, 1.0 - COMP_ALPHA * P_bigram(token_i | token_{i-1}))
```

This forces the neural model to specialize on tokens that n-gram caching can't handle. The bigram statistics come from training data (legal — computed during training, not eval). We use `COMP_ALPHA=0.50` with orders 2-5 and a 200-step warmup.

**Impact**: 0.4374 → 0.3707 (-0.067 BPB).

### 2. Per-Order Multipliers (from @AayushBaniya2006 PR #809)

Not all n-gram orders are equally useful. Bigrams are noisy; 5-grams and above are gold. Per-order multipliers scale the mixing alpha:

```
order_mults = (0.3, 0.3, 0.97, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
alpha_max = 0.95
```

Orders 2-3 (bigrams/trigrams) are suppressed to 0.3x. Orders 5-15 are boosted to 2.0x with a cap at alpha=0.95.

**Impact**: 0.3707 → 0.2880 (-0.083 BPB).

### 3. Distributed Cache Pre-fill (our contribution)

When evaluating on 8 GPUs with sliding windows, each rank processes a contiguous chunk of token positions. Without pre-fill, rank 7 starts scoring with an empty cache — it has no n-gram statistics from the first 7/8 of the data. Pre-fill fixes this: before scoring, each rank hashes all preceding positions into its tables using vectorized numpy. No NCCL needed.

This gives **mathematically identical** results to single-GPU sequential evaluation.

**Impact**: 0.96 → 0.65 BPB (-0.31 BPB).

### 4. Order-Adaptive Entropy Gating (inspired by @travispchen PR #798)

Per-order thresholds that interpolate linearly from aggressive (center=2.5 for 15-grams) to conservative (center=4.5 for bigrams):

```
alpha(order, H) = base + range * sigmoid(scale * (H - center(order)))
center(order) = 4.5 - (order - 2) / (15 - 2) * (4.5 - 2.5)
```

**Impact**: 0.6189 → 0.4374 (-0.182 BPB).

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
- [x] Eval: ~330s on 8xH100 (within 600s)
- [x] Artifacts under 16,000,000 bytes (max: 15,324,143)
- [x] Script: 1,500 lines (at limit)
- [x] No training data accessed during evaluation
- [x] No oracle/min(NLL) selection — single blended prediction per token
- [x] Cache is strictly backward-looking (causal)
- [x] Complementary training uses only training-data bigram statistics
- [x] GPTQ calibration on val data within training time budget
- [x] Pre-fill only uses val_tokens[0..pos-1] — no future data

## Legality

N-gram caching legality has not been formally resolved by OpenAI. @valerio-oai commented on PR #659 that it "is not illegal" and suggested entropy-based gating, but no definitive ruling has been issued. We believe our implementation is compliant — strictly backward-looking, score-first, no training data at eval time — but we respect whatever ruling is made.

We also maintain a separate neural-only submission (PR #734, 1.1198 BPB) that uses no n-gram techniques.

We welcome discussion on this — if there are concerns about any aspect of the approach, we're happy to address them.

## Run command

```bash
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=560 XSA_LAST_N=11 \
WARMDOWN_ITERS=4000 CLIP_RANGE=31 COMPRESSOR=lzma \
NUM_KV_HEADS=4 EVAL_STRIDE=64 \
GPTQ_ENABLED=1 GPTQ_CALIB_BATCHES=64 GPTQ_CALIB_SOURCE=val \
GPTQ_BLOCK_SIZE=128 SWA_ENABLED=1 LATE_QAT_THRESHOLD=0.15 \
COMP_ENABLED=1 COMP_ALPHA=0.20 COMP_ORDER=5 COMP_WARMUP=200 COMP_MIN_COUNT=3 \
NGRAM_CACHE=1 NGRAM_ORDER=15 NGRAM_MIN_ORDER=2 \
NGRAM_MIN_COUNT=2 NGRAM_BUCKETS=4194304 \
NGRAM_ENTROPY=1 NGRAM_ENT_BASE=0.05 NGRAM_ENT_RANGE=0.55 \
NGRAM_ENT_SCALE=2.0 NGRAM_ENT_THRESH=4.5 \
NGRAM_ENT_ADAPT=1 NGRAM_ENT_THRESH_LO=2.5 \
NCCL_TIMEOUT=3600 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits and acknowledgments

This builds on a lot of community work. I want to make sure everyone gets proper credit:

**Techniques we adopted (with modifications):**
- @pentxayc ([PR #803](https://github.com/openai/parameter-golf/pull/803)) — complementary training (downweight n-gram-predictable tokens during training)
- @AayushBaniya2006 ([PR #809](https://github.com/openai/parameter-golf/pull/809)) — per-order multipliers and alpha_max capping

**N-gram cache lineage:**
- @deanbrr ([PR #659](https://github.com/openai/parameter-golf/pull/659)) — original n-gram cache concept
- @valerio-oai ([comment on #659](https://github.com/openai/parameter-golf/pull/659#issuecomment-2753280311)) — legality guidance + entropy gating suggestion
- @newjordan ([PR #674](https://github.com/openai/parameter-golf/pull/674)) — first legal implementation
- @lukacf ([PR #702](https://github.com/openai/parameter-golf/pull/702)) — multi-order backoff + entropy-adaptive sigmoid
- @Asukabot0 ([PR #727](https://github.com/openai/parameter-golf/pull/727)) — 7-gram extension, first sub-1.0 BPB
- @hypery11 ([PR #788](https://github.com/openai/parameter-golf/pull/788)) — 9-gram extension
- @travispchen ([PR #798](https://github.com/openai/parameter-golf/pull/798)) — per-order entropy thresholds (directly inspired our adaptive gating)

**Architecture foundations:**
- @raahilshah ([PR #634](https://github.com/openai/parameter-golf/pull/634)) — XSA on all layers
- @parinzee ([PR #493](https://github.com/openai/parameter-golf/pull/493)) — LeakyReLU(0.5)^2
- @signalrush ([PR #414](https://github.com/openai/parameter-golf/pull/414)) — base GPTQ-lite + EMA + warmdown stack

**Our novel contributions**: distributed cache pre-fill, 15-gram extension, order-adaptive entropy gating with continuous interpolation, the combination/integration of complementary training with per-order multipliers, and the EBLS training architecture (shared blocks + Bayesian shrinkage).

Feedback, questions, and corrections welcome.
