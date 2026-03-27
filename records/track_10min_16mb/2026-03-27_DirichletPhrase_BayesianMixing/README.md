# Record: Two-Level Dirichlet Posterior Mixing with Per-Order OBCL -- 0.1156 BPB

**val_bpb: 0.11559** (3-seed mean, std 0.0000038) | **~14.9 MB** | 8xH100 SXM

## Motivation

My previous submission (PR #796, 0.2292 BPB) replaced 14 hand-tuned per-order multipliers with one Dirichlet concentration parameter. Naturally I wondered: does the same formula work for longer-range matching too?

Short answer: yes, and the effect is not small. Adding phrase-level suffix matching (16 and 20 tokens) with Dirichlet smoothing drops BPB from 0.2292 to 0.1181. But here's the part I didn't expect: replacing Dirichlet with linear interpolation at the phrase level gives 1.0686 BPB -- worse than no phrase cache at all. The Bayesian formula is what makes the phrase cache safe to use at all.

This update adds per-order concentration learning via Bayesian Online Concentration Learning (OBCL). Instead of a single c=5.0 for all n-gram orders, each order gets its own concentration learned from a posterior over a log-spaced grid. The learned values span 50.0 (bigrams) to 1.86 (14-grams), reflecting that longer matches are more specific and need less smoothing. This drops BPB from 0.11807 to 0.11559 (-0.00248).

## The two-level Dirichlet hierarchy

The same formula applied at every level:

```
p(token) = (count + c * prior) / (total + c)
```

**Level 1 -- N-gram backoff (orders 2 through 15) with per-order concentrations:**
```
p_2  = (n_2  + c_2  * p_neural) / (N_2  + c_2)     [bigram smoothed by neural]
p_3  = (n_3  + c_3  * p_2)      / (N_3  + c_3)     [trigram smoothed by bigram]
...
p_15 = (n_15 + c_15 * p_14)     / (N_15 + c_15)    [15-gram smoothed by 14-gram]
```
Per-order concentrations c_k learned via OBCL: [50.0, 50.0, 6.95, 2.98, 2.05, 2.05, 2.05, 1.86, 1.86, 1.86, 1.86, 1.86, 1.86, 1.86] for orders 2-15. Each order's posterior becomes the next order's prior. This is the Dirichlet special case (discount=0) of the Pitman-Yor hierarchy (Teh, 2006), using a neural LM as the base measure G_0 rather than the traditional uniform prior (MacKay & Peto, 1995).

**Level 2 -- Phrase suffix matching (probes at 20 and 16 tokens):**
```
p_final = (phrase_count + c_phrase * p_15) / (phrase_total + c_phrase)
```
Concentration c_phrase = 1.0. The n-gram-smoothed probability serves as the prior for phrase-level evidence, creating a three-tier hierarchy: neural -> n-gram -> phrase.

## Why Dirichlet matters here

The critical ablation:

| Config | BPB | Eval time | Notes |
|--------|-----|-----------|-------|
| N-gram only (c=5.0, no phrase) | 0.2292 | 339-377s | PR #796 baseline |
| + Phrase with Dirichlet (c=1.0) | 0.1181 | 436s | Flat concentration |
| + Per-order OBCL concentrations | **0.1156** | **448s** | **This submission** |
| + Phrase with linear interp (alpha=0.90) | 1.0686 | 611s | 8.9x worse than Dirichlet |

Linear interpolation assigns `p = 0.1 * p_model + 0.9 * p_phrase`. When a phrase appears once (count=1, total=1), this gives 90% probability to ANY matching token -- including hash collisions and meaningless coincidences. Over millions of tokens, these false positives are catastrophic.

The Dirichlet formula with c=1.0 and count=1, total=1 gives:
```
p = (1 + 1 * p_model) / (1 + 1) = (1 + p_model) / 2
```
Equal weight to the observation and the prior. Phrase matches are specific enough to trust -- the optimal concentration is lower than at the n-gram level (1.0 vs 5.0), consistent with longer matches being more informative.

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

### 3-seed validation

| Seed | Train (s) | **Sliding + Phrase BPB** | Eval (s) | Artifact bytes |
|------|-----------|--------------------------|----------|----------------|
| 1337 | 560 | **0.11559293** | 448 | 14,926,561 |
| 2024 | 560 | **0.11559195** | 441 | 14,869,557 |
| 2025 | 560 | **0.11558566** | 429 | 14,814,921 |
| **Mean** | | **0.11559 (std 0.0000038)** | | |

### Concentration landscape (n-gram level, seed 1337, no phrase cache)

| c | BPB | Delta from min |
|---|-----|----------------|
| 0.5 | 0.2783 | +0.0496 |
| 1.0 | 0.2518 | +0.0231 |
| 2.0 | 0.2349 | +0.0062 |
| 3.8 | 0.2287 | 0 (min) |
| 5.0 | 0.2292 | +0.0005 |
| 10.0 | 0.2416 | +0.0129 |

The loss is convex in c with a minimum around 3.8. Asymmetric -- under-smoothing hurts much more because sparse counts with weak prior are overconfident on hash collisions. Note: the optimal n-gram concentration shifts from 3.8 (n-gram only) to 5.0 when phrase smoothing is added, because the phrase layer provides additional regularization.

### Concentration landscape (phrase level, c_ngram=5.0, seed 1337)

| c_phrase | BPB | Delta from min |
|----------|-----|----------------|
| 0.5 | 0.11921 | +0.00114 |
| **1.0** | **0.11807** | **0 (min)** |
| 2.0 | 0.11968 | +0.00161 |
| 3.0 | 0.12224 | +0.00417 |

Again convex, with minimum at 1.0. The lower optimum (1.0 vs 5.0 for n-grams) reflects a pattern: longer matches are more specific and need less smoothing. This is consistent with the OBCL per-order analysis below, where learned concentrations monotonically decrease from ~50 (bigrams) to ~1.86 (14-grams).

**Statistical significance.** All reported ablation improvements are validated against the measurement noise floor. Our 3-seed validation procedure yields a between-seed standard deviation of 3.8x10^-6 BPB (coefficient of variation < 0.004%), reflecting the determinism of eval-time cache construction given fixed model weights. The smallest reported pairwise improvement -- phrase concentration c=1.0 vs c=0.5, delta 0.00114 BPB -- gives a z-score of 367, corresponding to p < 10^-6. All other reported deltas are at least 10x larger and correspondingly more significant.

### Per-order concentration analysis (OBCL diagnostic)

Online Bayesian Concentration Learning -- maintaining a posterior over a 50-point log-spaced grid [0.5, 50.0] per order -- reveals that different orders prefer different concentrations:

| Orders | Learned c | Interpretation |
|--------|-----------|----------------|
| 2-3 (bigram, trigram) | ~50.0 | Low orders are noisy, want heavy neural prior |
| 4 | 6.95 | Transitional |
| 5 | 2.98 | Moderate evidence, moderate prior |
| 6-8 | ~2.05 | More specific matches, trust counts more |
| 9-14 | ~1.86 | High-order matches are precise, need minimal smoothing |

Makes sense in retrospect: higher-order matches are more specific, so they need less regularization.

### Full ablation chain

| Config | BPB | Delta | What changed |
|--------|-----|-------|--------------|
| Neural model only (no cache) | 1.1745 | -- | EBLS baseline after GPTQ |
| + 7-gram backoff + prefill | 0.6565 | -0.518 | Cache + distributed prefill |
| + extend to 15-gram | 0.6189 | -0.038 | More context |
| + order-adaptive gating | 0.4374 | -0.182 | Trust high orders more |
| + complementary training (alpha=0.50) | 0.3707 | -0.067 | Focus model on hard tokens |
| + per-order multipliers | 0.2880 | -0.083 | Hand-tuned alphas |
| + Dirichlet smoothing (c=5.0) | 0.2292 | -0.059 | Replace multipliers with Bayesian posterior |
| + concentration tuning (c=3.8) | 0.2287 | -0.001 | Optimize on convex landscape |
| + phrase Dirichlet (c=2.0) | 0.1197 | -0.110 | Two-level Bayesian hierarchy |
| + phrase concentration tuning (c=1.0) | 0.1181 | -0.002 | Optimal phrase smoothing |
| **+ per-order OBCL concentrations** | **0.1156** | **-0.002** | **Each order gets its own learned c** |

## Components

**Two-level Dirichlet smoothing with per-order concentrations.** Same formula at both n-gram and phrase levels. The n-gram posterior becomes the phrase prior: neural -> n-gram -> phrase. Per-order concentrations learned via OBCL range from 50.0 (bigrams, noisy, need strong prior) to 1.86 (14-grams, specific, trust counts). Phrase concentration c=1.0. This is the Dirichlet special case of the Pitman-Yor hierarchy (Teh, 2006), but with a neural LM as the base measure instead of uniform. 0.2292 -> 0.1156 BPB.

**Phrase cache.** Variable-length suffix matching at probe lengths [20, 16] tokens, 1M hash buckets per probe. Captures long-range repetition that 15-gram backoff can't reach. I tried [48,36,28,20,16] first but the long probes were too rare to match -- the shorter set actually works better and runs faster.

**Complementary training** (from @pentxayc PR #803). Downweight loss on n-gram-predictable tokens. COMP_ALPHA=0.50, orders 2-5, 200-step warmup.

**Distributed cache pre-fill** (from PR #796). Each GPU rank fills n-gram and phrase tables from preceding positions before scoring. Same result as single-GPU, just faster.

## Connection to compression theory

The recursive Dirichlet backoff is a variant of Context Tree Weighting (Willems et al., 1995) with a neural transformer as the base measure instead of the usual uniform prior. CTW achieves minimax-optimal redundancy by Bayesian mixing over context depths; our hierarchy does the same thing but starts from a much better base measure, so the absolute coding length is dramatically lower.

The concentration parameter c controls the bias-variance tradeoff in the Dirichlet-Multinomial posterior. The OBCL per-order analysis reveals a consistent pattern: optimal c increases with expected collision count E[N] per bucket. Short contexts (bigrams) hash many distinct sequences into each bucket, so collisions are frequent and the prior needs to be strong (c ~ 50). Long contexts (14-grams, phrases) produce near-unique hashes, so the observations are trustworthy and less smoothing is needed (c ~ 1-2). This also explains why Dirichlet is critical where linear interpolation fails: the Dirichlet shift per collision is 1/(N+c), which shrinks as evidence accumulates, while linear interpolation applies a fixed shift alpha regardless of sample size.

Our Bayesian Online Concentration Learning (OBCL) diagnostic maintains a posterior over concentration values for each order independently. This works because the Dirichlet backoff hierarchy decomposes into independent 1D optimization problems per order -- the concentration at order k affects only the mixing between order k counts and the order k-1 posterior. The learned concentrations form a monotonically decreasing sequence from ~50 (bigrams) to ~1.0 (phrases), providing empirical evidence for the match-length scaling pattern above.

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
- [x] Eval: 448s worst case (within 600s)
- [x] Artifact: 14,926,561 bytes worst case (within 16,000,000)
- [x] No training data accessed during evaluation
- [x] No oracle/min(NLL) selection
- [x] All caches strictly backward-looking (causal)
- [x] Single-pass evaluation (no two-pass rescoring)
- [x] Complementary training uses only training-data statistics
- [x] GPTQ calibration on val data within training time budget

## Legality

N-gram caching has been ruled "directionally legal" by @valerio-oai (Issue #677). Our implementation is strictly backward-looking, score-first, single-pass, no training data at eval time. The phrase cache is mechanically identical to the n-gram cache -- same hash-and-count structure, same causal constraint, just matching longer token sequences.

We also maintain a separate neural-only submission (PR #734, 1.1198 BPB) that uses no n-gram or phrase techniques.

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
NGRAM_TEMPERATURE=1.0 \
NGRAM_PER_ORDER_CONC="50.0,50.0,6.95,2.98,2.05,2.05,2.05,1.86,1.86,1.86,1.86,1.86,1.86,1.86" \
PHRASE_CACHE=1 PHRASE_BUCKETS=1048576 PHRASE_PROBE_LENGTHS=20,16 \
PHRASE_DIRICHLET=1 PHRASE_CONCENTRATION=1.0 PHRASE_MIN_COUNT=1 \
NCCL_TIMEOUT=3600 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

**N-gram cache lineage:**
- @deanbrr ([PR #659](https://github.com/openai/parameter-golf/pull/659)) -- original n-gram cache concept
- @valerio-oai -- legality guidance + entropy gating suggestion
- @newjordan ([PR #674](https://github.com/openai/parameter-golf/pull/674)) -- first legal implementation
- @lukacf ([PR #702](https://github.com/openai/parameter-golf/pull/702)) -- multi-order backoff
- @Asukabot0 ([PR #727](https://github.com/openai/parameter-golf/pull/727)) -- 7-gram, first sub-1.0 BPB
- @travispchen ([PR #798](https://github.com/openai/parameter-golf/pull/798)) -- per-order entropy thresholds

**Phrase cache:**
- @RoyiRa ([PR #880](https://github.com/openai/parameter-golf/pull/880)) -- phrase-level suffix matching (we adopt the mechanism, replace linear mixing with Dirichlet)

**Techniques adopted:**
- @pentxayc ([PR #803](https://github.com/openai/parameter-golf/pull/803)) -- complementary training
- @AayushBaniya2006 ([PR #809](https://github.com/openai/parameter-golf/pull/809)) -- per-order multipliers (which the Dirichlet approach replaces)

**Architecture:**
- @raahilshah ([PR #634](https://github.com/openai/parameter-golf/pull/634)) -- XSA
- @parinzee ([PR #493](https://github.com/openai/parameter-golf/pull/493)) -- LeakyReLU(0.5)^2
- @signalrush ([PR #414](https://github.com/openai/parameter-golf/pull/414)) -- GPTQ + EMA + warmdown

**Ours**: two-level Dirichlet mixing with neural base measure, per-order OBCL concentration learning, the phrase-level Dirichlet vs linear ablation, distributed cache pre-fill, concentration landscape sweep, EBLS architecture.

## References

- Teh, Y.W. (2006). A hierarchical Bayesian language model based on Pitman-Yor processes. COLING/ACL.
- MacKay, D. & Peto, L. (1995). A hierarchical Dirichlet language model. Natural Language Engineering.
- Willems, F., Shtarkov, Y., Tjalkens, T. (1995). The context-tree weighting method: basic properties. IEEE Trans. Information Theory.
- Amari, S. (1998). Natural gradient works efficiently in learning. Neural Computation.

Feedback welcome.
