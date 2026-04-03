# Vocabulary Optimized Language Model for Parameter Golf

**Result: 1.1207 BPB** (mean of 3 seeds: 1.1199, 1.1213, 1.1208 | std: 0.0007) | Artifact: 15.97 MB | Architecture: V=8192, 512d, 11 layers, GQA + BigramHash

## Summary

This submission explores vocabulary size as an underutilized scaling variable. By increasing the BPE vocabulary from 1,024 to 8,192 tokens — guided by entropy decomposition analysis of FineWeb data — and rebalancing the architecture to accommodate the larger embedding table, this approach achieves 1.1207 BPB (3-seed mean) using standard training infrastructure with a single additional feature (BigramHash). The final architecture matches the SOTA's shape (512d, 11 layers, 8 heads, 4 KV heads) with the vocabulary trade: MLP capacity reduced from 3× to 2× to fund the 8× larger embedding table. Systematic ablation of 20+ techniques revealed that BigramHash is the only per-layer feature that provides genuinely complementary information at V=8192 — all others provide compensatory information that the vocabulary has already absorbed.

## Core Idea: Vocabulary as Entropy Filter

The competition baseline uses a 1,024-token BPE vocabulary, allocating only 2.7% of model parameters to the embedding layer. Analysis of production language models (GPT-2 through Qwen2.5) shows this is dramatically low for a model at this scale — Qwen2.5-0.5B, the smallest competitive production model at 494M parameters, allocates 27.6% to embeddings with a 151K vocabulary, and GPT-2 Small (124M params) allocates 31.1%.

Mutual information spectrum measurements of FineWeb at different vocabulary sizes revealed that vocabulary acts as a filter on the information structure of tokenized text:

- **V=1024**: 63% of baseline-corrected MI is concentrated in the adjacent band (lags 1-2), with 77% in lags 1-6. Long-range signal is weak (26% negative PMI pairs at distance 64).
- **V=8192**: Adjacent concentration drops to 48% (63% in lags 1-6), redistributing MI to phrasal and sentence bands. Long-range corrected MI is ~2.2× stronger with 2× fewer noise pairs (12% negative at distance 64).

The larger vocabulary absorbs local character patterns into individual tokens, making the remaining inter-token structure richer and more learnable. The information spectrum follows a power law decay I(lag) = 0.563 × lag^(-0.619) (R² = 0.91), with vocabulary size controlling the distribution across scales.

## Architecture

| Component | Setting | Attribution |
|-----------|---------|-------------|
| **Vocabulary** | **8,192 BPE (entropy-optimized)** | **This work** |
| **Architecture** | **512d, 11 layers, 8 heads, 4 KV heads (GQA)** | **This work (sweep-optimized)** |
| **MLP** | **2× (1024) with LeakyReLU(0.5)²** | Activation: [#493](https://github.com/KellerJordan/modded-nanogpt/pull/493) @parinzee |
| Embedding | Tied, 17.2% of parameters | This work (scaling law analysis) |
| RoPE | Partial (16/64 dims) | [#315](https://github.com/KellerJordan/modded-nanogpt/pull/315) @jfprincz |
| LN Scale | 1/√(layer+1) | [#315](https://github.com/KellerJordan/modded-nanogpt/pull/315) @jfprincz |
| SmearGate | Position-mixing gate | [#65](https://github.com/KellerJordan/modded-nanogpt/pull/65) @aquariouseworkman |
| U-Net skips | Encoder-decoder connections | [#289](https://github.com/KellerJordan/modded-nanogpt/pull/289) |
| Weight avg | EMA(0.997) + Tight SWA(every 50) | [#401](https://github.com/KellerJordan/modded-nanogpt/pull/401) @newjordan |
| Quantization | Full Hessian GPTQ int6 (AR self-gen calibration) | GPTQ: [#535](https://github.com/KellerJordan/modded-nanogpt/pull/535) @raahilshah |
| Compression | LZMA preset=9 | [#160](https://github.com/KellerJordan/modded-nanogpt/pull/160) @ChaseWNorton |
| Optimizer | Parallel Muon + Parameter Banking | [#399](https://github.com/KellerJordan/modded-nanogpt/pull/399) @abaybektursun |
| Late QAT | STE at LR scale < 0.15 | [#286](https://github.com/KellerJordan/modded-nanogpt/pull/286) @chris-buckley |
| Selective pruning | ±1 values by reconstruction error | [#609](https://github.com/KellerJordan/modded-nanogpt/pull/609) @saml212 |
| Flash Attention 3 | Hopper warp-specialized kernels | [#122](https://github.com/KellerJordan/modded-nanogpt/pull/122) @mtybadger |
| Warmdown | 3,500 iterations (cosine) | [#364](https://github.com/KellerJordan/modded-nanogpt/pull/364) @shikhar1729 |

The contribution of this work is the vocabulary rebalancing and architecture optimization; all other training infrastructure was adapted from the collaborative work of the Parameter Golf community, primarily consolidated in [PR #1019](https://github.com/KellerJordan/modded-nanogpt/pull/1019) by @raahilshah.

## Key Results

| Version | BPB | Δ Baseline | Artifact | Fits? | Description |
|---------|----:|----------:|--------:|:-----:|-------------|
| Baseline | 1.2244 | — | 15.86 MB | ✓ | V=1024, 512d, 9 layers |
| Vocab v1a | 1.1923 | -0.032 | 18.47 MB | ✗ | V=8192, 384d/14L, baseline code only |
| Vocab v2a 14L | 1.1399 | -0.085 | 13.04 MB | ✓ | V=8192, 384d/14L + SOTA infrastructure |
| Vocab v2a 16L | 1.1319 | -0.093 | 14.49 MB | ✓ | V=8192, 384d/16L + SOTA infrastructure |
| Vocab v2a 448d | 1.1244 | -0.100 | 15.84 MB | ✓ | V=8192, 448d/13L MHA + SOTA infrastructure |
| **Vocab v2a 512d** | **1.1212** | **-0.103** | **15.72 MB** | **✓** | **V=8192, 512d/11L GQA + SOTA infrastructure** |
| **Vocab v2b** | **1.1199** | **-0.105** | **15.99 MB** | **✓** | **V=8192, 512d/11L GQA + BigramHash** |
| **Vocab v9** | **1.1207** | **-0.104** | **15.97 MB** | **✓** | **v2b cleaned + shard preloading (submission, 3-seed mean)** |
| Vocab v4 int7 | 1.1156 | -0.109 | 18.51 MB | ✗ | v2b + int7 mlp/attn (±63). Best BPB, can't fit |
| Vocab v7 +bank QAT | 1.1208 | -0.104 | 15.98 MB | ✓ | v2b + bank QAT at int6. Neutral result |
| SOTA #1 | 1.1147 | -0.110 | ~15.9 MB | ✓ | V=1024, 512d/11L GQA, AR Self-Gen GPTQ + XSA (leaderboard, Mar 25 2026) |

*Note: Vocab v1a uses baseline evaluation (no sliding window). All v2a results use sliding window evaluation (stride=64, seq_len=2048). BPB values represent the competition-comparable score for each version's evaluation method.*

### Compute Efficiency: 10 Minutes vs 4 Hours

A non-record baseline submission trained the V=1024 architecture for 4 hours on 8×H100s, achieving 1.2074 BPB — 24× this model's training budget. The V=8192 model achieves 1.1212 BPB in 10 minutes:

| Config | Training Time | BPB | Compute Multiple |
|--------|-------------:|----:|:----------------:|
| V=1024, 4-hour extended | 4 hours | 1.2074 | 24× |
| V=8192, 10-minute (this work) | 10 minutes | 1.1212 | 1× |

The V=8192 model achieves 0.086 BPB better in 1/24th the training time. Note that the 4-hour baseline uses the original 9-layer architecture without the community's subsequent optimizations (GQA, 11 layers, Muon, etc.). A fairer comparison would be the SOTA V=1024 stack at 10 minutes (1.1147 BPB), against which the V=8192 approach is within 0.0060 BPB (3-seed mean). Nevertheless, the comparison illustrates that vocabulary quality has outsized impact relative to training compute: at V=1024, 24× more compute produces only modest improvement, while rebalancing the vocabulary achieves a larger gain in standard time.

### The Vocabulary Trade

The final submission (512d/11L GQA) shares the SOTA's architectural shape but makes a different parameter allocation:

| Component | SOTA (V=1024) | V=8192 (this work) | Difference |
|-----------|-------------:|-------------:|----------:|
| Embedding (tied) | 524,288 | 4,194,304 | +3,670,016 |
| MLP (per layer) | 1,572,864 (3×) | 1,048,576 (2×) | -524,288 |
| MLP (11 layers) | 17,301,504 | 11,534,336 | -5,767,168 |
| KV attention | 2,883,584 | 2,883,584 | 0 |
| Q/O attention | 5,767,168 | 5,767,168 | 0 |
| **Net effect** | | | **-2,097,152** |

The design trades 5.8M MLP parameters for 3.7M embedding parameters — a net reduction of 2.1M params. The larger embedding table captures local character patterns directly in the token representation, reducing the work the MLP layers need to do. The result is within 0.0060 BPB of the leaderboard SOTA (1.1147, as of March 25, 2026) despite having 2.1M fewer parameters and relying on a single auxiliary feature (BigramHash) rather than the full per-layer feature stack (XSA, VE, MTP — see ablation below).

## How This Model Was Built

### 1. Entropy Analysis of FineWeb

Custom tools measured mutual information at different token-lag distances across FineWeb validation data at multiple vocabulary sizes (512 to 16,384). The corrected MI spectrum revealed that vocabulary size controls how information is distributed across scales:

| Vocab | Adjacent (lag 1-2) | Local (3-6) | Phrasal (7-16) | Sentence (17-48) | Paragraph (49-128) | Document (129-512) |
|------:|-------------------:|------------:|---------------:|-----------------:|-------------------:|-------------------:|
| 1,024 | 62.7% | 14.4% | 8.6% | 8.1% | 3.7% | 2.6% |
| 8,192 | 48.3% | 14.4% | 13.4% | 13.2% | 6.2% | 4.5% |

*V=1024 measured on 50M tokens (entropy_histogram_full.json); V=8192 on 20M tokens (entropy_histogram_8192_official.json). Both use production tokenizers with shuffled-baseline correction.*

At V=8192, the adjacent band drops by 14 percentage points (from 63% to 48%), with the freed MI redistributed primarily to phrasal (+4.8pp) and sentence (+5.1pp) bands. The larger vocabulary exposes substantially more mid-range structure for the model to learn. The corrected MI follows a power law decay: I(lag) = 0.563 × lag^(-0.619), R² = 0.91, measured across 60 lag points on 20M tokens.

**PMI Distribution Quality (surprisal_variance.py).** Beyond average MI, the *shape* of the pointwise mutual information distribution determines how learnable the token relationships are. PMI statistics were measured at 18 lag distances for both vocabularies:

| Metric | V=1024 (lag 1) | V=8192 (lag 1) | V=1024 (lag 64) | V=8192 (lag 64) |
|--------|---------------:|---------------:|----------------:|----------------:|
| Mean PMI (bits) | 2.93 | 4.60 | 0.67 | 3.10 |
| Std PMI | 2.22 | 3.08 | 1.11 | 3.10 |
| % Negative pairs | 7.5% | 3.4% | 26.1% | 12.4% |
| % Strong positive | 64.4% | 79.7% | 11.7% | 52.6% |
| Kurtosis | +0.57 | +0.02 | +3.21 | -0.20 |

Three patterns emerge. First, V=8192 has far fewer negative PMI pairs — token pairs that co-occur *less* than expected by chance. These are noise that the model must learn to ignore. At lag 64, V=1024 has 26% negative pairs (one in four token relationships is anti-correlated), while V=8192 has only 12%. Second, V=8192 has 4-5× more strong positive pairs at long range (52.6% vs 11.7% at lag 64), meaning the majority of distant token pairs carry meaningful signal. Third, V=8192 exhibits negative kurtosis (platykurtic distribution) while V=1024 has positive kurtosis (leptokurtic, heavy tails). Negative kurtosis means fewer extreme outliers and more weight in the moderate range — a more uniform, learnable distribution.

**Compression Analysis (compression_entropy_v2.py).** As an independent validation, block compression was used to measure how redundancy is exploited at different scales. Compressing the token stream in blocks of increasing size reveals how much structure exists at each context depth:

| Context Scale | V=1024 marginal (BPT) | V=8192 marginal (BPT) | V=1024 has more? |
|--------------:|----------------------:|----------------------:|:----------------:|
| Local (8 tokens) | 8.03 | 8.02 | Equal |
| Phrasal (32 tokens) | 2.25 | 2.18 | Equal |
| Sentence (128 tokens) | 0.91 | 0.44 | 2.1× more |
| Paragraph (512 tokens) | 0.43 | 0.35 | 1.2× more |
| Document (2048 tokens) | 0.22 | 0.18 | 1.2× more |

V=1024 has substantially more compressible redundancy at the sentence scale (0.91 vs 0.44 bits/token marginal). This redundancy represents patterns that *could* be captured by a model with enough context — but at V=8192, the vocabulary has already absorbed these patterns into the token definitions. The compression analysis independently confirms what the MI measurements show: larger vocabulary pre-compresses the text stream, leaving less work for the model.

**Sliding Window Analysis.** Compression ratio was also measured as a function of the compressor's context window (analogous to a model's sequence length):

| Window (tokens) | V=1024 ratio | V=8192 ratio | V=1024 Δ | V=8192 Δ |
|-----------------:|-------------:|-------------:|---------:|---------:|
| 256 | 0.621 | 0.721 | — | — |
| 1,024 | 0.566 | 0.681 | 0.055 | 0.040 |
| 4,096 | 0.535 | 0.666 | 0.031 | 0.015 |
| 16,384 | 0.497 | 0.652 | 0.038 | 0.014 |

V=8192's marginal improvement from longer windows diminishes faster — extending from 1K to 16K tokens provides only 0.029 ratio improvement for V=8192 vs 0.069 for V=1024. The larger vocabulary has already captured much of the long-range structure, reducing the model's dependence on long sequence context. This directly predicted the V×L experimental finding: V=8192 at seq_len=1024 performs nearly as well as V=8192 at seq_len=2048.

### 2. Vocabulary Sweep and Scaling Law Analysis

Comparing embedding proportions across 20 production models (GPT-2 through Qwen2.5-72B) revealed that the baseline's 2.7% embedding allocation is dramatically low for a model at comparable scale. Models under 1B parameters allocate 15–31% to embeddings; the rebalancing to 17.2% (V=8192, 512d) falls within this validated range.

### 3. Architecture Sweep on 8×H100

Five configurations were swept at approximately matched parameter count (18.5-19.9M) using baseline code and V=8192 to find the optimal width/depth balance:

| Config | BPB | ms/step | Steps |
|--------|----:|--------:|------:|
| 384d/14L (6 heads, hd=64) | **1.1923** | 55.9 | 10,736 |
| 384d/13L | 1.1953 | 52.3 | 11,481 |
| 320d/21L | 1.1977 | 69.9 | 8,579 |
| 320d/20L | 1.2006 | 66.6 | 9,006 |
| 336d/18L (7 heads, hd=48) | 1.2025 | 74.2 | 8,088 |

**Width wins over depth on H100.** Per-layer wall-clock cost scales sub-quadratically with dimension (384d is only 9% slower per layer than 320d despite 44% more FLOPs), while each additional layer adds unavoidable serial latency. The 384d/14L config runs at 55.9 ms/step vs 320d/20L at 66.6 ms/step — having fewer layers more than compensates for the wider matmuls.

### 4. Width Scaling and GQA

With the SOTA infrastructure producing artifacts well under 16 MB, progressively wider architectures were explored. The key breakthrough was adopting GQA (4 KV heads) at 512d, which saved enough parameters to match the SOTA's width while accommodating the larger vocabulary:

| Config | BPB | Artifact | Fits? | Steps | ms/step |
|--------|----:|--------:|:-----:|------:|--------:|
| 384d/14L MHA | 1.1399 | 13.04 MB | ✓ | 7,973 | 75.3 |
| 384d/16L MHA | 1.1319 | 14.49 MB | ✓ | 7,020 | 85.5 |
| 448d/13L MHA | 1.1244 | 15.84 MB | ✓ | 6,775 | 88.6 |
| **512d/11L GQA** | **1.1212** | **15.72 MB** | **✓** | **7,521** | **79.8** |

The 512d/11L GQA configuration achieves the best BPB while being the fastest per step and producing the smallest artifact. GQA saves ~2.9M KV parameters with negligible quality loss, enabling 512d width where MHA would have exceeded the 16 MB budget.

### 5. V×L Interaction Discovery

During optimization, it was discovered that vocabulary size and sequence length are partially substitutable dimensions of context. V=8192 at seq_len=1024 covers approximately the same text span (~3.3 KB) as V=1024 at seq_len=2048 (~3.5 KB).

Empirical validation on 448d/13L:
- s=2048, e=2048: **1.1244 BPB** (best for this config)
- s=1024, e=1024: 1.1288 BPB (only 0.004 worse, 5% faster training)
- s=1024, e=2048: 1.2418 BPB (broken — positional encoding mismatch)

The 0.004 BPB gap between matched s=1024 and s=2048 shows that the larger vocabulary's information density substantially compensates for shorter training sequences, though not completely. Compression analysis independently predicted this: V=8192's marginal gain from extending context diminishes faster than V=1024's, consistent with the vocabulary having already absorbed much of the long-range redundancy.

To the authors' knowledge, this V×L interaction has not been explicitly studied in the scaling laws literature (Kaplan 2020, Chinchilla 2022, Tao 2024), and represents a potential fifth scaling variable: L_opt(V, C) — optimal sequence length as a function of vocabulary size and compute budget.

## Why Per-Layer Techniques Don't Help at V=8192

One of the most striking findings of this work is that none of the per-layer techniques from the SOTA stack provide meaningful improvement when combined with V=8192. Training enhancements and evaluation techniques were systematically tested on the 512d/11L GQA architecture — all proven contributors on the V=1024 SOTA — and every one was neutral or negative:

| Technique | Config | BPB | Δ from base | ms/step | Steps | Verdict |
|-----------|--------|----:|----------:|--------:|------:|---------|
| **Base 512d/11L** | v2a, no extras | **1.1212** | **—** | **79.8** | **7,521** | **BEST** |
| + XSA (last 3) | v2a + XSA_LAST_N=3 | 1.1214 | +0.0002 | 81.0 | 7,410 | Neutral |
| + XSA + MTP(1) | v2a + XSA + MTP | 1.1241 | +0.0029 | 87.0 | 6,900 | Negative |
| + Turbo-Muon | v2c, AOL preconditioned NS | 1.1216 | +0.0004 | 79.7 | 7,533 | Neutral |
| + BigramHash + VE + XSA | v2b (per-layer feature stack), 448d/13L | 1.1212 | 0.0000 | 96.0 | 6,251 | Neutral* |
| + Warmdown 4000 | v2a + WD=4000, 448d/13L | 1.1245 | +0.0001† | 88.5 | 6,785 | Neutral |
| + Warmdown 2500 | v2a + WD=2500, 448d/13L | 1.1256 | +0.0012† | 88.5 | 6,781 | Negative |
| Eval: stride=16 | 4× more eval windows | 1.1216 | +0.0004 | — | — | Neutral |
| Eval: TTT (stride=64) | Score-first backward TTT | 1.1211 | -0.0001 | — | — | Neutral |
| Eval: TTT (stride=16) | TTT + fine-grained scoring | 1.1203 | -0.0009 | — | — | Marginal |
| Embed LR 0.06 | Higher LR for tied embedding | 1.1233 | +0.0021 | 79.8 | 7,521 | Negative |

*Per-layer features on 448d/13L matched v2a 512d/11L BPB but required a narrower architecture and didn't fit under 16 MB at 448d.
†Warmdown experiments run on 448d/13L; deltas shown relative to that config's base (1.1244 BPB).*

These same techniques contribute 0.003–0.010 BPB on the V=1024 SOTA stack. Why the difference?

The answer connects directly to the entropy analysis. XSA forces cross-token attention patterns, BigramHash captures adjacent-pair statistics, VE adds token-specific value biases, and MTP provides auxiliary gradient signal about future tokens. These features all solve the same underlying problem: **extracting more information from an information-sparse token stream**. At V=1024 (~2.4 bytes per token), each token carries limited information, and the model needs every trick to compose meaning from many small pieces.

At V=8192 (~3.7 bytes per token), the vocabulary has already absorbed the local patterns these techniques are designed to capture. The entropy measurements showed this directly: V=8192 has ~2.2× more baseline-corrected mutual information at long range (lags 64+), 2× fewer negative PMI pairs, and a flatter information spectrum with only 48% of MI concentrated in the adjacent band (vs 63% at V=1024). The signal quality is high enough that per-layer techniques are solving a problem that no longer exists.

This can be framed as a **signal quality hierarchy**: the vocabulary determines the "sea level" — the baseline information quality of the token stream — while per-layer techniques and evaluation tricks are "waves" that modulate on top. At V=1024 (low sea level), waves can reach 0.003-0.010 BPB above the surface. At V=8192 (high sea level), the same waves barely register (0.000-0.001 BPB) because the vocabulary has already raised the floor.

**Quantization as a ceiling on training improvements.** Investigation of the quantization pipeline revealed an additional constraint: the attention and MLP layers (82% of model parameters) use int6 quantization (±31, 64 levels) with GPTQ calibration, while the embedding is already at int8 (±127) via the original PR #1019 design. The int6 quantization introduces a noise floor of ~0.001-0.002 BPB per technique — any pre-quant improvement smaller than this floor gets destroyed during compression. This explains why Turbo-Muon (pre-quant -0.0008) vanishes post-quant, while the adapter (pre-quant -0.0018) barely survives. Full int8 for all layers achieved 1.1181 BPB (-0.003 from int6 base) but inflated the artifact to 20.5 MB. The quantization precision of the mlp/attn body thus acts as a ceiling on the value of any training or architectural optimization — a finding with implications for the broader competition, where many techniques may be bumping against the same int6 noise floor.

**The tied embedding learning rate experiment** probed the one remaining "sea level" variable: whether the embedding itself was learning optimally at V=8192. Each V=8192 token appears ~8× less frequently in training data than a V=1024 token, meaning each embedding row receives proportionally fewer gradient updates. Testing raised the embedding LR from 0.035 to 0.06 to compensate for this sparsity. The result was worse (+0.002 BPB) due to a fundamental conflict in tied embeddings: the same matrix serves as both input embedding (which benefits from higher LR to compensate for sparse updates) and output projection (which receives dense gradients through the softmax and destabilizes at higher LR). This conflict is well-documented in the literature (Press & Wolf, 2017) — larger models (GPT-3, LLaMA-3 70B) tend to untie embeddings precisely because the optimization conflict grows with vocabulary size.

This also explains the Turbo-Muon result. The optimizer overhead (Newton-Schulz iterations) is only 1-2% of step time at 512d matrices, so saving one iteration is invisible. At larger scales with wider matrices, the same optimization would be expected to yield meaningful speedup — but the model is too small for it to matter.

**Evaluation-time techniques show the same pattern.** Reducing the sliding window stride from 64 to 16 (giving each token 2032 vs 1984 tokens of context) produced identical BPB to 5 decimal places (1.12161 vs 1.12161). The MI decay analysis predicted this: mutual information is flat at lag 1984, so 48 additional context tokens add zero predictive value. Test-time training (TTT), which adapts the dequantized model on already-scored validation chunks, provided only -0.0001 BPB at stride=64 and -0.0009 at stride=16. At V=1024, TTT typically contributes 0.003-0.010 BPB — again, the larger vocabulary has already captured the local distribution that TTT would otherwise learn during evaluation.

### Per-Layer Feature Analysis

Having established that the combined per-layer features (BigramHash + VE + XSA) were net-neutral at V=8192, each feature was isolated on the 512d/11L GQA architecture to determine which, if any, provided independent value:

| Feature | BPB | Δ from base | ms/step | Steps | Artifact | Verdict |
|---------|----:|----------:|--------:|------:|--------:|---------|
| **Base 512d/11L** | **1.1212** | **—** | **79.8** | **7,521** | **15.72 MB** | — |
| + BigramHash only (4096×64) | 1.1199 | -0.0013 | 79.9 | 7,507 | 15.99 MB | **Best** |
| + BigramHash (8192×32) | 1.1204 | -0.0008 | 79.8 | 7,519 | 15.68 MB | Positive |
| + BigramHash + Trigram | 1.1213 | +0.0001 | 79.9 | 7,513 | 15.98 MB | Negative |
| + VE only (layers 9,10) | 1.1209 | -0.0003 | 80.3 | 7,478 | 15.95 MB | Noise |
| + XSA only (last 3) | 1.1214 | +0.0002 | 81.0 | 7,410 | 15.58 MB | Neutral |
| + All three combined | 1.1212 | 0.0000 | 96.0 | 6,251 | 16.41 MB | Neutral |

BigramHash is the only per-layer feature that provides a real improvement at V=8192 — and it does so with near-zero throughput cost (+0.16 ms/step). The combined result (neutral) makes sense in retrospect: BigramHash's -0.0013 gain was being cancelled by the throughput cost of VE and XSA, which consumed training steps without contributing quality.

The BigramHash variant sweep reveals a further insight: **embedding capacity per bucket matters more than collision reduction**. The 8192×32 configuration doubled the number of hash buckets (halving the collision rate from 16K to 8K bigrams per bucket) while halving the embedding dimension (64→32) to maintain the same parameter count. Despite fewer collisions, it performed worse than 4096×64 (-0.0008 vs -0.0013). The transition patterns need enough dimensions to be represented faithfully — 32 dimensions cannot capture the full structure of token transition statistics, even when each pattern has its own bucket. This suggests the information content of bigram relationships at V=8192 requires at least 64 dimensions to encode effectively.

Adding ungated trigram hashing erased the BigramHash gain entirely (+0.0001), consistent with findings from the V=1024 competition (PR #609: +0.0049 with ungated trigram). At V=8192, 8192³ ≈ 550 billion possible trigrams map into 4095 hash buckets — ~134 million collisions per bucket — making the signal-to-noise ratio catastrophically poor. Context-aware gating (as in EngramLite) would be needed to make trigrams viable, but that requires additional architectural complexity.

This hypothesis was tested directly with **EngramLite** — a gated multi-head variant (2 heads, 8192 buckets, gated bigram+trigram with sigmoid gate on the hidden state) adapted from the technique used in the competition's pending frontier (PR #1089). EngramLite achieved 1.1277 BPB — significantly worse than BigramHash's 1.1199, and worse even than the base model (1.1212). Two factors contributed: first, the larger embedding table (8192×64 vs 4096×64) forced 33% pruning of the base model's ±1 values to fit the artifact budget, damaging the transformer body to make room for the N-gram feature. Second, the pre-quant EMA was 1.1317, worse than BigramHash's 1.1299 — meaning the gated trigram provided negative value even before quantization. At V=8192's extreme collision rates, even learned gating cannot rescue trigrams. The gate would need to suppress >99.99% of trigram lookups to filter the noise, which is effectively learning to turn trigrams off entirely.

This result reveals a distinction between **complementary** and **compensatory** information in the feature stack. BigramHash captures *transition statistics* between tokens — which specific token tends to follow which specific token. This is genuinely new information that neither the vocabulary nor the transformer layers naturally encode: the vocabulary defines what each token *means*, and the transformer layers learn contextual patterns, but neither explicitly models the pairwise transition structure. BigramHash provides complementary information that adds to the model's knowledge.

VE and XSA, by contrast, provide *compensatory* information. VE reinjects token identity into attention values at deep layers — but at V=8192, the token representations are already information-rich, so there is less identity signal to recover. XSA forces cross-sequence attention patterns — but V=8192's dense tokens already facilitate rich inter-token attention naturally. These features compensate for information deficiencies in the token stream that exist at V=1024 but largely vanish at V=8192.

The complementary vs compensatory distinction explains why BigramHash alone succeeds where the full per-layer stack fails. It also suggests a design principle for future feature engineering at large vocabulary sizes: features that add *new* information channels (like transition statistics) will help, while features that *recover* information lost to tokenization will not.

The practical implication is nuanced: at V=8192, **almost** the simplest configuration is the best one. The one exception is BigramHash, which captures genuinely independent transition statistics at near-zero throughput cost. All other techniques — XSA, VE, MTP, Turbo-Muon, TTT, stride tuning — provide compensatory information that the vocabulary has already absorbed.

## Theoretical Framework: Vocabulary in the Small Model Regime

The experimental findings above suggest a general framework for understanding vocabulary sizing in resource-constrained language models. Six variables are identified that govern model quality at small scale:

- **V** (vocabulary, input resolution): how much information each token carries
- **L** (layers, processing depth): how many refinement stages the model applies
- **N** (network capacity): total parameters available for learned representations
- **C** (compute): training FLOPs, bounded by hardware and wall-clock time
- **Q** (quantization precision): how faithfully the trained model survives compression
- **A** (artifact budget): hard limit on stored model size

These variables interact across three distinct regimes.

**Regime 1: Large C, N.** When compute and capacity are abundant, V and L matter relatively little. The network has enough parameters and training steps to compose small tokens into rich representations regardless of vocabulary size. This is the regime of production LLMs, where vocabulary choices are driven by inference efficiency (fewer tokens = faster generation) rather than learning capacity. Most scaling law research (Kaplan et al., Hoffmann et al.) operates in this regime, which is why vocabulary has received little attention as a scaling variable.

**Regime 2: Small C, N — vocabulary becomes critical.** When compute and capacity are severely limited, each token must carry maximum information signal. A larger vocabulary pre-composes character-level patterns into single tokens, effectively front-loading part of the model's representational work into the tokenizer. The model's limited layers then start their refinement chain at a higher level of abstraction — learning phrase-level and discourse-level patterns rather than spending capacity on character composition. This is where entropy analysis of the training data becomes essential for sizing V to match the information structure of the domain.

The experiments support this from two angles. First, the architecture sweep showed width and depth are substitutable at V=8192: 512d/11L matches 448d/13L to within 0.003 BPB (1.1212 vs 1.1244), with the wider-shallower model winning on throughput. Second, the sequence length experiment showed V=8192 at seq_len=1024 matches V=8192 at seq_len=2048 to within 0.004 BPB (1.1288 vs 1.1244) — the vocabulary's information density compensates for shorter context. Both findings suggest vocabulary, depth, and context length are substitutable along a multi-dimensional optimization surface. But the substitution has diminishing returns: at V=8192, adding layers beyond 11 yields negligible improvement because the vocabulary has already absorbed the local structure that additional layers would learn.

**Regime 3: Small C, N with artifact constraint — the full optimization.** Parameter Golf operates in this regime, where A (artifact budget) introduces a hard constraint that interacts with every other variable. Larger V means a larger embedding table consuming more artifact bytes. Higher Q means less compressible weights. Additional features (BigramHash, adapters, EngramLite) add parameters that compete for the same fixed budget. The optimization becomes:

> *Maximize quality subject to: artifact(V, L, N, Q, features) ≤ A and training_time(V, L, N, C, features) ≤ T*

This constraint explains several findings from the ablation. The low-rank output adapter improved pre-quantization quality by 0.002 BPB but consumed 557 KB of artifact space in fp16, forcing aggressive pruning that destroyed the gain. Full int8 quantization reduced the quantization penalty from ~0.007 to ~0.001 BPB — the single largest improvement — but inflated the artifact from 15.7 MB to 20.5 MB due to higher entropy in the stored weights. The quality was there; the artifact constraint killed it.

**Q as the hidden bottleneck.** The quantization experiments revealed that Q is arguably the most impactful variable at small scale, yet it receives the least attention in the model design process. The quantization penalty at V=8192 (~0.007 BPB with int6) exceeds the gain from every architectural feature tested: BigramHash (-0.0013), adapter (-0.0016), EngramLite (-0.0012), Turbo-Muon (-0.0008). The trained model "knows" more than it can "remember" after compression. Investigation of the quantization pipeline revealed that the embedding layer is already quantized at int8 precision (±127, via the original PR #1019 codebase design), while the attention and MLP layers — 82% of model parameters — use int6 (±31) with GPTQ calibration. The ~0.007 BPB penalty comes primarily from these int6 layers, not the embedding. Full int8 for all layers reduced the penalty to ~0.001 BPB but inflated the artifact beyond 16 MB. Mixed-precision quantization (int7 for the mlp/attn body, keeping int8 for the embedding) is a natural compromise, targeting the actual bottleneck while managing compression cost.

**Why bolt-on features work at small V but not large V.** At V=1024, the model faces an information deficit: small tokens carry little per-token signal, and the limited network must compose characters into words into phrases. Features like XSA (cross-sequence attention), VE (value embeddings), and MTP (multi-token prediction) compensate for this deficit by recovering information lost during coarse tokenization. They act as extensions to the model's representational capacity, and their value is proportional to the information gap between what the vocabulary provides and what the network needs.

At V=8192, this gap largely closes. Each token carries ~53% more bytes of text on average, bigram patterns are absorbed into single tokens, and the remaining inter-token structure is richer and more learnable. The compensatory features have nothing left to compensate for. Only BigramHash survives because it provides genuinely *complementary* information — explicit pairwise transition statistics that neither the vocabulary nor the attention mechanism naturally encodes.

This distinction — compensatory vs complementary — predicts which features will transfer across vocabulary sizes. If a feature helps because it recovers lost signal (compensatory), it will lose value as V increases. If it helps because it adds an independent information channel (complementary), it will remain useful regardless of V. This is a design principle for feature engineering: at large vocabularies, invest in new information channels rather than signal recovery.

**Implications beyond competition.** Parameter Golf's constraints (time-limited training, multi-iteration experimentation, fixed 8×H100 hardware) create a game-theoretic incentive to chase small bolt-on improvements with small vocabularies. In production settings, where compute may be genuinely limited and models must be trained in fewer iterations, properly sizing the vocabulary to the information structure of the domain becomes more important — not less. The entropy analysis tools developed for this work (mutual information spectra, PMI analysis, surprisal variance decomposition) are directly applicable to production vocabulary design decisions. The core finding — that vocabulary is an underexplored scaling variable with outsized impact at small model scales — holds independent of the competition format.

## Warmdown Tuning

Three warmdown schedules were tested on the 448d/13L configuration:

| Warmdown | BPB | Artifact | % of Training |
|---------:|----:|--------:|--------------:|
| 2,500 | 1.1256 | 16.11 MB | 37% |
| 3,500 | 1.1244 | 15.84 MB | 52% |
| 4,000 | 1.1245 | 15.73 MB | 59% |

Shorter warmdown (2500) hurt both BPB and artifact compression. The 3500–4000 range is near-optimal for the step count (~6,800 steps). Longer warmdown produces more structured weight distributions that compress better under LZMA, a useful side effect for the 16 MB constraint.

## Quantization Precision Experiments

The most impactful finding emerged from quantization experiments rather than architectural changes. Investigation of the quantization pipeline revealed that the embedding layer (tok_emb.weight, 17% of params) is already quantized at int8 precision (±127) via the original PR #1019 design, while the attention and MLP layers (82% of params) use int6 (±31) with GPTQ calibration. The quantization penalty — the gap between pre-quantization model quality and post-quantization artifact quality — comes primarily from these int6 layers.

| Quantization | Sliding BPB | Pre-quant EMA | Quant Penalty† | Artifact | Fits 16 MB? |
|-------------|----------:|-------------:|-------------:|--------:|:-----------:|
| Int6 (±31) + BigramHash | 1.1199 | 1.1299 | 0.0067 | 15.97 MB | ✓ |
| Int7 (±63) + BigramHash | 1.1156 | 1.1300 | 0.0020 | 18.51 MB | ✗ |
| Int8 (±127) no BigramHash | 1.1181 | 1.1331 | 0.0014 | 20.47 MB | ✗ |

*†Quant Penalty = non-sliding roundtrip BPB minus DIAGNOSTIC pre-quant EMA BPB. Int6 row uses v9 (seed 1337) values; int7 and int8 rows from earlier code versions (v4, v3c).*

Int7 quantization achieved **1.1156 BPB** — the best single result in this work, 0.0009 behind the leaderboard SOTA (1.1147). The pre-quant EMA is comparable across all three configurations (1.1299–1.1331), confirming the improvement comes entirely from reduced quantization noise, not better training. Notably, int7 + BigramHash (1.1156) outperforms int8 without BigramHash (1.1181), demonstrating that BigramHash's complementary information channel is worth more than the extra precision from int7 to int8.

However, int7+LZMA compresses to 18.51 MB and int8+LZMA to 20.47 MB — both exceeding the 16 MB limit. The fundamental issue is entropy: more quantization levels give LZMA less redundancy to exploit. Even aggressive pruning of ±1 and ±2 values (3.25M candidates, 16% of all quantized weights) at int7 only reduced the artifact to 17.32 MB while destroying quality (+0.036 BPB penalty from zeroing 16% of weights).

**Bank QAT (quantization-aware training for bank weights).** This experiment injected STE quantization noise into the 82% of parameters stored in parameter banks during the last 7% of training steps. The hypothesis was that training with noise awareness would reduce the quantization penalty. Results: bank QAT at int6 was **neutral** — pre-quant quality was unchanged (1.1299 EMA) and the quant penalty was within noise of the baseline (0.0080 vs 0.0067, within the ±0.0009 range observed across identical-architecture runs with different seeds). The likely explanation is a QAT-GPTQ mismatch: the STE simulation uses simple per-row rounding, while GPTQ uses Hessian-aware error compensation that adjusts neighboring weights to minimize reconstruction error. The model adapted to noise pattern A during training, but faces noise pattern B at export.

This result reveals that **quantization precision, not model architecture, is the binding constraint** at V=8192. The trained model "knows" ~0.007 BPB more than the int6-quantized artifact can preserve. Upgrading to int7 recovers 70% of this penalty (0.0067 → 0.0020) purely from better weight precision, with zero architectural changes — but the 16 MB artifact budget prevents its use.

## Entropy Analysis Tools

Four analysis tools were developed that guided the vocabulary decision. These tools are vocabulary-agnostic and can be applied to any BPE-tokenized dataset. Two independent methodologies — mutual information estimation and compression-based complexity approximation — reach the same conclusion, providing cross-validation that the finding does not depend on any particular measurement method.

**entropy_analysis_v3_1.py** — Core MI measurement tool. Computes mutual information I(X_t; X_{t+k}) between tokens at configurable lag distances using the standard estimator: MI = Σ P(x,y) × log2(P(x,y) / (P(x) × P(y))), with joint and marginal probabilities estimated from frequency counts on the token stream. Implements shuffled-baseline correction: measures MI on randomly permuted tokens (3 shuffles, averaged) to establish the frequency-only MI floor, then subtracts this from raw MI to isolate genuinely sequential structure. Fits a power law MI(lag) = a × lag^(-b) via log-log linear regression with R² goodness-of-fit. Decomposes corrected MI into six scale bands (adjacent through document) to produce the information spectrum histogram. Also generates exploratory attention head allocation recommendations based on the MI distribution — these recommendations informed initial architecture exploration but were not used in the final model, which uses standard full attention with GQA.

**compression_entropy_v2.py** — Independent validation via Kolmogorov complexity approximation. Divides the token stream into non-overlapping blocks of increasing size (4 to 8192 tokens), compresses each block independently with zlib (level 9), and reports bits-per-token at each block size. The marginal improvement from doubling block size reveals how much exploitable structure exists at each context depth — a fundamentally different measurement from MI that reaches the same conclusion. Also performs sliding window analysis using zlib's wbits parameter (9–15, corresponding to 256–16384 token windows) to directly measure how much a model gains from longer context at each vocabulary size. This tool produced the V×L interaction finding: V=8192's marginal improvement from extended context diminishes faster than V=1024's.

**surprisal_variance.py** — PMI distribution analysis, going beyond mean MI to characterize the shape of token-pair dependencies. Computes pointwise mutual information PMI(x,y) = log2(P(x,y) / (P(x) × P(y))) for sampled token pairs at each lag distance, using joint probabilities estimated from the full token stream. Reports distribution statistics: standard deviation, skewness, kurtosis, percentage of negative (anti-correlated) pairs, percentage of strong positive pairs (PMI > 2 bits), and tail behavior (p99, p999). The shape metrics reveal learnability — V=8192's platykurtic (low-kurtosis) distribution with fewer negative pairs is more learnable than V=1024's leptokurtic (heavy-tailed) distribution with 2× more anti-correlated pairs.

**vocab_sweep_v2.py** — Orchestrates MI measurement across vocabulary sizes on identical underlying text. Decodes existing V=1024 FineWeb tokens back to raw text via SentencePiece, trains new BPE tokenizers at each target vocabulary size (512 through 16384), re-tokenizes the same text, and runs MI analysis at each size. Computes baseline-corrected band histograms and parameter budget implications for each vocabulary. This produced the central finding that vocabulary size acts as a filter on the information spectrum: larger vocabularies shift MI weight from adjacent to mid-range bands, with V=8192 showing ~2.2× more baseline-corrected MI at long range (lags 64+) compared to V=1024.

## Scaling Laws Reference

Embedding proportions were compiled across 20 production models with training sequence lengths, revealing the relationship between vocabulary size, model scale, and context length. Key finding: sequence length scales faster than vocabulary across model generations (GPT-2→LLaMA-3: V grew 2.6×, L grew 8×), suggesting vocabulary captures local patterns while sequence length captures the remaining long-range frontier.

## Reproduction

```bash
# Train the 8192 BPE tokenizer and re-tokenize FineWeb
python3 data/download_hf_docs_and_tokenize.py \
  --output-root ./data \
  --tokenizer-config ./data/tokenizer_specs_8192.json \
  --tokenizer-train-docs 5000000 \
  --skip-byte

# Train the model (8×H100, 10 minutes)
# All architecture params (512d, 11L, 8 heads, 4 KV heads, V=8192) are defaults
# Run all 3 seeds for validation:
for SEED in 1337 42 2026; do
  RUN_ID=vocab_v9_seed_${SEED} SEED=${SEED} \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

## Timeline

This work was conceived, developed, and validated in a compressed timeline:

| Phase | Timestamp | Activity |
|-------|-----------|----------|
| Concept | Mar 31, 16:00 UTC | Initial hypothesis: vocabulary as entropy filter on information spectrum |
| Analysis | Mar 31 | Entropy decomposition tools, vocabulary sweep (512–16,384), mutual information measurements across 56 lag distances on FineWeb |
| Scaling Laws | Apr 1 | Production model comparison (20 models), embedding proportion analysis, identified baseline's 2.7% as severe outlier |
| Architecture Sweep | Apr 1 | Five configurations on 8×H100 — width vs depth at matched params, 384d/14L wins |
| v2a Code Merge | Apr 1 | Integrated SOTA infrastructure from PR #1019 onto 8192-vocab architecture |
| Optimization | Apr 1 | Iterative runs: 14L→16L→17L, per-layer feature testing, 448d exploration, V×L experiments |
| Width Scaling | Apr 1 | v2a 448d/13L achieves 1.1244 BPB in 15.84 MB artifact |
| Warmdown Tuning | Apr 2 | LR schedule sweep (WD=2500/3500/4000), confirmed 3500-4000 optimal |
| GQA Discovery | Apr 2 | v2a 512d/11L GQA achieves 1.1212 BPB — new best, matching SOTA shape |
| Ablation Sweep | Apr 2 | XSA, MTP, Turbo-Muon, warmdown — all neutral or negative at V=8192 |
| Eval Techniques | Apr 2 | TTT integration, stride=16 testing — marginal improvements at best |
| Embed LR | Apr 2 | Tied embedding LR sweep — diagnosed input/output gradient conflict |
| Low-rank Adapter | Apr 2 | Output adapter to decouple tied embedding optimization |
| BigramHash | Apr 2 | v2b achieves 1.1199 BPB — development milestone |
| EngramLite | Apr 2 | Gated trigram testing — worse than BigramHash at V=8192 collision rates |
| Quant Discovery | Apr 2 | Discovered embedding always at int8, mlp/attn at int6 — corrected narrative |
| Int7 Experiment | Apr 2 | v4: int7 mlp/attn achieves 1.1156 BPB (best ever, 18.51 MB — doesn't fit) |
| Bank QAT | Apr 2 | v6/v7: STE noise for bank weights — neutral at int6, QAT-GPTQ mismatch |
| v8 Code Cleanup | Apr 2 | Removed dead code (MTP, XSA, VE, int8 pipeline), added shard preloading, coprime stride |
| v8 Validation | Apr 2 | v8 run revealed catastrophic GPTQ failure (val_loss 11.75 post-quant) and 1.2 nat training regression |
| Muon Cache Bug | Apr 2 | Root cause identified: NS5 warm-start cache introduced in v8 caused bf16 precision drift compounding over 7,500 steps. Steps 0-2 identical to v2b, divergence at step 3 growing to +1.21 nats by step 7500 |
| v9 Fix | Apr 2 | Removed NS5 cache — three lines. v9 NS5 signature matches v2b exactly |
| v9 Seed 1337 | Apr 3 | 1.1199 BPB, 15.97 MB, 7544 steps. DIAGNOSTIC val_bpb=1.1299, tokens_per_byte=0.268339 |
| v9 Seed 42 | Apr 3 | 1.1213 BPB, 15.97 MB, 7548 steps. DIAGNOSTIC val_bpb=1.1309 |
| v9 Seed 2026 | Apr 3 | 1.1208 BPB, 15.97 MB, 7544 steps. DIAGNOSTIC val_bpb=1.1313 |
| 3-seed validation | Apr 3 02:00 UTC | Mean: 1.1207 BPB, std: 0.0007, spread: 0.0014. All artifacts <16 MB |

**Total elapsed: ~84 hours from concept to 3-seed validated result.** Active 8×H100 GPU time: ~13 hours across approximately 43 experimental runs. Each run is 10 minutes of training plus ~5 minutes of GPTQ quantization and evaluation.

This compressed timeline means the result is not the global optimum of this approach — it's a proof of concept demonstrating that vocabulary rebalancing is a viable and underexplored dimension of the scaling law space. The v8→v9 debugging episode also yielded a cautionary finding: a Muon NS5 warm-start cache that appeared theoretically sound produced a 1.2 nat training regression due to bfloat16 precision drift compounding over thousands of steps — a reminder that optimizer modifications require multi-run validation even when the per-step effect is below measurement precision.

## Future Work

Several promising directions remain unexplored due to time constraints:

**Vocabulary size sweep at scale.** V=8192 was tested based on entropy analysis but did not run full training sweeps at V=2048, V=4096, or V=16384. V=4096 is particularly interesting: it would consume fewer embedding parameters, allowing MLP 3× (matching the SOTA's MLP capacity) while still benefiting from vocabulary rebalancing. The optimal vocabulary size depends on the compute-quality tradeoff — V=8192 may be slightly beyond the sweet spot for a 10-minute training window, where ~7,500 steps is the ceiling.

**GPTQ-calibrated embedding.** The embedding currently uses simple per-row int8 quantization (no Hessian calibration), while the mlp/attn layers get full GPTQ-calibrated int6. Adding Hessian-aware rounding to the embedding's output projection (the tied-weight path `F.linear(x, tok_emb.weight)`) would be a free quality improvement with no artifact size change — but requires custom Hessian collection for the embedding layer.

**Untied embeddings at smaller vocabulary.** At V=4096, untying embeddings would cost ~2.1M extra params (~1.3 MB) — potentially affordable if the MLP is sized appropriately. This would eliminate the tied embedding LR conflict entirely and could unlock higher embedding learning rates for the input side.

**V×L joint optimization.** The s=1024 experiments showed substantial substitutability between vocabulary and sequence length (only 0.004 BPB gap at matched evaluation) — a finding with, to the authors' knowledge, no precedent in the scaling laws literature. A systematic sweep across (V, L) pairs at fixed compute would map the full substitutability surface and could yield the scaling law L_opt(V, C).

**Formalizing the V, L, N, C, Q, A framework.** The theoretical framework presented above identifies six interacting variables and three operating regimes. A formal treatment could extend the Kaplan/Chinchilla/Tao scaling laws with vocabulary and quantization precision as coupled variables, deriving optimal V(N, A) and Q(V, A) relationships. The power law MI decay measurements (I ∝ lag^(-0.62)) and V×L substitutability surface provide empirical anchors for such a theory.

**Multi-resolution attention head allocation.** The entropy_analysis_v3_1.py tool produces per-band MI percentages that map directly to attention head allocation recommendations — for example, dedicating more heads to local context (where MI is concentrated) and fewer to document-scale context (where MI is sparse but high-variance). The V=8192 MI histogram suggests allocating ~4 heads to adjacent/local bands and ~4 to phrasal/sentence bands, with the sentence band receiving disproportionate weight relative to its V=1024 allocation. Standard full attention was used in this submission, but the MI-guided allocation could be implemented via mixed window sizes per head or sliding-window attention with variable spans. The surprisal_variance.py analysis adds a refinement: at long range, the PMI distribution has heavy tails (high p99/p999) despite low mean MI, suggesting that even bands with small average MI may justify at least one global attention head to capture rare but highly informative long-range dependencies.

## Three-Seed Validation

| Seed | Sliding BPB | DIAGNOSTIC BPB | Roundtrip BPB | Steps | Artifact |
|-----:|----------:|-------------:|-------------:|------:|--------:|
| 1337 | 1.1199 | 1.1299 | 1.1366 | 7,544 | 15.97 MB |
| 42 | 1.1213 | 1.1309 | 1.1383 | 7,548 | 15.97 MB |
| 2026 | 1.1208 | 1.1313 | 1.1376 | 7,544 | 15.97 MB |
| **Mean** | **1.1207** | **1.1307** | **1.1375** | **7,545** | **15.97 MB** |
| **Std** | **0.0007** | **0.0007** | **0.0009** | **2** | **3.6 KB** |

All three runs completed within 600s on 8×H100, all artifacts fit under 16 MB, and `tokens_per_byte=0.268339` is identical across all seeds (confirming it is a tokenizer property, not model-dependent). The inter-seed spread of 0.0014 BPB and std of 0.0007 are consistent with the competition norm (std ≈ 0.0005 BPB for comparable V=1024 entries).

## Acknowledgments

This work builds on the extraordinary collaborative effort of the Parameter Golf community. The training infrastructure — Parallel Muon optimizer, GPTQ int6 quantization, sliding window evaluation, SmearGate, U-Net skip connections, EMA/SWA, and many other innovations — represents contributions from dozens of researchers. This work adapted the infrastructure from [PR #1019](https://github.com/KellerJordan/modded-nanogpt/pull/1019) by @raahilshah, which consolidated the community's best techniques.

The contribution of this work is the vocabulary rebalancing approach: demonstrating through entropy analysis that vocabulary size is an underexplored scaling variable, and that rebalancing from V=1024 to V=8192 with a matching architecture achieves competitive performance (1.1207 BPB, 3-seed mean, within 0.0060 of the leaderboard SOTA) from a fundamentally different parameter allocation — trading MLP capacity for embedding capacity while keeping the same architectural shape. Systematic ablation of 20+ techniques established that vocabulary quality makes most per-layer features redundant, with BigramHash being the sole exception due to its complementary (not compensatory) information channel. Quantization experiments (int6/int7/int8) further revealed that the remaining gap is primarily a quantization precision constraint: at int7 precision the same architecture achieves 1.1156 BPB, within 0.0009 of SOTA, but cannot fit in the 16 MB artifact budget.

## Appendix: Production Model Comparison

Embedding proportions, vocabulary sizes, and training sequence lengths across 20 production language models, sorted by embedding percentage. This table informed the vocabulary rebalancing decision by showing that the baseline's 2.7% embedding allocation is an outlier at sub-billion parameter scales.

| Model | Embed % | Total Params | Vocab | Dim | Layers | Seq Len | ~Eff. Context | Tied | Source |
|-------|--------:|-------------:|------:|----:|-------:|--------:|--------------:|:----:|--------|
| GPT-2 Small | 31.1% | 124M | 50,257 | 768 | 12 | 1,024 | ~4.5 KB | Yes | Radford et al. 2019 |
| Qwen2.5-0.5B | 27.6% | 494M | 151,936 | 896 | 24 | 32,768 | ~180 KB | Yes | Alibaba/Qwen 2024 |
| **PG Vocab 8192** | **17.2%** | **24.4M** | **8,192** | **512** | **11** | **2,048** | **~6.5 KB** | **Yes** | **This work** |
| Qwen2.5-1.5B | 15.1% | 1.54B | 151,936 | 1,536 | 28 | 32,768 | ~180 KB | Yes | Alibaba/Qwen 2024 |
| GPT-2 Medium | 14.5% | 355M | 50,257 | 1,024 | 24 | 1,024 | ~4.5 KB | Yes | Radford et al. 2019 |
| Qwen2.5-7B | 14.3% | 7.62B | 151,936 | 3,584 | 28 | 32,768 | ~180 KB | No | Alibaba/Qwen 2024 |
| LLaMA-3 8B | 13.1% | 8.03B | 128,256 | 4,096 | 32 | 8,192 | ~45 KB | No | Grattafiori et al. 2024 |
| Qwen2.5-14B | 10.5% | 14.8B | 152,064 | 5,120 | 48 | 32,768 | ~180 KB | No | Alibaba/Qwen 2024 |
| Qwen2.5-3B | 10.1% | 3.09B | 151,936 | 2,048 | 36 | 32,768 | ~180 KB | Yes | Alibaba/Qwen 2024 |
| GPT-2 Large | 8.3% | 774M | 50,257 | 1,280 | 36 | 1,024 | ~4.5 KB | Yes | Radford et al. 2019 |
| GPT-2 XL | 5.2% | 1.56B | 50,257 | 1,600 | 48 | 1,024 | ~4.5 KB | Yes | Radford et al. 2019 |
| LLaMA-2 7B | 3.9% | 6.74B | 32,000 | 4,096 | 32 | 4,096 | ~18 KB | No | Touvron et al. 2023 |
| Mistral 7B | 3.6% | 7.24B | 32,000 | 4,096 | 32 | 32,768 | ~143 KB | No | Jiang et al. 2023 |
| Qwen2.5-72B | 3.4% | 72.7B | 152,064 | 8,192 | 80 | 32,768 | ~180 KB | No | Alibaba/Qwen 2024 |
| LLaMA-3 70B | 3.0% | 70.6B | 128,256 | 8,192 | 80 | 8,192 | ~45 KB | No | Grattafiori et al. 2024 |
| **PG Baseline** | **2.7%** | **19.4M** | **1,024** | **512** | **9** | **1,024** | **~1.7 KB** | **Yes** | **Parameter Golf** |
| LLaMA-1 7B | 1.9% | 6.74B | 32,000 | 4,096 | 32 | 2,048 | ~9 KB | Yes | Touvron et al. 2023 |
| LLaMA-1 13B | 1.3% | 13.0B | 32,000 | 5,120 | 40 | 2,048 | ~9 KB | Yes | Touvron et al. 2023 |
| GPT-3 175B | 0.7% | 175B | 50,257 | 12,288 | 96 | 2,048 | ~9 KB | No | Brown et al. 2020 |
| Chinchilla 70B | 0.7% | 70.0B | 32,000 | 8,192 | 80 | 2,048 | ~9 KB | No | Hoffmann et al. 2022 |
| LLaMA-2 70B | 0.7% | 70.0B | 32,000 | 8,192 | 80 | 4,096 | ~18 KB | No | Touvron et al. 2023 |
| LLaMA-1 65B | 0.4% | 65.2B | 32,000 | 8,192 | 80 | 2,048 | ~9 KB | Yes | Touvron et al. 2023 |

*Effective context is estimated as seq_len × avg_bytes_per_token, where bytes/token varies by vocabulary size and tokenizer training data. Embedding % for models with untied embeddings includes both input and output embedding layers.*

Key observations:
- Embedding proportion scales inversely with model size: 31% at 124M params (GPT-2 Small) to 0.4% at 65B (LLaMA-1 65B)
- At the sub-1B scale relevant to Parameter Golf, all production models allocate 15–31% to embeddings — the baseline's 2.7% is 5-11× lower
- This submission at 17.2% falls within the validated range for small-scale models
- Sequence length has grown faster than vocabulary across model generations (GPT-2→LLaMA-3: V grew 2.6×, L grew 8×), consistent with vocabulary capturing local patterns while sequence length captures the long-range frontier

## Appendix: MI Decay Curve and Power Law Fit

The corrected mutual information (baseline-subtracted) for V=8192 follows a power law decay across 60 lag points measured on 20M tokens of FineWeb data:

**Fit: I(lag) = 0.563 × lag^(-0.619), R² = 0.909**

| Lag | Raw MI (bits) | Corrected MI (bits) | Power law prediction | Error |
|----:|-------------:|--------------------:|--------------------:|------:|
| 1 | 3.340 | 2.594 | 0.563 | 2.031 |
| 2 | 1.701 | 0.954 | 0.366 | 0.588 |
| 4 | 1.013 | 0.266 | 0.238 | 0.028 |
| 8 | 0.865 | 0.118 | 0.155 | -0.036 |
| 16 | 0.825 | 0.079 | 0.101 | -0.022 |
| 32 | 0.803 | 0.057 | 0.066 | -0.009 |
| 64 | 0.789 | 0.042 | 0.043 | 0.000 |
| 128 | 0.777 | 0.031 | 0.028 | 0.003 |
| 256 | 0.767 | 0.021 | 0.018 | 0.003 |
| 512 | 0.759 | 0.013 | 0.012 | 0.001 |

The power law fits well from lag 4 onward (error < 0.03). Lags 1-2 exceed the power law prediction because adjacent tokens carry disproportionate MI from sub-word co-occurrence patterns that the power law (which models longer-range semantic structure) does not capture. The baseline MI (0.746 bits, the asymptotic floor) represents vocabulary-independent statistical regularities in the token stream.

The exponent -0.619 characterizes the hierarchical information structure of English text at V=8192 tokenization. For comparison, natural language MI decay exponents in the literature range from -0.5 to -0.8 depending on tokenization and genre, consistent with this measurement.
