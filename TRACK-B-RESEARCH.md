# Parameter Golf Track B Research: N-gram Memorization + Hybrid Approaches

**Date:** March 26, 2026  
**Researcher:** Prometheus (DeepSeek V3.2)  
**Purpose:** Strategic pivot analysis for Parameter Golf competition

## Executive Summary

The Parameter Golf competition has experienced a seismic shift with the emergence of "Track B" approaches that combine neural networks with n-gram lookup tables. These hybrid techniques achieve **0.3-0.9 bits per byte (BPB)** compared to **1.1 BPB** for pure neural models—a 3-4x improvement in compression efficiency.

The breakthrough is **eval-time n-gram backoff caching**, which memorizes frequent token sequences from the training data and uses them during evaluation. This is legal under competition rules as long as the n-gram tables fit within the 16MB artifact limit alongside the model weights and code.

## Competition Rules Context

Key constraints from OpenAI's Parameter Golf:
- **16,000,000 byte artifact limit** (decimal MB, not MiB)
- **No external data, network calls, or training dataset access during evaluation**
- **10-minute training budget** on 8×H100 SXM
- **10-minute evaluation budget**
- **Tokenizer-agnostic BPB metric** (bits per byte)

**Critical interpretation:** N-gram tables are allowed IF they're stored in the artifact and count toward the 16MB limit. The tables effectively "pay for bits" of memorized training data within the artifact budget.

## PR Analysis

### 1. PR #796 (0.4374 BPB) — "Distributed Prefill + Order-Adaptive 15-gram + EBLS"
**Author:** Robby955  
**Key Innovations:**
- **Distributed cache pre-fill using pure numpy**: Each GPU rank pre-populates n-gram hash tables with ALL preceding token positions before scoring
- **Order-adaptive 15-gram**: Dynamically adjusts n-gram order based on entropy
- **EBLS (Empirical Bayes Layer Sharing)**: Cross-layer weight sharing with learned gating
- **Memory split**: ~13MB for neural model, ~3MB for n-gram tables
- **Legal status**: Compliant—tables stored in artifact, no external data access

**Technical Details:**
- Mathematically identical to single-GPU sequential evaluation
- 8×H100 SXM, 560s training + ~300s eval
- 3-seed validated with std 0.0003
- Distributed prefill accounts for -0.31 BPB improvement
- Order-adaptive entropy gating accounts for -0.18 BPB

### 2. PR #809 (0.295 BPB) — "Chunk-Based N-gram Backoff + Score-First TTT"
**Author:** AayushBaniya2006  
**Current SOTA (0.29519 BPB)**

**Architecture:**
- 11 layers, 512 dimension, GQA 8/4, MLP 3.0x, XSA-4, LeakyReLU(0.9)²
- BigramHash(4096), GPTQ int5 quantization
- **Order-9 chunk-based N-gram eval cache** with entropy-adaptive alpha and per-order multipliers
- **Score-first TTT (LoRA)**: Test-time training that adapts only based on already-scored tokens
- **Artifact size**: 13.4MB (model + tables)
- **Training**: 525s + 340s eval on 8×H100 SXM

**N-gram Implementation:**
- Chunk-based processing for memory efficiency
- Backoff mechanism: try 9-gram → 8-gram → ... → unigram
- Entropy-adaptive alpha weights n-gram predictions vs neural predictions
- Per-order multipliers optimize contribution of each n-gram order

**Score-First TTT vs Regular TTT:**
- **Regular TTT**: Adapts model weights during evaluation using gradient descent
- **Score-First TTT**: Only adapts based on tokens that have already been scored (backward-looking)
- **Legal distinction**: Score-first TTT doesn't "peek" at future tokens, complying with "no adaptation on unevaluated data" rule

### 3. PR #811 (0.4377 BPB) — "Complementary Training + Backoff N-gram Mixer"
**Author:** quietsmile  
**Key Concept: Complementary Training**

**Complementary Training Explained:**
- Neural model is trained to specialize on tokens that n-gram caches **can't predict well**
- **Bigram-weighted loss reweighting** (COMPLEMENT_ALPHA=0.5)
- During training, downweight loss on tokens that are easily predictable by n-grams
- Forces neural network to focus on "hard" patterns that require deeper understanding
- Results in better synergy between neural and n-gram components

**Architecture:**
- Reproduction of PR #803's approach on 8× L20Z (H100 equivalent)
- BackoffNgramMixer (orders 2-10)
- Legal score-first AdamW TTT
- 2-seed validation: 0.4377 (seed=1337), 0.4380 (seed=42)

### 4. PR #769 (0.8495 BPB) — "PROTEUS+STYX — LeakyReLU(0.9)² + 5-gram Eval Cache"
**Author:** MatoTeziTanka  
**Early n-gram implementation (pre-breakthrough)**

**Key Features:**
- 5-gram eval cache (simpler than later approaches)
- LeakyReLU(0.9)² activation (squared leaky ReLU)
- Fixed torch.compile double-invocation bug that killed sliding window eval
- 3-seed mean: 0.8495 BPB (std 0.0013)
- All artifacts under 16,000,000 bytes

**Significance:** Demonstrated that even simple n-gram caches (~5-gram) could achieve 0.85 BPB, beating pure neural SOTA (1.1194) by 0.27 BPB.

### 5. PR #813 (0.6671 BPB) — "BackoffNgramMixer"
**Author:** hypery11  
**Reference implementation for n-gram backoff technique**

**Contribution:** Provides clean, reusable implementation of the n-gram backoff cache that other PRs reference. The breakthrough technique is **purely eval-time**—adding it to an existing best checkpoint should immediately jump from ~1.119 to ~0.67 BPB.

### 6. PR #812 — "BankLinear: cross-layer shared weight bank"
**Author:** andrewmouldon  
**Different approach: Parameter efficiency, not n-grams**

**Technique:**
- Per-layer weights constructed as mixtures over a shared bank of learned and fixed random matrices
- Replaces explicit per-layer weight storage with compositional weight synthesis
- Enables parameter reuse across depth while maintaining per-layer specialization
- **Not currently competitive** on time-constrained leaderboard (~1.25× slower training)
- **Results**: Average 1.2297 BPB vs baseline 1.2331 BPB
- **Artifact savings**: ~1.2MB (15.73MB vs 15.86MB baseline)

**Significance:** Shows alternative path to parameter efficiency, but n-gram approaches currently dominate.

## N-gram Language Models: Technical Foundation

### Kneser-Ney Smoothing
- **Problem**: Standard n-gram models fail on unseen sequences
- **Solution**: "Smooth" probability mass toward unknown n-grams
- **Kneser-Ney**: Most effective smoothing method for n-gram LMs
- **Interpolated modified Kneser-Ney**: State-of-the-art for pure n-gram models

### Backoff vs Interpolation
- **Backoff models** (Katz, stupid backoff): Try higher-order n-gram first, fall back to lower order if not found
- **Interpolation models**: Always combine probabilities from all orders
- **Competition approach**: Uses backoff (try 15-gram → 14-gram → ... → unigram)

### Memory Efficiency of N-gram Tables
**Storage requirements:**
- Token IDs: 2 bytes each (65,536 vocabulary)
- N-gram of length L: 2L bytes for tokens + overhead
- Frequency counts: 1-4 bytes depending on compression
- Hash table overhead: ~30-50%

**Theoretical capacity in ~10MB:**
- Assume 50% overhead for hash tables
- ~6.7MB for actual n-gram data
- At 4 bytes per n-gram entry (2 bytes token + 2 bytes frequency/pointer)
- **~1.7 million n-gram entries**
- With average length 7: **~12 million tokens memorized**

**FineWeb context:** 10B tokens in dataset. 12M tokens = 0.12% coverage. But n-grams capture **frequent patterns** disproportionately—the Pareto principle applies.

## Memory Budget Analysis

**Typical split in top submissions:**
- **Neural model**: 11-13MB (weights + code)
- **N-gram tables**: 3-5MB
- **Total**: 14-16MB (at limit)

**Optimization tradeoffs:**
1. **More n-gram memory** → better memorization but smaller neural model
2. **Higher n-gram orders** → better prediction but exponential memory growth
3. **Backoff depth** → fallback chain length affects accuracy vs memory

**Current optimal point:** ~4MB for n-grams, orders 2-15, with intelligent backoff.

## Legal Compliance Analysis

### Why These Approaches Are Likely Legal
1. **No external data**: N-gram tables are built from training data and stored in artifact
2. **No network calls**: Everything is local
3. **Fits in artifact**: Counts toward 16MB limit
4. **Eval-time only**: No adaptation on unevaluated data (score-first TTT compliant)
5. **Backward-looking**: Only uses already-scored tokens for adaptation

### Potential Concerns
1. **Memorization vs Learning**: Argument that n-grams are "cheating" by memorizing rather than learning
2. **Spirit of competition**: Whether OpenAI intended models to memorize training data
3. **Verification complexity**: Harder to verify no data leakage

**Community consensus (Issue #140):** These approaches are considered a valid "eval-time augmentation track" (0.97-1.10 BPB range mentioned). The competition has effectively split into:
- **Track A**: Pure neural models (~1.1 BPB)
- **Track B**: Neural + n-gram hybrids (~0.3-0.9 BPB)

## Theoretical Limits

### N-gram Compression vs Neural Compression
**N-gram advantage:**
- Perfect memorization of frequent patterns
- No generalization error for memorized sequences
- Extremely efficient storage for high-frequency n-grams

**Neural advantage:**
- Generalization to unseen patterns
- Context understanding beyond local window
- Parameter efficiency for complex patterns

**Hybrid theoretical limit:** Close to entropy of English text (~1.0 bits per character = ~0.125 BPB for UTF-8). Current 0.295 BPB is ~2.4× above this limit.

### Shannon's Source Coding Theorem
- Lower bound: Entropy of source
- English entropy estimates: 0.6-1.3 bits per character
- UTF-8: 1 byte per character → 0.075-0.163 BPB theoretical minimum
- **0.295 BPB is 1.8-3.9× above theoretical minimum**

## Combination with Transformer Models (Hybrid Track A + B)

**All current approaches ARE hybrids:** They combine:
1. **Transformer backbone** (11L, 512d, etc.)
2. **N-gram lookup tables** (orders 2-15)
3. **Gating mechanism** to blend predictions

**Architecture patterns:**
1. **Parallel prediction**: Neural and n-gram make independent predictions, weighted sum
2. **Residual prediction**: N-gram predicts, neural predicts correction
3. **Conditional gating**: Use entropy to decide when to trust n-gram vs neural

**Score-First TTT integration:** Adapt neural weights based on n-gram prediction errors, creating a feedback loop.

## Strategic Implications

### Should We Pivot to Track B?

**Arguments FOR pivot:**
1. **Massive BPB improvement**: 3-4× better than pure neural
2. **Proven techniques**: Multiple implementations with reproducible results
3. **Legal (likely)**: Community consensus accepts as valid track
4. **Time efficiency**: Eval-time addition to existing checkpoints
5. **Competitive necessity**: To compete on leaderboard

**Arguments AGAINST pivot:**
1. **Rule uncertainty**: OpenAI could retroactively reject
2. **Different skill set**: Requires n-gram LM expertise
3. **Diminishing returns**: Frontier already at 0.295 BPB
4. **Pure neural track** might have separate recognition

**Recommendation:** **YES, pivot immediately.** The BPB gap is too large to ignore. Even if OpenAI creates separate tracks, being competitive in the n-gram track is essential.

### Implementation Priority
Based on PR #809 analysis:

1. **Immediate**: Add BackoffNgramMixer to existing best checkpoint (expected: 1.119 → ~0.67 BPB)
2. **Short-term**: Implement complementary training (neural specializes on hard tokens)
3. **Medium-term**: Optimize n-gram table compression and memory layout
4. **Long-term**: Explore novel neural-n-gram integration architectures

### Risk Mitigation
1. **Keep pure neural baseline** as fallback
2. **Document compliance** thoroughly
3. **Engage with community** on legality discussions
4. **Prepare for rule clarifications** from OpenAI

## Technical Implementation Guide

### Core Components to Implement

1. **NgramEvalCache Class** (from PR #813):
   - Hash table implementation
   - Backoff logic (try order N → N-1 → ... → 1)
   - Memory-efficient storage

2. **Entropy-Adaptive Alpha**:
   - Dynamic weighting of n-gram vs neural predictions
   - Based on prediction confidence/entropy

3. **Complementary Training**:
   - Bigram-weighted loss reweighting
   - Force neural to focus on hard patterns

4. **Score-First TTT**:
   - AdamW with cosine learning rate
   - Only adapt on already-scored tokens
   - LoRA-style low-rank adaptation

### Memory Optimization Techniques
1. **Token ID compression**: Collapse equivalent IDs (-23% vocab)
2. **Multi-head hashing**: K=4 heads per n-gram order to reduce collisions
3. **Context-aware gating**: Sigmoid gate suppresses noisy lookups
4. **Selective storage**: Only store n-grams above frequency threshold

## Conclusion

The Parameter Golf competition has fundamentally changed with the n-gram memorization breakthrough. What seemed like a pure neural architecture challenge has become a hybrid compression problem.

**Key insights:**
1. **Eval-time n-gram backoff** is the dominant technique
2. **0.295 BPB is achievable** with current methods
3. **Approaches are likely legal** under current rules
4. **Hybrid neural+n-gram** is the optimal architecture
5. **Implementation is straightforward** as eval-time addition

**Strategic recommendation:** Immediately pivot to Track B approach. The BPB improvement is too significant to ignore, and the techniques are well-documented in the PRs. Start by adding BackoffNgramMixer to our best checkpoint, then iterate with complementary training and optimization.

The frontier will likely push below 0.2 BPB soon. Being competitive requires embracing this new meta.

---

## References

1. PR #796: https://github.com/openai/parameter-golf/pull/796
2. PR #809: https://github.com/openai/parameter-golf/pull/809  
3. PR #811: https://github.com/openai/parameter-golf/pull/811
4. PR #769: https://github.com/openai/parameter-golf/pull/769
5. PR #813: https://github.com/openai/parameter-golf/pull/813
6. PR #812: https://github.com/openai/parameter-golf/pull/812
7. Issue #140 (Community discussion): https://github.com/openai/parameter-golf/issues/140
8. Competition rules: https://github.com/openai/parameter-golf

## Appendix: N-gram Compression Math

### Storage Efficiency Calculation
```
Vocabulary size: 65,536 tokens (2 bytes each)
N-gram entry: tokens (2L bytes) + frequency (2 bytes) = 2L + 2 bytes
With 50% hash overhead: (2L + 2) × 1.5 bytes
10MB budget: 10,000,000 bytes
Maximum entries: 10,000,000 / [(2L + 2) × 1.5]

For L=7 (7-gram): 10M / (16 × 1.5) ≈ 416,667 entries
Tokens memorized: 416,667 × 7 ≈ 2.9M tokens
```

### BPB Improvement Estimation
```
Pure neural baseline: 1.119 BPB
Add 5-gram cache: ~0.85 BPB (-0.269)
Add 15-gram with backoff: ~0.67 BPB (-0.449)
Add complementary training: ~0.44 BPB (-0.679)
Add score-first TTT: ~0.30 BPB (-0.819)
```

Each technique compounds, but with diminishing returns.