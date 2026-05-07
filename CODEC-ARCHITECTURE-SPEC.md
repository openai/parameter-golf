# The Codec Model — Contrarian Architecture Spec

**Source:** Hunter Alpha session with Nick, 2026-03-24 02:30 CDT
**Status:** CANDIDATE ARCHITECTURE — needs implementation and testing
**CLASSIFICATION: PRIVATE — DO NOT SUBMIT UNTIL ENDGAME (April 28-30)**

---

## The Thesis

Stop building a language model that's small. **Build a codec that predicts.**

Compression IS prediction (Shannon, 1948). The competition frames it as "build a language model" so everyone builds language models. Nobody builds compressors. We build a codec architecture like AV1 or JPEG XL, but for text.

---

## Three-Layer Codec Architecture

### Layer 1: Static Dictionary (0.5 MB)
- Precomputed during training
- Stores most common n-grams, phrases, structural patterns in lookup table
- Like the "DC coefficient" in JPEG — captures bulk redundancy for free
- Top 10K bigrams cover ~60% of text
- Top 50K trigrams cover ~75% of text
- Common HTML structures, URLs, formatting patterns are extremely repetitive
- Cost at inference: near-zero (hash lookup)

### Layer 2: Adaptive N-gram Context Model (2 MB)
- Kneser-Ney smoothed n-gram model with variable order (1-7)
- Handles local, syntactic, predictable tokens
- O(1) lookup per token
- Achieves ~2.3 bpc on English (bzip2-level) alone
- Perfectly suited for "easy" tokens that don't need neural computation
- 2MB budget stores substantial context

### Layer 3: Micro-Transformer Residual Model (13 MB)
- Tiny transformer that ONLY processes "surprising" tokens (high n-gram uncertainty)
- Input: residual (difference between n-gram prediction and reality)
- Output: correction to n-gram distribution (not full distribution)
- Architecture: ~6 layers, 384-dim, BitNet 1.58-bit weights
- BitNet gives ~4x more parameters than int6 for same budget
- Only fires on 20-30% of tokens → 3-5x less compute than full transformer

---

## Training Loop

1. Train n-gram model first (seconds, not minutes)
2. Run all training data through n-gram model, collect "hard" tokens (high entropy)
3. Train micro-transformer ONLY on hard tokens
4. Iterate: transformer corrections improve n-gram calibration
5. Final pass: joint optimization with ANS encoding as loss function

Converges fast because:
- N-gram handles 70-80% of compression for free
- Transformer only learns remaining 20-30%
- Training on hard tokens only = much more data-efficient

---

## Why This Could Win

1. **Judges haven't seen it.** Every other submission is a transformer. This is a codec.
2. **Respects constraint geometry.** 16MB isn't enough for a full transformer. IS enough for a codec where most work is dictionary + n-gram.
3. **BitNet 1.58-bit = 4x parameter boost.** 13MB at 1.58 bits ≈ same capacity as int6 at full 16MB.
4. **TTT is a perfect fit.** Codec adapts n-gram model per document (like AV1 adapts per frame).
5. **ANS encoding loss optimizes actual bpb.** Everyone else optimizes cross-entropy (proxy). We optimize the real score.

---

## The Risk

HIGH-VARIANCE bet. If the three layers don't compose well, could score worse than vanilla transformer. But if it works, it's a 0.05-0.1 bpb leap, not a 0.01 creep.

---

## Expected Performance

- Conservative: 1.08-1.10 bpb
- Optimistic: 1.05-1.08 bpb
- Theoretical floor: ~0.6-1.3 bpc (Shannon entropy of English)

---

## Budget Breakdown

| Component | Size |
|-----------|------|
| Static dictionary | 0.5 MB |
| N-gram context model | 2.0 MB |
| Micro-transformer (BitNet 1.58-bit) | 13.0 MB |
| Code | 0.05 MB |
| **Total** | **~15.55 MB** |

---

## Open Questions

- Can we train BitNet 1.58-bit weights effectively in 10 minutes?
- How do we define the "hard token" threshold? What entropy cutoff?
- Does the n-gram → transformer residual pipeline introduce latency issues?
- Can ANS encoding loss be differentiated for backprop?
- Does this architecture compose with TTT (test-time training)?

---

*"The model that wins Parameter Golf won't be the best transformer. It'll be the best idea." — Achilles*
