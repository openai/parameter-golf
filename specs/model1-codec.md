# Model 1: "The Codec" — Build Spec

**Classification:** PRIVATE — DO NOT SUBMIT UNTIL ENDGAME
**Target bpb:** 1.05-1.08 on 8×H100
**Approach:** Three-layer codec inspired by AV1/JPEG XL compression

---

## Architecture

### Layer 1: Static Dictionary (0.5 MB)
- Precomputed lookup table of most common n-grams in FineWeb
- Top 10K bigrams, top 50K trigrams
- Hash-based lookup — O(1) per token
- Provides base prediction for common patterns
- Built during first pass of training data (seconds, not minutes)

### Layer 2: Adaptive N-gram Context Model (2 MB)
- Variable-order n-gram model (order 1-7)
- Kneser-Ney smoothing for unseen n-grams
- Handles local syntactic patterns
- O(1) per token lookup
- Trainable — updated during training phase
- Can adapt per-document during TTT (eval phase)

### Layer 3: Micro-Transformer Residual (13 MB)
- Input: residual signal (what Layers 1+2 couldn't predict)
- 6 layers, 384 dim, 6 heads (3 KV heads)
- BitNet 1.58-bit ternary weights {-1, 0, +1}
- BitNet gives ~4x more parameters than int6 for same budget
- Only processes tokens where Layer 2 uncertainty is high
- Output: correction to the n-gram distribution

### LPC Preprocessing
- Linear Predictive Coding applied to token sequences
- Removes linear predictability from the signal
- Neural model only handles the residual (non-linear patterns)
- Borrowed from audio compression (proven theory)

### ANS-Aware Loss
- Training loss = actual bpb via Asymmetric Numeral Systems encoding
- Not cross-entropy (which is a proxy for bpb)
- Directly optimizes the competition metric

## Build Instructions for Codex

### Phase 1: Dictionary Builder
- Script to analyze FineWeb training data
- Count all bigrams and trigrams
- Build hash table of top-N patterns
- Output: dictionary file + lookup function
- **Verify:** dictionary correctly matches known patterns

### Phase 2: N-gram Model
- Implement variable-order Kneser-Ney n-gram model
- Train on FineWeb training data
- Output: probability distribution for each token given context
- **Verify:** achieves ~2.3 bpc standalone (bzip2-level)

### Phase 3: Residual Calculation
- Run Layers 1+2 on training data
- Calculate per-token entropy/uncertainty
- Identify "hard" tokens (high entropy under n-gram model)
- Output: hard token dataset for Phase 4

### Phase 4: Micro-Transformer
- Build small transformer with BitNet 1.58-bit weights
- Train ONLY on hard tokens (residual signal)
- Input is the residual, output is distribution correction
- **Verify:** reduces bpb when combined with Layers 1+2

### Phase 5: Integration
- Combine all three layers into single train_gpt.py
- Joint optimization pass
- Implement LPC preprocessing
- Implement ANS-aware loss
- **Verify:** end-to-end bpb is better than any layer alone

### Phase 6: TTT
- Implement test-time training that adapts Layer 2 (n-gram model) per document
- AdamW optimizer during eval, only on already-evaluated tokens
- **Verify:** per-document adaptation improves bpb

## Key Risks
- BitNet 1.58-bit training may be unstable in 10 minutes
- Three-layer composition may have gradient flow issues
- ANS-aware loss may not be differentiable (may need STE)
- N-gram model + neural model interpolation weights need tuning

## Fallback
If BitNet doesn't work → use int6 quantization like everyone else
If LPC doesn't help → drop it, use standard input
If ANS loss doesn't work → fall back to cross-entropy
Each component should be independently toggleable

## Output
- `train_gpt_model1.py` — complete codec model
- `build_dictionary.py` — dictionary builder (can be run as preprocessing)
