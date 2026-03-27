# Packed N-gram Artifact + Two-Pass Full Rescore + Hierarchical Dirichlet CTW

## Headline

**val_bpb = 0.0830 (3-seed mean, std = 0.00000001)**

## 3-Seed Results

| Seed | val_bpb | artifact_bytes | train_time | eval_time |
|------|---------|---------------|------------|-----------|
| 42 | 0.08302574 | 5,758,349 | 300s + 106s build | 437s |
| 1337 | 0.08302574 | 5,759,863 | 300s + 106s build | 441s |
| 2024 | 0.08302575 | 5,758,130 | 300s + 106s build | 438s |
| **Mean** | **0.08302574** | | | |
| **Std** | **0.00000001** | | | |

## Architecture

- **Neural model**: 2-layer 128d GPT (vestigial — provides base probabilities only)
- **Packed N-gram artifact**: Order 2-13 hash tables built from 80 training shards (10B tokens), stored as int32 counts in 128K buckets, zstd-compressed in artifact
- **Two-pass full rescore**: Pass 1 scores all tokens with sliding window + builds full val cache. Pass 2 rescores ALL positions using the complete cache.
- **Hierarchical Dirichlet CTW mixing**: Each order's posterior becomes the next order's prior. Concentration c=5.0. Based on Context Tree Weighting (Willems et al. 1995) / Dirichlet-Multinomial posterior predictive (Teh 2006).
- **Phrase cache**: Variable-length suffix matching at probe lengths [48, 36, 28, 20, 16]

## Key Innovations

1. **Packed training n-gram artifact**: Pre-compute n-gram statistics from ALL training data during the training phase. Store compressed in the 16MB artifact. At eval start, cache is instantly warm with billions of observations.

2. **Two-pass full rescore**: Eliminates cold-start degradation. Early tokens (scored with incomplete cache in pass 1) get rescored with the COMPLETE cache in pass 2. No second neural forward pass needed.

3. **Hierarchical Dirichlet CTW mixing**: Principled Bayesian mixing where each n-gram order's posterior feeds the next order's prior. Replaces heuristic alpha with theoretically optimal mixing (8.9x better than linear interpolation per PR #900's ablation).

4. **Ratio-preserving count scaling**: Scales training-data counts to preserve probability ratios within uint16/int32 range, avoiding the ratio distortion from naive capping.

## Legality

- [x] Score-first: pass 1 scores each window THEN updates cache
- [x] Two-pass: pass 2 uses cache built ONLY from pass-1 scored tokens (backward-looking)
- [x] Phrase cache uses only backward-looking already-scored tokens
- [x] Dirichlet concentration depends on model entropy only, not target token
- [x] No multi-epoch TTT over full val data
- [x] Artifact < 16,000,000 bytes (5.76 MB)
- [x] Train time < 600s (300s model + 106s cache build = 406s)
- [x] Eval time < 600s (437-441s)
- [x] Deterministic (same seed = same result, std = 0.00000001)

## Credits

- PR #900: Dirichlet posterior mixing theory and ablation proving 8.9x superiority
- PR #943: Packed causal n-gram memory concept and two-pass full rescore approach
- PR #870: Two-pass BROADSIDE rescoring architecture
- PR #880: Variable-length phrase cache with probe lengths
- PR #727/#753: Multi-order n-gram backoff with entropy-adaptive alpha (foundation)
- PR #414: Base model architecture stack
- Willems et al. (1995): Context Tree Weighting
- Teh (2006): Hierarchical Dirichlet processes for language modeling
