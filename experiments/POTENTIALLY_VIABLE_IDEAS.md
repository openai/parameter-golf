# Potentially Viable Ideas — March 21, 2026

**Current SOTA: 1.1187 sliding BPB (exp113, 15.95MB)**

Ideas that could push below 1.1187. Ranked by viability.

---

## Tier 1: High Viability (worth implementing)

### 1. Product Quantization (PQ) for weight compression
**Status: IMPLEMENTING**
- Replace int6 scalar quantization with vector quantization (PQ) for MLP weights
- Cluster 8-dim sub-vectors into 256 codebook entries, store 8-bit indices
- Exploits weight correlations that scalar int6 ignores
- Expected: 2-3x better compression than int6 for same reconstruction error
- Could save 2-3MB -> enough for 15-16 layers at WD=0.06
- Implementation: post-training only (no QAT needed), ~100 lines
- Risk: codebook overhead, interaction with TTT dequantization

### 2. Iterative Lottery-Ticket Pruning + Regrowth
**Status: VIABLE, NOT YET TRIED**
- Every 2000 steps: prune 20% weakest weights, regrow via gradient magnitude
- End with 30-40% structured sparsity (whole rows zeroed)
- zstd crushes structured zeros -> massive compression savings
- Unlike our one-shot post-training pruning (exp115, hurt 0.018 BPP), iterative lets model adapt
- Could free 1-2MB for extra layers
- Risk: only ~3 prune-regrow cycles in 6660 steps, interaction with Muon optimizer

### 3. XSA (Exclusive Self Attention) from PR#265
**Status: VIABLE, NOT YET TRIED WITH FA3**
- After attention output, subtract each token's self-value projection
- Forces model to learn from context rather than self-reference
- Zero extra parameters, ~2ms/step overhead
- PR#265 claims +0.002 BPP on last 3 layers
- We tried exp104 (before FA3) and it hurt, but PR#265's efficient GQA-aware version may work
- Risk: may not help with our specific architecture/config

### 4. Multi-Token Prediction (MTP) auxiliary loss
**Status: VIABLE, ALREADY IN CODEBASE**
- Our script already has MTP_NUM_HEADS parameter (currently 0)
- Add auxiliary next-2 and next-3 token prediction heads during training
- Weighted loss (e.g., 0.2 weight on auxiliary heads)
- Improves representation quality at zero inference cost (heads are dropped at eval)
- Risk: may slow training per step, uncertain benefit at this scale

---

## Tier 2: Medium Viability (interesting but uncertain)

### 5. Better Tokenizer (vocab 2048-4096)
- Larger vocab = fewer tokens per byte = better BPB by construction
- But embedding table grows (2048 vocab x 512 dim = 1M params extra)
- Need to retrain tokenizer on FineWeb
- Risk: larger embedding eats param budget, may not compress well

### 6. Depth Recurrence (weight sharing)
- Share weights across transformer blocks (e.g., 4 unique blocks x 3 loops = 12 effective)
- Freed params go to wider model (dim=640+)
- Per-iteration layer norms break symmetry
- Risk: quality loss from sharing, untested at this scale with our specific config

---

## Tier 3: Tested and Failed

### 7. Mamba-2 SSM
- **FAILED**: 15L = 1.2646 BPP (0.14 worse than attention)
- 119-155ms/step without torch.compile vs 90ms with FA3
- Not competitive at this param scale

### 8. KAN (Kolmogorov-Arnold Networks)
- **FAILED**: 361ms/step (4x slower than attention)
- B-spline operations too expensive without custom CUDA kernels
- Can't compensate for 75% fewer training steps

### 9. Attention Gate (per-head sigmoid)
- **FAILED**: exp109 = 1.1229 (marginally better) but 16.17MB (over limit)
- Extra params push artifact over budget for negligible BPP gain

### 10. Structured Channel Pruning (post-training)
- **FAILED**: exp115 = 1.1369 (+0.018 worse)
- One-shot structured pruning is too destructive
- (Iterative version in Tier 1 could work differently)

### 11. Hypernet Weight Generation
- **NOT VIABLE**: Low-rank factorization with extra complexity
- Gradient flow issues, Muon incompatibility, quantization amplification
- Model capacity isn't the bottleneck — training steps are

---

## Key Insight from Ablation Study

The bottleneck hierarchy is:
1. **Training steps** (FA3 gave +1000 steps -> -0.004 BPP)
2. **Compression efficiency** (manual serialization saved 330KB -> fit under 16MB)
3. **Architecture capacity** (more layers help but are limited by 16MB)
4. **Hyperparameter tuning** (WD=0.06 is Pareto-optimal, activation 0.5 is optimal)

Any viable improvement must either:
- A) Get more training steps (faster per-step or less overhead)
- B) Compress better (PQ, lottery pruning) to fit more capacity
- C) Extract more quality per step (MTP, XSA)
