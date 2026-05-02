# Parameter Golf Competitive Analysis
## Date: 2026-03-18/19
## Analysis of 30+ PRs from the openai/parameter-golf competition

---

## 1. LEADERBOARD — Current Best Results

| Rank | PR | Author | val_bpb | Hardware | Steps | Technique | Status |
|------|-----|--------|---------|----------|-------|-----------|--------|
| 🥇 | #39 | nanlliu | **1.2230** | 8×H200 | 14,421 | Lower LR (MATRIX_LR=0.02) | VERIFIED ✅ |
| 🥈 | — | baseline | 1.2244 | 8×H100 | 13,780 | Default config | Official ✅ |
| 🥉 | #31 | JackYoung27 | 1.2663 | 4×H100 SXM | 2,651 | 5×3 d768 recurrence | Partial (still improving) |
| 4 | #38 | kxddry | 2.962* | MLX local | 200 | 3×3 d768 QAT+LoRA | WIP (local only) |
| 5 | #37 | khasinski | 1.3217 | 1×5090 | 938 | SP4096 vocab | WIP |
| 6 | #33 | JusticeShultz | 2.061 | 1×3090 | — | Compression-aware training | WIP |
| 7 | #40 | bahman2017 | 3.231 | MLX | — | PhiDelay mixer | WIP |
| 8 | #29 | nidhilak-Aquarius | N/A | — | — | 1×12 recurrence + MQA | No results |

*PR #38 result is on 200 steps local only

---

## 2. TECHNIQUE BREAKDOWN — What Everyone Is Trying

### A. Depth Recurrence / Weight Sharing (7 PRs — MOST POPULAR)
The dominant approach, tried by ~50% of serious submissions:

| PR | Config | dim | Unique × Loops | Eff layers | LoRA? | Result |
|----|--------|-----|----------------|------------|-------|--------|
| #31 | 5×3 | 768 | 5 × 3 | 15 | No | 1.2663 (4×H100) |
| #21 | 5×2 | 704 | 5 × 2 | 10 | rank-16 | No result yet |
| #38 | 3×3 | 768 | 3 × 3 | 9 | rank-4 | 2.962 (local) |
| #8 | 4×3 | 640 | 4 × 3 | 12 | No (gates) | No result (M3) |
| #29 | 1×12 | 512 | 1 × 12 | 12 | No | No result |
| #15 | ? | ? | recursive | ? | ? | No details |
| #5 | 3→9 | ? | 3 → 9 | 9 | No | No result |

**KEY PATTERNS:**
- 5+ unique blocks needed (3 blocks consistently fails — our exp003 confirms this)
- dim=640-768 is the sweet spot after freeing params via weight sharing
- LoRA adapters (rank 4-16) used for per-loop specialization
- Nobody has validated recurrence at full 8×H100 scale yet!
- PR #31's 1.2663 was on ONLY 2651 steps (4×H100), still improving at cutoff

### B. Hyperparameter Tuning (2 PRs)
| PR | Change | Impact |
|----|--------|--------|
| #39 | MATRIX_LR: 0.04→0.02, SCALAR_LR→0.02, EMBED_LR→0.03 | **-0.0056 BPB** 🏆 |
| #4 | Muon + muP exploration | No results |

**CRITICAL INSIGHT: PR #39 is the ONLY submission that beat the baseline, and it did so with JUST a learning rate reduction. No architecture changes at all.**

The LR sweep showed:
- 0.06 → 1.2445 (+0.016, much worse)
- 0.04 → 1.2286 (baseline on H200)
- 0.03 → 1.2279 (-0.001)
- 0.025 → 1.2250 (-0.004)
- **0.02 → 1.2230 (-0.006)** ← sweet spot
- 0.015 → 1.2234 (-0.005, slightly worse)

BUT: Run on H200 (41.6ms/step vs 43.5ms/step on H100) = ~400 extra steps = ~2.8% more training. Some of the 0.006 BPB improvement may come from extra steps, not just LR.

### C. QAT — Quantization-Aware Training (3 PRs)
| PR | Approach | Impact |
|----|----------|--------|
| #38 | STE fake-quantize in CastedLinear forward, claims 18× less int8 degradation | WIP |
| #20 | STE fake_quantize_per_row in training loop | WIP |
| #33 | Compression-aware regularization + outlier penalty | Best local: 2.061 |

**PR #38's QAT is the most sophisticated:** fake_quantize_int8 uses STE (straight-through estimator) where the forward pass simulates int8 rounding noise but gradients pass through unchanged. Claims int8 degradation drops from 0.002 BPB to 0.00012 BPB. BUT this hasn't been validated at scale.

**PR #33's compression-aware approach is novel:** adds regularization that penalizes weight distributions that compress poorly. Small gain (0.05 BPB on local 3090 proxy), but interesting angle nobody else is trying.

### D. Tokenizer / Vocab Changes (1 PR)
| PR | Change | Impact |
|----|--------|--------|
| #37 | SP1024 → SP4096 | 12.4% BPB improvement (1.5086 → 1.3217 on 1×5090) |

**VERY PROMISING:** 4× larger vocab = 26% better text compression ratio (0.306 vs 0.414 tokens/byte). Fewer tokens per byte = better BPB directly. The 1×5090 comparison shows dramatic improvement. BUT:
- Adds embedding params: 4096×512 = 2M params (4× more than 1024×512)
- Artifact grows from 9.8MB to 13.6MB (still under 16MB)
- Needs 8×H100 validation
- Step speed IMPROVED (640ms vs 1316ms) because fewer tokens per sequence = fewer attention operations

### E. SwiGLU MLP (3 PRs)
| PR | Config |
|----|--------|
| #21 | SwiGLU replacing relu² with 2/3 expansion for param neutrality |
| #8 | SwiGLU in recurrent model |
| #29 | SwiGLU FFN |

SwiGLU is popular but NONE have validated vs relu² on this specific task. The parameter count is roughly equivalent (3 matrices at 2/3 width vs 2 matrices at full width).

### F. Multi-Token Prediction (1 PR)
| PR | Approach |
|----|----------|
| #21 | MTP auxiliary heads with separate projection layers |

PR #21 uses dedicated MTPHead modules (proj + norm per head). This adds parameters that must fit in the 16MB budget. Our exp006/007/008 showed MTP hurts at 2K steps — but modded-nanogpt showed it helps at full training length.

### G. Exotic / Novel Approaches (3 PRs)
| PR | Technique | Status |
|----|-----------|--------|
| #40 | PhiDelay mixer (replaces self-attention with delay operator) | 3.23 BPB — terrible |
| #5 | Sliding window attention + recursive sharing | No results |
| #33 | Eval-time sidecar (bigram + cache mixing at inference) | Slight regression |
| #16 | "BioDNA + FractalLinear" | Likely unserious |

### H. Other Training Tricks Mentioned but Not Validated
- **LAWA** (Checkpoint averaging during warmdown): PR #38
- **NTK RoPE scaling** for 2048-token eval (trained at 1024): PR #38
- **EMA** model averaging: PR #21
- **NuMuon** (nuclear norm constraint for compression-friendly weights): PR #38
- **Byte grouping** for better zlib compression: PR #38
- **Value embeddings with gating**: mentioned in modded-nanogpt but NOT in any PR

---

## 3. CRITICAL ANALYSIS — Where the Competition Actually Stands

### The Uncomfortable Truth
**Almost nobody has validated their ideas at scale.** The ONLY verified result that beats baseline is PR #39's learning rate tune, and even that has the H200 confound.

- PR #31 (best architecture change) ran on 4×H100 and got ONLY 2651 steps — vs baseline's 13780. At that step count, 1.2663 is NOT competitive.
- ALL recurrence submissions are WIP with no 8×H100 validation
- ALL QAT submissions are WIP
- The SP4096 vocab change is very promising but also WIP

### What's Actually Working
1. **Lower learning rate (+0.006 BPB)** — simplest, most validated improvement
2. **Larger vocab (potentially +0.19 BPB from SP4096)** — dramatic but unvalidated at scale
3. **Depth recurrence with 5+ unique blocks** — promising direction, needs speed optimization

### What's NOT Working
1. **1-3 unique blocks (too few)** — our experiments + PR #29 (1×12) theory confirm this
2. **PhiDelay / exotic mixers** — replacing attention is too risky
3. **Eval-time tricks (cache, bigram)** — PR #33 shows marginal-to-negative effect
4. **BitNet ternary** — PR #40 tried BitLinear and got 6.09 BPB (catastrophic)

### What Nobody Has Tried (OPPORTUNITY ZONES 🎯)
1. **Lower LR + recurrence COMBINED** — PR #39's LR=0.02 + PR #31's 5×3 d768
2. **Lower LR + larger vocab COMBINED** — PR #39 + PR #37
3. **Batch size curriculum** (from modded-nanogpt) — 131K → 262K → 393K
4. **Warmdown schedule tuning** — longer warmdown, different LR floor
5. **Adam β2 tuning** — baseline uses 0.95, modern LLMs often use 0.99
6. **Attention scale hardcoding** (from modded-nanogpt, saved 20 steps)
7. **60% cooldown** (from modded-nanogpt)
8. **Gradient clipping** — most PRs don't use it
9. **Wider model WITHOUT recurrence** — e.g., 7 layers at dim=576 (fewer layers, wider)
10. **INT4 quantization** — could fit 2× more params in 16MB, but training harder

---

## 4. STRATEGIC RECOMMENDATIONS

### Tier 1: Do Immediately (Highest ROI, Lowest Risk)
1. **Lower LR to 0.02** on our baseline. This is FREE and proven (+0.006 BPB). Apply to ALL future experiments.
2. **Run our best config (7×2 d672) with MATRIX_LR=0.02** at full scale. Combining our best architecture with the best hyperparams.

### Tier 2: High Priority Experiments
3. **SP4096 tokenizer** — PR #37 showed 12.4% BPB improvement. Train SP4096 tokenizer and test. The embedding params increase is worth it.
4. **5×2 d704 with LR=0.02** — Our dim=672 showed promise, PR #31's dim=768 worked. Try the middle ground with lower LR.
5. **QAT with STE** — Copy PR #38's fake_quantize_int8 into our training. Simple change, potentially +0.03 BPB from eliminating quantization gap.

### Tier 3: Combo Experiments (After Tier 1-2)
6. **LR=0.02 + SP4096 + 7×2 recurrence** — the nuclear option combining all validated improvements
7. **Warmdown tuning** — try 2000 warmdown iters instead of 1200 (with LR=0.02 we may benefit from longer cooldown)
8. **Batch curriculum** — start at 262K tokens/step, ramp to 524K or 786K

### What to SKIP
- ❌ SwiGLU (unvalidated, adds complexity for unclear benefit)
- ❌ PhiDelay / exotic attention replacements (proven to fail)
- ❌ BitNet (proven to fail at this scale)
- ❌ Eval-time cache/bigram mixing (marginal at best)
- ❌ 1-3 unique blocks in recurrence (proven insufficient)

---

## 5. COMPETITIVE TIMELINE & RISK ASSESSMENT

### Current State of Play
- **Leader**: PR #39 at 1.2230 BPB (just an LR change!)
- **Our best**: exp014 at 1.2797 (2K steps only, 1×H100)
- **Gap to beat**: We need < 1.2230 BPB with p<0.01

### Estimated Impact of Combined Improvements
| Technique | Expected BPB improvement | Confidence |
|-----------|-------------------------|------------|
| LR=0.02 (from PR #39) | +0.006 | HIGH (verified) |
| 7×2 d672 recurrence (exp014) | +0.018 at 2K, ~+0.01 at full | MEDIUM |
| QAT STE (from PR #38) | +0.01-0.03 | MEDIUM |
| SP4096 vocab (from PR #37) | +0.05-0.10 | HIGH (if validated) |
| Warmdown tuning | +0.002-0.005 | LOW |

**Estimated combined potential: 0.03-0.06 BPB over baseline = 1.17-1.19 BPB**

### Main Risks
1. **Speed penalty from recurrence**: 7×2 recurrence is 2.5× slower per step → fewer total steps. May offset quality gains.
2. **SP4096 artifact size**: larger embedding + larger vocab = larger artifact. Must verify it still fits in 16MB.
3. **Compute access**: We need reliable 8×H100 with NVLink mesh for final validation. Thunder Compute's topology issues are a blocker.
4. **LR interaction**: LR=0.02 was tuned FOR the baseline architecture. Different architectures (recurrence, wider) may have different optimal LR.

---

## 6. KEY INTEL FROM COMPETITOR CODE

### PR #38's Most Interesting Ideas (kxddry)
1. **Byte grouping for zlib**: Separates int8 weight bytes from fp16 scale bytes before compression, claims 2× better zlib ratio. This is a FREE compression improvement that could give us more parameter headroom.
2. **NTK RoPE scaling**: Evaluates at 2048 tokens despite training at 1024. `eval_seq_len = 2048` with scaled RoPE base. Could improve BPB by giving longer context during eval.
3. **LAWA** (Latest Averaged Weights Averaging): Takes snapshots every 100 steps during warmdown and averages them. FREE quality boost.
4. **NuMuon**: Nuclear norm constraint variant of Muon. Cosine-annealed SVD rank parameter. Interesting but complex.
5. **LoRA on every effective layer** (not just per-loop): `num_shared_blocks * num_loops` LoRA deltas vs PR #21's per-loop sharing. More adaptation capacity.

### PR #21's Most Interesting Ideas (monroestephenson)
1. **LoRA on Q/K/V/O** (4 adapters per loop) vs PR #38's single LoRA delta on the block output
2. **Per-recurrence scales** on every effective layer (dim-sized learnable scale)
3. **U-Net skip connections work with recurrence** — maps encoder/decoder halves across effective layers
4. **MTP with separate heads** — MTPHead has its own projection + norm, uses shared embedding

### PR #33's Most Interesting Ideas (JusticeShultz)
1. **Compression-aware regularization**: COMPRESSION_REG_WEIGHT penalizes weights that don't compress well with zlib
2. **Outlier regularization**: OUTLIER_REG_WEIGHT penalizes weight outliers that quantize poorly
3. **Roundtrip-proxy evaluation**: evaluates on the ACTUAL quantized model during training, not just pre-quant

---

## APPENDIX: All PRs Categorized

### Serious Competition Submissions (with results or promising approach)
- PR #39: Lower LR — **CURRENT LEADER 1.2230**
- PR #31: 5×3 d768 recurrence — 1.2663 (partial)
- PR #21: 5×2 d704 recurrence + LoRA + MTP + EMA
- PR #38: 3×3 d768 recurrence + QAT + LoRA + LAWA
- PR #37: SP4096 vocabulary — 1.3217 (single GPU)
- PR #20: QAT + architecture exploration
- PR #33: Compression-aware training

### Research/Exploratory (WIP, no competitive results)
- PR #11: 3×5 d640 looped + LoRA + MTP
- PR #5: Sparse attention + recursive sharing
- PR #8: 4×3 d640 recurrence + SwiGLU (M3)
- PR #29: 1×12 universal transformer
- PR #15: Recursive weight sharing (no details)
- PR #4: Muon/muP research

### Non-competitive / Infrastructure
- PR #40: PhiDelay (3.23 BPB — not competitive)
- PR #16: "BioDNA" (likely unserious)
- PR #24: Score harness
- PR #13: Sweep helpers
- PR #14: CPU fallback
- PR #19: MLX requirements
- PR #36: MLX smoke test fix
- PR #34: Dispersion loss (closed)
