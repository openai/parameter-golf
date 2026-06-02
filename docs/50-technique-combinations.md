# 50 Technique Combinations for Parameter Golf

> **Goal**: Identify the most promising combinations to beat SOTA and get recognized by OpenAI.
> **Generated**: April 19, 2026 | **Deadline**: April 30, 2026 (11 days)
> **Strategy**: Pick 10 of these to run as experiments on M4, then deploy the best on 8xH100.

---

## Our M4 Baseline vs The Field

| Run | BPB | Steps | Hardware | Notes |
|-----|-----|-------|----------|-------|
| **Our M4 Baseline** | **3.2178** | 71 | Mac M4, 1 shard | 10-min wall-clock, int8+zlib, 5.6MB artifact |
| Naive Baseline (official) | 1.2244 | ~6000+ | 8xH100, 80 shards | Same architecture, full data |
| Current Official SOTA | 1.1147 | ~6900 | 8xH100, 80 shards | Full technique stack (PR #1019) |
| Best Pending (TTT-based) | ~1.07 | ~7000 | 8xH100, 80 shards | SP8192 + TTT + QK-Gain |
| Best Pending (Tokenizer) | ~0.95 | ~7000 | 8xH100, 80 shards | Scylla/TokenMonster 998-vocab |

### Why Our Score is 3.22 (Not a Problem)

Our 3.22 BPB is **expected and normal** for M4 local testing:

| Factor | M4 Local | 8xH100 Official | Impact |
|--------|----------|-----------------|--------|
| Training shards | 1 (100M tokens) | 80 (8B tokens) | **~0.10 BPB penalty** |
| Training steps | 71 | 6,900+ | **~1.5 BPB penalty** (huge) |
| Batch size | 65K tokens/step | 524K tokens/step | **~0.05 BPB penalty** |
| Techniques | None (baseline) | Full SOTA stack | **~0.10 BPB penalty** |
| Hardware throughput | ~10K tok/s | ~800K tok/s | Fewer steps in 10 min |

**M4 is a directional tool**: if Combo A scores 2.80 and Combo B scores 2.95 on M4, A is better. Absolute numbers don't matter -- relative ordering does.

### Scaling Factor: M4 to H100

Based on the depth recurrence research (PR #363), 1-shard single-GPU runs consistently show ~0.10-0.15 BPB higher than 80-shard 8xH100 runs at the same step count. The step-count gap is the dominant factor. On M4 with 10 min, expect ~70-100 steps. On 8xH100, expect ~6,500-7,200 steps.

**Rule of thumb**: If a technique combination improves M4 BPB by 0.05, it should improve 8xH100 BPB by ~0.01-0.02.

---

## Technique Catalog (Building Blocks)

### Architecture Techniques

| ID | Technique | Est. BPB Gain | Complexity | Status |
|----|-----------|---------------|------------|--------|
| A1 | NUM_LAYERS=11 (from 9) | -0.003 | Trivial | In SOTA |
| A2 | MLP_MULT=3 (1536 hidden) | -0.005 | Trivial | In SOTA |
| A3 | LeakyReLU(0.5)^2 activation | -0.002 | Trivial | In SOTA |
| A4 | SmearGate | -0.010 | Medium | In SOTA |
| A5 | U-Net skip connections | -0.003 | Medium | In SOTA |
| A6 | XSA all layers | -0.005 | Medium | In SOTA |
| A7 | XSA last 4 layers only | -0.003 | Medium | Older record |
| A8 | Partial RoPE (16/64 dims) | -0.002 | Trivial | In SOTA |
| A9 | LN Scale 1/sqrt(layer+1) | -0.001 | Trivial | In SOTA |
| A10 | VE128 on layers 9-10 | -0.001 | Medium | In SOTA |
| A11 | BigramHash 3072x112 | -0.005 | Medium | In SOTA |
| A12 | BigramHash 4096+ | -0.002 extra | Medium | Untested at scale |
| A13 | QK-Gain 4.0-5.25 (from 1.5) | -0.003 to -0.006 | Trivial | New, April PRs |
| A14 | Focal loss (P2, gamma=2) | -0.002 | Low | New, April PRs |
| A15 | Anti-Layer Removal | -0.001 | Low | New, research |

### Optimizer Techniques

| ID | Technique | Est. BPB Gain | Complexity | Status |
|----|-----------|---------------|------------|--------|
| O1 | Muon momentum 0.99 | -0.003 | Trivial | In SOTA |
| O2 | Muon WD 0.04 | -0.002 | Trivial | In SOTA |
| O3 | Warmdown 3500-4000 iters | -0.003 | Trivial | In SOTA |
| O4 | EMA(0.997) + SWA(every 50) | -0.003 | Low | In SOTA |
| O5 | Parallel Muon + Param Banking | Systems speedup | High | In SOTA (H100 only) |
| O6 | Turbo-Muon (AOL + Polar Express) | -0.002 + speed | High | New, April PRs |
| O7 | Progressive window warmup | -0.001 | Medium | New, research |

### Quantization Techniques

| ID | Technique | Est. BPB Gain | Complexity | Status |
|----|-----------|---------------|------------|--------|
| Q1 | Int6 STE QAT | -0.010 | Medium | In top runs |
| Q2 | Late QAT (at LR < 0.15) | -0.002 | Low | In SOTA |
| Q3 | Full Hessian GPTQ + AR self-gen | -0.008 | High | In SOTA |
| Q4 | GPTQ-lite (diagonal Hessian) | -0.005 | Medium | Older approach |
| Q5 | Selective +/-1 pruning | -0.001 | Low | In SOTA |
| Q6 | Hadamard Rotation pre-quant | -0.003 | Medium | New (PolarQuant paper) |
| Q7 | AWQ (Activation-Aware) | -0.003 | Medium | New, April PRs |
| Q8 | Layer-wise mixed precision | -0.002 | Medium | New, April PRs |
| Q9 | Noisy QAT (calibrated noise) | -0.002 | Low | From PR #363 |
| Q10 | Int5 mixed MLP | -0.003 (more params) | Medium | In mid-range runs |

### Compression Techniques

| ID | Technique | Est. BPB Gain | Complexity | Status |
|----|-----------|---------------|------------|--------|
| C1 | LZMA preset=9 | -0.001 | Trivial | In SOTA |
| C2 | Zstandard level 22 | -0.0005 | Trivial | Alternative |
| C3 | Brotli-11 + byte-shuffle | -0.002 (580KB saved) | Medium | New, April PRs |

### Evaluation Techniques

| ID | Technique | Est. BPB Gain | Complexity | Status |
|----|-----------|---------------|------------|--------|
| E1 | Sliding window eval (stride=64) | -0.019 | Low | In top runs |
| E2 | Sliding window eval (stride=16) | -0.025 | Low | New, better |
| E3 | Temperature scaling (T=0.90) | -0.002 | Trivial | PR #641 |
| E4 | SLOT (output-head delta TTT) | -0.018 to -0.028 | Medium | New, April PRs |
| E5 | LoRA TTT (legal score-first) | -0.0025 | High | PR #549 |
| E6 | LaCT (Large Chunk TTT) | -0.005 | High | ICLR 2026 paper |
| E7 | MP-SGD TTT (multi-phase) | -0.010 | High | New, April PRs |

### Tokenizer Techniques

| ID | Technique | Est. BPB Gain | Complexity | Status |
|----|-----------|---------------|------------|--------|
| T1 | SP8192 (larger BPE vocab) | -0.020 | High | New, April PRs |
| T2 | Scylla/TokenMonster 998-vocab | -0.030 | High | New, PR #1184 |
| T3 | CaseOps Tokenizer | -0.015 | High | New, PR #1729 |

### Systems Techniques (H100 only)

| ID | Technique | Est. Effect | Complexity | Status |
|----|-----------|-------------|------------|--------|
| S1 | Flash Attention 3 (Hopper) | More steps in 10 min | Medium | In SOTA |
| S2 | Liger-Kernel fused ops | 10-15% throughput | Medium | New |
| S3 | V:N:M activation sparsity | 1.4-1.7x speedup | High | Research |
| S4 | torch.compile | Variable | Low | Standard |

### Hybrid Techniques (Neural + Classical)

| ID | Technique | Est. BPB Gain | Complexity | Status |
|----|-----------|---------------|------------|--------|
| H1 | N-gram cache (orders 2-5) | -0.050 to -0.100 | High | New, April PRs |
| H2 | Complementary training | -0.020 | High | New, April PRs |
| H3 | BackoffNgramMixer (entropy-adaptive) | -0.100+ | Very High | New, best pending |

---

## The 50 Combinations

### Tier A: Foundation Stacks (Combos 1-10)
*Reproduce known-good stacks. These are table stakes for a competitive submission.*

#### Combo 1: "Exact SOTA Replica"
**Target**: ~1.115 BPB | **Risk**: Low | **Track**: Record
```
A1 + A2 + A3 + A4 + A5 + A6 + A8 + A9 + A10 + A11
O1 + O2 + O3 + O4 + O5
Q1 + Q2 + Q3 + Q5
C1
E1 + E3
```
**Rationale**: Exact reproduction of the merged SOTA (PR #1019). Baseline to measure all other combos against. Must score ~1.115 on 8xH100 to validate your pipeline.
**M4 test config**: `NUM_LAYERS=11 MLP_MULT=3 WARMDOWN_ITERS=1200`

---

#### Combo 2: "SOTA Minus TTT/GPTQ" (M4-Testable)
**Target**: ~1.135 BPB (H100) | **Risk**: Very Low | **Track**: Record foundation
```
A1 + A2 + A3 + A8 + A9
O1 + O3 + O4
Q1 + Q2
C1
E1
```
**Rationale**: Core SOTA architecture without H100-only features (Flash Attn 3, Parallel Muon, Full GPTQ). Testable on M4. Good for validating architecture changes.
**M4 test config**: `NUM_LAYERS=11 MLP_MULT=3 WARMDOWN_ITERS=1200 MUON_MOMENTUM=0.99`

---

#### Combo 3: "Pre-TTT SOTA" (PR #549 Replica)
**Target**: ~1.119 BPB | **Risk**: Low | **Track**: Record
```
A1 + A2 + A3 + A4 + A5 + A7 + A8 + A9 + A10
O1 + O2 + O3 + O4 + O5
Q1 + Q2 + Q4
C1
E1 + E3 + E5
```
**Rationale**: The TTT-inclusive SOTA from PR #549. Useful to compare TTT vs no-TTT path.

---

#### Combo 4: "Minimal Competitive" (Fewest Techniques)
**Target**: ~1.13 BPB | **Risk**: Very Low | **Track**: Record
```
A1 + A2 + A6 + A11
O1 + O3 + O4
Q1 + Q3
C1
E1
```
**Rationale**: Minimum viable competitive stack. Tests whether fewer, high-impact techniques can match SOTA.

---

#### Combo 5: "EMA-Heavy Foundation"
**Target**: ~1.12 BPB | **Risk**: Low | **Track**: Record
```
A1 + A2 + A3 + A6 + A11
O1 + O2 + O3 + O4 (EMA=0.998, SWA=30)
Q1 + Q3
C1
E1 + E3
```
**Rationale**: Push EMA decay higher (0.998 vs 0.997) and SWA more frequently (every 30 vs 50). The depth recurrence paper showed SWA can actually *help* quantization.

---

#### Combo 6: "Warmdown-Tuned Foundation"
**Target**: ~1.11-1.12 BPB | **Risk**: Low | **Track**: Record
```
A1 + A2 + A3 + A6 + A8 + A11
O1 + O2 + O3 (WARMDOWN=5000) + O4
Q1 + Q3
C1
E1
```
**Rationale**: Push warmdown to 5000 (beyond the tested 4000). The longer warmdown may help the model settle into a better minimum for quantization.

---

#### Combo 7: "High QK-Gain Foundation"
**Target**: ~1.11 BPB | **Risk**: Low | **Track**: Record
```
A1 + A2 + A3 + A6 + A11 + A13 (QK_GAIN=4.0)
O1 + O2 + O3 + O4
Q1 + Q3
C1
E1
```
**Rationale**: QK-Gain scaling from 1.5 to 4.0 is a new finding from April PRs. Simple config change that reportedly gives -0.003 to -0.006 BPB.
**M4 test config**: `QK_GAIN_INIT=4.0`

---

#### Combo 8: "BigramHash-Wide Foundation"
**Target**: ~1.11 BPB | **Risk**: Low | **Track**: Record
```
A1 + A2 + A3 + A6 + A12 (BigramHash 4096x128)
O1 + O2 + O3 + O4
Q1 + Q3
C1
E1
```
**Rationale**: Push BigramHash to 4096 buckets with dim=128. Previous SOTA stopped at 3072x112 due to artifact budget. With better compression (Brotli), wider might fit.

---

#### Combo 9: "Sliding Stride-16 Foundation"
**Target**: ~1.10 BPB | **Risk**: Low | **Track**: Record
```
A1 + A2 + A3 + A6 + A11
O1 + O2 + O3 + O4
Q1 + Q3
C1
E2 (stride=16) + E3
```
**Rationale**: Stride-16 sliding eval vs stride-64. The binary submission reported -0.015 BPB from stride-16 alone. More eval time but within the 10-min eval budget.

---

#### Combo 10: "Full Foundation + Focal Loss"
**Target**: ~1.10 BPB | **Risk**: Low-Medium | **Track**: Record
```
A1 + A2 + A3 + A6 + A11 + A14 (focal loss gamma=2)
O1 + O2 + O3 + O4
Q1 + Q3
C1
E1 + E3
```
**Rationale**: Focal loss (P2) down-weights easy tokens and focuses training on hard tokens. Cheap to implement, potentially compounds well with BigramHash (which handles easy bigram predictions).

---

### Tier B: Incremental Improvements (Combos 11-25)
*Stack new April 2026 techniques on top of SOTA foundation.*

#### Combo 11: "SOTA + QK-Gain 5.25 + SLOT"
**Target**: ~1.08 BPB | **Risk**: Medium | **Track**: Record
```
Combo 1 + A13 (QK_GAIN=5.25) + E4 (SLOT)
```
**Rationale**: Two highest-impact new techniques stacked on existing SOTA. SLOT alone reportedly gives -0.018 to -0.028 BPB. QK-Gain has been validated in multiple April PRs.

---

#### Combo 12: "SOTA + Hadamard Rotation + AWQ"
**Target**: ~1.10 BPB | **Risk**: Medium | **Track**: Record
```
Combo 1 + Q6 (Hadamard) + Q7 (AWQ)
Replace Q3 (GPTQ) with Q7 (AWQ)
```
**Rationale**: PolarQuant (ICLR 2026) showed Hadamard rotation before quantization reduces outliers. AWQ may be more robust than GPTQ when calibration distribution differs from eval.

---

#### Combo 13: "SOTA + Layer-wise Mixed Precision"
**Target**: ~1.10 BPB | **Risk**: Medium | **Track**: Record
```
Combo 1 + Q8 (layer-wise precision)
Attention layers: int6, MLP layers: int5, Embeddings: fp16
```
**Rationale**: Different layers have different sensitivity to quantization. Allocate precision where it matters most.

---

#### Combo 14: "SOTA + Brotli Compression + Wider BigramHash"
**Target**: ~1.10 BPB | **Risk**: Low-Medium | **Track**: Record
```
Combo 1 + C3 (Brotli-11 + byte-shuffle) + A12 (BigramHash 4096x128)
```
**Rationale**: Brotli-11 saves ~580KB over LZMA. Use those freed bytes for wider BigramHash. Net effect: more capacity at same artifact size.

---

#### Combo 15: "SOTA + SLOT + Stride-16 Eval"
**Target**: ~1.07 BPB | **Risk**: Medium | **Track**: Record
```
Combo 1 + E4 (SLOT) + E2 (stride=16) + E3 (T=0.90)
```
**Rationale**: Triple-stack eval improvements. SLOT (-0.020) + stride-16 (-0.006 over stride-64) + temperature scaling (-0.002). These are independent gains that should compound.

---

#### Combo 16: "SOTA + Noisy QAT + Higher Quant"
**Target**: ~1.10 BPB | **Risk**: Medium | **Track**: Record
```
Combo 1 + Q9 (Noisy QAT calibrated to int6) + Q1 (int6 STE)
Replace Q3 with Q4 (GPTQ-lite, simpler)
```
**Rationale**: Noisy QAT from the depth recurrence paper collapsed quant gaps by 185x. Apply to flat architecture with noise calibrated to int6 step size.

---

#### Combo 17: "SOTA + Turbo-Muon"
**Target**: ~1.10 BPB | **Risk**: Medium-High | **Track**: Record
```
Combo 1 + O6 (Turbo-Muon)
```
**Rationale**: Turbo-Muon (AOL preconditioning + Polar Express coefficients) reportedly trains faster than standard Muon. More steps in 600s = lower BPB.

---

#### Combo 18: "SOTA + LaCT (Large Chunk TTT)"
**Target**: ~1.08 BPB | **Risk**: Medium-High | **Track**: Record
```
Combo 1 + E6 (LaCT)
```
**Rationale**: LaCT is an ICLR 2026 Oral paper. Uses document-sized chunks for TTT achieving 70% GPU utilization. Should be more efficient than LoRA TTT.

---

#### Combo 19: "SOTA + MP-SGD TTT (4-Phase)"
**Target**: ~1.07 BPB | **Risk**: Medium | **Track**: Record
```
Combo 1 + E7 (MP-SGD 4-phase TTT)
```
**Rationale**: Multi-phase SGD TTT with progressive learning rates. April PRs show ~1.072 BPB with this approach.

---

#### Combo 20: "SOTA + Focal Loss + QK-Gain + EMA-Heavy"
**Target**: ~1.09 BPB | **Risk**: Low-Medium | **Track**: Record
```
Combo 1 + A13 (QK_GAIN=4.0) + A14 (focal loss) + O4 (EMA=0.998)
```
**Rationale**: Three independent low-risk improvements. Expected compound gain: -0.008 to -0.012 BPB.

---

#### Combo 21: "SOTA + All April Low-Hanging Fruit"
**Target**: ~1.08 BPB | **Risk**: Medium | **Track**: Record
```
Combo 1 + A13 (QK_GAIN=4.0) + E2 (stride=16) + E3 (T=0.90) + C3 (Brotli) + Q6 (Hadamard)
```
**Rationale**: Stack every new low-risk technique from April PRs. None of these conflict with each other.

---

#### Combo 22: "SOTA + Liger-Kernel Systems Opt"
**Target**: ~1.10 BPB | **Risk**: Medium | **Track**: Record
```
Combo 1 + S2 (Liger-Kernel fused cross-entropy + MLP)
```
**Rationale**: Pure systems optimization. Liger-Kernel fuses the cross-entropy computation and MLP operations, gaining 10-15% throughput. More steps = lower BPB. No ML change.

---

#### Combo 23: "SOTA + Progressive Window Warmup"
**Target**: ~1.10 BPB | **Risk**: Low | **Track**: Record
```
Combo 1 + O7 (progressive window warmup)
Start training with seq_len=512, ramp to 1024 at step 500
```
**Rationale**: Train on shorter sequences early (faster steps, more iterations) then increase to full context. Net more tokens seen in 600s.

---

#### Combo 24: "SOTA + Anti-Layer Removal"
**Target**: ~1.10 BPB | **Risk**: Low | **Track**: Record
```
Combo 1 + A15 (anti-layer removal)
After training, identify and remove layers that increase eval loss. Retrain remaining.
```
**Rationale**: Some layers may be counterproductive after quantization. Removing them frees artifact bytes for higher precision on remaining layers.

---

#### Combo 25: "All Tier 1+2 Techniques Combined"
**Target**: ~1.08 BPB | **Risk**: Medium | **Track**: Record
```
A1+A2+A3+A4+A5+A6+A8+A9+A10+A11+A13+A14
O1+O2+O3+O4+O5+O6
Q1+Q2+Q3+Q5+Q6+Q8
C3
E2+E3+E4
```
**Rationale**: Everything including new April techniques. Maximum possible stacking. Risk: some techniques may conflict.

---

### Tier C: Novel Approaches (Combos 26-40)
*Differentiation strategies. These are what OpenAI wants to see.*

#### Combo 26: "SP8192 Tokenizer Path"
**Target**: ~1.08 BPB | **Risk**: High | **Track**: Record
```
T1 (SP8192 vocab) + A1 + A2 + A3 + A6
O1 + O3 + O4
Q3
E1 + E4 (SLOT)
```
**Rationale**: Larger vocabulary captures more subword patterns. April PRs show multiple 1.07-1.08 BPB results with SP8192. Requires re-downloading dataset with `--variant sp8192`.

---

#### Combo 27: "SP8192 + Hadamard + SLOT"
**Target**: ~1.06 BPB | **Risk**: High | **Track**: Record
```
T1 (SP8192) + A1 + A2 + A6 + A13 (QK_GAIN=5.25)
O1 + O3 + O4
Q3 + Q6 (Hadamard)
C3
E2 + E3 + E4 (SLOT)
```
**Rationale**: Matches the architecture of the best April pending PRs. SP8192 + QK-Gain + SLOT is the current frontier.

---

#### Combo 28: "Scylla Tokenizer (TokenMonster 998)"
**Target**: ~0.95-1.00 BPB | **Risk**: Very High | **Track**: Record/Non-Record
```
T2 (Scylla/TokenMonster 998-vocab) + A1 + A2 + A6
O1 + O3 + O4
Q3
E1 + E4 (SLOT)
```
**Rationale**: TokenMonster's ungreedy tokenization produces ~37.5% fewer tokens than BPE. The 0.95 BPB submission used this. Requires building a custom tokenizer + dataset.

---

#### Combo 29: "CaseOps Tokenizer + Tapered WD"
**Target**: ~1.07 BPB | **Risk**: High | **Track**: Record
```
T3 (CaseOps) + A1 + A2 + A6
O1 + O3 (tapered WD) + O4
Q3
C3
E1 + E4
```
**Rationale**: CaseOps tokenizer encodes capitalization as marker tokens (like capcode). PR #1729 achieved 1.0678 BPB with this approach.

---

#### Combo 30: "N-gram Cache Hybrid (Orders 2-5)"
**Target**: ~0.90-1.00 BPB | **Risk**: Very High | **Track**: Non-Record (likely)
```
H1 (n-gram cache orders 2-5) + Combo 1
During eval: interpolate neural LM probs with n-gram backoff
```
**Rationale**: N-gram caches built from already-evaluated tokens are legal. Combining a neural model with classical n-gram statistics is the highest-performing approach in pending PRs. **Caution**: May be ruled against challenge spirit.

---

#### Combo 31: "Complementary Training + N-gram"
**Target**: ~0.85-0.95 BPB | **Risk**: Very High | **Track**: Non-Record (likely)
```
H2 (complementary training) + H1 (n-gram cache)
Train neural model with lower loss weight on n-gram-predictable tokens
```
**Rationale**: The neural model specializes on what n-grams can't predict, while n-grams handle the easy patterns. At eval time, mix both. Two of the top pending PRs use this.

---

#### Combo 32: "SSM (State-Space Model) Hybrid"
**Target**: Unknown | **Risk**: Moonshot | **Track**: Non-Record
```
Replace transformer blocks with Mamba-style SSM blocks
+ A11 (BigramHash) + Q1 (int6)
O3 + O4
E1
```
**Rationale**: SSMs are on OpenAI's explicit wishlist. Nobody has tried it yet. Even a negative result will be accepted as non-record. High visibility.

---

#### Combo 33: "Text Diffusion LM"
**Target**: Unknown | **Risk**: Moonshot | **Track**: Non-Record
```
Masked diffusion language model within 16MB
Discrete diffusion with absorbing state
```
**Rationale**: On OpenAI's explicit wishlist. Novel architecture that nobody has submitted. Even an uncompetitive result would stand out. Reference: MDLM (Sahoo et al., 2024).

---

#### Combo 34: "JEPA for Language"
**Target**: Unknown | **Risk**: Moonshot | **Track**: Non-Record
```
Joint Embedding Predictive Architecture adapted for text
Predict latent representations instead of tokens
```
**Rationale**: On OpenAI's explicit wishlist and nobody has attempted it. Would get immediate OpenAI researcher attention, even with poor BPB.

---

#### Combo 35: "Universal Transformer (Weight-Sharing)"
**Target**: ~1.18 BPB | **Risk**: High | **Track**: Non-Record
```
3 unique blocks x 4 repeats = 12 effective depth
+ Q9 (Noisy QAT) + A11 (BigramHash)
O1 + O3 + O4
E1
```
**Rationale**: On OpenAI's wishlist. The depth recurrence paper (PR #363) showed 0.025 BPB penalty vs flat. But with Noisy QAT and better config (3x3 > 2x5), the gap might shrink. Submit as non-record with the Noisy QAT finding highlighted.

---

#### Combo 36: "Megakernel Attention + MLP"
**Target**: ~1.10 BPB (via speed) | **Risk**: Very High | **Track**: Record
```
Custom Triton kernel fusing attention + MLP into single kernel
+ Combo 1 base
```
**Rationale**: On OpenAI's wishlist. A fused attention+MLP megakernel could double throughput, enabling 14,000 steps in 600s instead of 7,000. Pure systems optimization with massive BPB impact.

---

#### Combo 37: "H-net Tokenization"
**Target**: Unknown | **Risk**: Moonshot | **Track**: Non-Record
```
Hierarchical subword tokenization with multi-level embeddings
Character → subword → word hierarchy
```
**Rationale**: On OpenAI's wishlist. Unexplored. Would demonstrate creativity.

---

#### Combo 38: "Byte-Level Model (No Tokenizer)"
**Target**: ~1.20 BPB | **Risk**: High | **Track**: Non-Record
```
VOCAB_SIZE=256 (raw bytes), deeper model (15L, narrow 384d)
+ MegaByte-style local/global hierarchy
All ternary quantization for max params
```
**Rationale**: Eliminates tokenizer entirely. BPB is tokenizer-agnostic, so a byte-level model is directly comparable. Novel approach nobody has tried.

---

#### Combo 39: "Mixture-of-Experts (MoE) Miniature"
**Target**: ~1.15 BPB | **Risk**: High | **Track**: Non-Record
```
2 experts per MLP, top-1 routing
9 layers, 512d, int6 quantization
+ A11 (BigramHash) + E1 (sliding eval)
```
**Rationale**: MoE has been reported as -0.06 to -0.08 BPB penalty at this scale, but nobody has tried it with modern quantization. The April analysis says "dead architecture" but that conclusion was from March.

---

#### Combo 40: "Learning Adapters on Random Linear Maps"
**Target**: Unknown | **Risk**: Moonshot | **Track**: Non-Record
```
Fix random linear projections (frozen)
Only train small adapter matrices between them
Ultra-low parameter count, extreme compression
```
**Rationale**: On OpenAI's explicit wishlist. Completely unexplored. A working prototype would generate significant interest.

---

### Tier D: Maximum Ambition Stacks (Combos 41-50)
*The "go big or go home" combinations. High risk, high reward.*

#### Combo 41: "Kitchen Sink Record Attempt"
**Target**: ~1.05 BPB | **Risk**: High | **Track**: Record
```
T1 (SP8192) + A1 + A2 + A3 + A4 + A5 + A6 + A8 + A9 + A10 + A13 (QK=5.25)
O1 + O2 + O3 (WD=4000) + O4 (EMA=0.998) + O5 + O6
Q1 + Q2 + Q3 + Q5 + Q6 + Q8
C3
E2 (stride=16) + E3 (T=0.90) + E4 (SLOT) + E7 (MP-SGD TTT)
S1 + S2
```
**Rationale**: Every known technique combined. The risk is interactions between techniques, but the ceiling is very high.

---

#### Combo 42: "Best New Stack (April Meta)"
**Target**: ~1.06 BPB | **Risk**: Medium-High | **Track**: Record
```
T1 (SP8192) + A1 + A2 + A3 + A6 + A11 + A13 (QK=4.0)
O1 + O2 + O3 + O4
Q3 + Q6 (Hadamard)
C3
E2 + E3 + E4 (SLOT)
S1
```
**Rationale**: The current meta from April PRs. SP8192 + QK-Gain + SLOT + Hadamard is what the top pending submissions use.

---

#### Combo 43: "TokenMonster + SLOT + N-gram"
**Target**: ~0.80-0.90 BPB | **Risk**: Very High | **Track**: Record (if legal)
```
T2 (TokenMonster 998) + Combo 1 neural base
H1 (n-gram cache orders 2-5)
E4 (SLOT)
```
**Rationale**: Combines the three highest-impact innovations: better tokenizer, n-gram cache, and SLOT. If all three are ruled legal, this is the path to sub-0.9.

---

#### Combo 44: "Ternary + Extended Training" (Unlimited Track)
**Target**: ~1.05 BPB | **Risk**: Medium | **Track**: Non-Record (unlimited)
```
Ternary quantization (1.6 bits/param) + 100M params
15L, 768d, 4x MLP
50,000 steps (~2+ hours)
+ SmearGate + YaRN
```
**Rationale**: The binary submission showed 1.1239 with 2 hours. With better training schedule and newer techniques, sub-1.10 should be reachable. Demonstrates compression frontier.

---

#### Combo 45: "E2E TTT with Progressive Adaptation"
**Target**: ~1.05 BPB | **Risk**: High | **Track**: Record
```
Combo 1 base + End-to-end TTT
Phase 1: Standard sliding eval (score all tokens)
Phase 2: Full-weight SGD adaptation on scored tokens (3 epochs)
Phase 3: Re-score with adapted model
Progressive LR decay across phases
```
**Rationale**: More aggressive than LoRA TTT. The full-weight TTT failed in PR #363 because of AdamW overfitting. Using SGD with momentum and careful LR decay should work better.

---

#### Combo 46: "Sparse Attention + Dense MLP"
**Target**: ~1.10 BPB | **Risk**: High | **Track**: Record
```
A1 (11L) + local attention (window=256) on 8 layers + full attention on 3 layers
A2 (MLP_MULT=4 on dense attention layers, 2 on sparse)
Q1 + Q3
E1
```
**Rationale**: Sparse attention is faster, allowing more steps. Dense MLP compensates for reduced attention capacity. Novel architecture.

---

#### Combo 47: "Multi-Token Prediction + SLOT"
**Target**: ~1.08 BPB | **Risk**: Medium-High | **Track**: Record
```
Combo 1 + MTP (4 heads, weight=0.3)
E4 (SLOT)
```
**Rationale**: MTP was "neutral to negative" on the GPTQ stack, but it was never tested with SLOT. The auxiliary prediction heads may improve the representations that SLOT adapts.

---

#### Combo 48: "2:4 Structured Sparsity + Higher Precision"
**Target**: ~1.10 BPB | **Risk**: High | **Track**: Record
```
Combo 1 + V:N:M (2:4) activation sparsity
Use freed compute for int8 instead of int6 (higher precision)
More steps per second
```
**Rationale**: 2:4 sparsity gives 1.4-1.7x speedup on H100 Tensor Cores. Trade sparsity penalty for higher quantization precision and more steps.

---

#### Combo 49: "Knowledge Distillation from Large Run"
**Target**: ~1.08 BPB | **Risk**: High | **Track**: Record
```
Phase 1: Train large model for 4 hours (unlimited track)
Phase 2: Distill into 16MB model in 10 minutes
Student: Combo 1 architecture
Teacher: 44 (ternary extended)
```
**Rationale**: Train a powerful teacher on unlimited compute, then distill into the competition model. The April analysis said distillation failed (~11ms overhead), but that was online distillation. Offline distillation (pre-computed soft targets) avoids the latency.

---

#### Combo 50: "The Sanath Stack" (Your Signature)
**Target**: ~1.06-1.08 BPB | **Risk**: Medium-High | **Track**: Record
```
SP8192 tokenizer
11L, 512d, MLP 3x, LeakyReLU(0.5)^2
XSA all layers + BigramHash 3072x112
QK-Gain 4.0 + Partial RoPE 16/64 + LN Scale
EMA(0.998) + SWA(30) + Warmdown 4000
Full Hessian GPTQ (AR self-gen) + Hadamard pre-rotation
Brotli-11 + byte-shuffle compression
Focal loss (P2, gamma=2) during training
Sliding eval stride=16 + T=0.90 + SLOT
Parallel Muon + Liger-Kernel
```
**Rationale**: Your best-shot combination. Takes the proven SOTA stack, adds the four most promising April innovations (SP8192, SLOT, QK-Gain, Hadamard), and includes two easy wins (focal loss, Brotli). Named for your submission.

---

## Recommended Top 10 to Experiment With

Based on risk/reward analysis, here are the 10 combos to test:

| Priority | Combo | Why | M4 Testable? |
|----------|-------|-----|--------------|
| 1 | **#2** SOTA Minus TTT/GPTQ | Validate your pipeline | Yes |
| 2 | **#7** High QK-Gain | Easiest new technique, highest confidence | Yes |
| 3 | **#10** Focal Loss | Independent of architecture, easy to implement | Yes |
| 4 | **#9** Stride-16 Eval | Pure eval improvement, no training change | Yes |
| 5 | **#6** Warmdown 5000 | One config change, free experiment | Yes |
| 6 | **#50** "The Sanath Stack" | Your submission stack | Partial |
| 7 | **#27** SP8192 + SLOT | Current competition meta | No (need H100) |
| 8 | **#32** SSM Hybrid | OpenAI wishlist, unique | Yes (research) |
| 9 | **#34** JEPA | Maximum OpenAI attention | Yes (research) |
| 10 | **#30** N-gram Cache | Highest absolute BPB potential | Yes (partially) |

---

## Implementation Priority for M4 Testing

### Round 1: Quick Config Changes (30 min each)
Test relative impact by changing config.py values:

```bash
# Combo 7: QK-Gain
QK_GAIN_INIT=4.0 NUM_LAYERS=11 MLP_MULT=3 python3 train_gpt_mlx.py

# Combo 6: Warmdown
WARMDOWN_ITERS=5000 NUM_LAYERS=11 MLP_MULT=3 python3 train_gpt_mlx.py

# Combo 2: Full foundation
NUM_LAYERS=11 MLP_MULT=3 MUON_MOMENTUM=0.99 WARMDOWN_ITERS=3500 python3 train_gpt_mlx.py
```

### Round 2: Code Changes (2-4 hours each)
- Implement focal loss in `train_gpt_mlx.py`
- Implement stride-16 sliding window eval
- Implement BigramHash (port from SOTA train_gpt.py)

### Round 3: H100 Deployment
- SP8192 dataset download + tokenizer setup
- Full GPTQ implementation
- SLOT evaluation implementation
- Full "Sanath Stack" (#50) run

---

## What Gets You Noticed by OpenAI

OpenAI explicitly said they're looking for:

1. **Creative problem-solving** - Not just stacking known techniques
2. **Rigorous methodology** - Statistical significance, proper ablations
3. **Open knowledge sharing** - Detailed READMEs explaining what worked AND what didn't
4. **Novel architectures** - JEPA, SSM, text diffusion (even if they don't win)
5. **Technical depth** - Understanding *why* techniques work, not just *that* they work

### The Two Paths to Recognition

**Path A: Beat SOTA (Record Track)**
- Stack known techniques efficiently
- Focus on #50 "The Sanath Stack" or #42 "Best New Stack"
- Requires 8xH100, 3 seeds, statistical significance
- Crowded field, many strong competitors

**Path B: Do Something Nobody Else Has (Non-Record Track)**
- Implement JEPA (#34), SSM (#32), or text diffusion (#33)
- Even poor BPB scores will be accepted
- Write an excellent README documenting your journey and findings
- Much less competition, more attention from OpenAI researchers
- The depth recurrence paper (PR #363) by a high school waitress got massive community attention despite not beating SOTA

**Recommended**: Do both. Submit a competitive record attempt AND a novel non-record submission. The novel submission is what gets you recognized; the record attempt shows technical chops.

---

## Sources

- [OpenAI Parameter Golf Repository](https://github.com/openai/parameter-golf)
- [Parameter Golf Leaderboard](https://parameter-golf.github.io/)
- [DeepWiki: Parameter Golf Analysis](https://deepwiki.com/openai/parameter-golf)
- [Issue #140: Live Commentary & Analysis](https://github.com/openai/parameter-golf/issues/140)
- [PolarQuant: Hadamard Rotation for Quantization](https://arxiv.org/html/2603.29078v1)
- [ParoQuant: Pairwise Rotation Quantization (ICLR 2026)](https://arxiv.org/pdf/2511.10645)
- [TokenMonster Repository](https://github.com/alasdairforsythe/tokenmonster)
- [Liger-Kernel: Efficient Triton Kernels](https://github.com/linkedin/Liger-Kernel)
- [PR #363: Depth Recurrence Research](records/track_non_record_16mb/2026-03-21_DepthRecurrence_MixedPrecisionQuant/README.md)
- [PR #1019: Current SOTA](records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/README.md)
- [PR #549: TTT SOTA](records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md)
- [OpenAI Parameter Golf Blog Post](https://openai.com/index/parameter-golf/)
- [RunPod Partnership Blog](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod)
