# Experiment Log: Phase 2

## Master Summary Table

All results are `final_int8_zlib_roundtrip_exact val_bpb` unless otherwise noted. Competition SOTA: **1.1147**.

| ID | Name | val_bpb (5090 1GPU) | val_bpb (5090 4GPU) | val_bpb (H100 8GPU) | val_bpb (H100 Benchmark) | Delta vs R1-5 H100 |
|----|------|---------------------|---------------------|---------------------|--------------------------|---------------------|
| B | Baseline (relu², 9L, 2x) | — | — | 1.23719691 | — | — |
| P1-leaky | LeakyReLU(0.5)² (9L, 2x) | — | — | 1.23333344 | — | — |
| P1-leaky-sc20 | LeakyReLU(0.5)² + softcap=20 (9L, 2x) | — | — | 1.24057110 | — | — |
| P1-sc20 | softcap=20 (9L, 2x) | — | — | 1.24259779 | — | — |
| P1-silu | SiLU (9L, 2x) | — | — | 1.25086983 | — | — |
| P1-gelu | GELU (9L, 2x) | — | — | 1.25415677 | — | — |
| P1-sin2 | sin² (9L, 2x) | — | — | 1.29369816 | — | — |
| P1-sin2-sc20 | sin² + softcap=20 (9L, 2x) | — | — | 1.29714690 | — | — |
| R1-1 | LeakyReLU(0.5)² + 11L + 3x MLP | 1.39162128 | 1.3440* | 1.41537824 | — | +0.02224 |
| R1-2 | R1-1 + BigramHash 3072 | 1.39091596 | — | 1.41102116 | — | +0.01789 |
| R1-3 | R1-2 + XSA (last 4 layers) | 1.38500806 | — | 1.40922714 | — | +0.01609 |
| R1-4 | U-Net skip (already in baseline) | N/A | N/A | N/A | — | N/A |
| R1-5 | R1-3 + Value Residual | CRASHED | 1.33003108 | 1.39313418 | 1.39626259 | baseline |
| R2-1 | FAN Periodic MLP | — | — | 1.36291392 | — | -0.03022 |
| R2-2 | DML-Gated MLP + Barlow Twins | — | — | 1.35047954 | 1.35238667 | -0.04265 |
| R2-3 | DML Gram-Schmidt MLP | — | — | 1.37531143 | — | -0.01782 |
| R2-4 | FAN + DML-Gated combo | — | — | 1.36213200 | — | -0.03100 |
| R2-5 | Token dropout (10%) | — | — | 1.30447668 | — | -0.08866 |
| R2-8 | Graduated token dropout (20%->0%) | — | — | 1.31214176 | — | -0.08099 |
| R2-11 | Corrupted context (10%) | — | — | 1.30041527 | 1.30085969 | -0.09272 |
| R2-12 | Graduated corruption (0%->20%->0%) | — | — | 1.30306634 | — | -0.09007 |
| R2-13 | CausalWide MLP (8L x 5x, 3-bank) | — | — | 1.37949957 | — | -0.01363 |
| R2-14 | DML-CausalWide (8L x 5x, nested) | — | — | 1.37039319 | — | -0.02274 |
| R2-15 | Adaptive Causal Probing | — | — | — | — | not impl. |
| R3-slide | Sliding window eval (stride=64) | — | — | 1.39480979 | — | +0.00168 |
| R3-ttt | Legal Score-First TTT | — | — | 1.29833254 | — | -0.09480 |
| BM-sota1 | SOTA #1 repro (GPTQ int6 + XSA + slide) | — | — | — | 1.33148292 | -0.06165 |
| BM-sota2 | SOTA #2 repro (LeakyReLU + TTT) | — | — | — | 1.34714950 | -0.04598 |
| BM-ours-corrupt | Our best (R2-11 config) | — | — | — | 1.30085969 | -0.09227 |
| BM-ours-dml-corrupt | DML-Gated + Corrupt combo | — | — | — | 1.31003806 | -0.08309 |
| BM-ours-dml-gated | DML-Gated (benchmark rerun) | — | — | — | 1.35238667 | -0.04074 |
| BM-ours-r1-baseline | R1-5 baseline (benchmark rerun) | — | — | — | 1.39626259 | +0.00313 |

\* R1-1 5090 4GPU value (1.3440) from prior run, no result file preserved in `results/`.

**Note**: Phase 1 experiments (B, P1-*) use the original 9L/2x baseline architecture (~17M params). All R1+ experiments use 11L/3x (~27M params). These are not directly comparable due to different model sizes and thus different training step counts within the 10-minute window.

---

## Phase 1: Activation Screen (8xH100, 10 min, 1 seed)

Baseline architecture: 9 layers, 2x MLP (512d), 1024 vocab, tied embeddings.
All experiments use int8+zlib roundtrip quantization.

| Experiment | val_loss | val_bpb | Delta vs baseline |
|-----------|----------|---------|-------------------|
| LeakyReLU(0.5)² | 2.08243184 | 1.23333344 | -0.00386 |
| Baseline (relu²) | 2.08895515 | 1.23719691 | — |
| LeakyReLU(0.5)² + softcap=20 | 2.09465234 | 1.24057110 | +0.00337 |
| softcap=20 | 2.09807433 | 1.24259779 | +0.00540 |
| SiLU | 2.11204132 | 1.25086983 | +0.01367 |
| GELU | 2.11759118 | 1.25415677 | +0.01696 |
| sin² | 2.18435514 | 1.29369816 | +0.05650 |
| sin² + softcap=20 | 2.19017821 | 1.29714690 | +0.05995 |

**Takeaway**: LeakyReLU(0.5)² is the best activation (-0.004 bpb). sin² is catastrophic (+0.057). Softcap=20 hurts in all combinations. SiLU and GELU are both worse than relu².

Logs: `results/h100_matrix_20260405/`

---

## Round 1: Proven Technique Stack

All R1 experiments build cumulatively on 11L/3x architecture with LeakyReLU(0.5)².

### R1-1: LeakyReLU(0.5)² + 11 Layers + 3x MLP

**Idea**: Stack three proven techniques from the current SOTA submissions:
1. **LeakyReLU(0.5)²** -- preserves negative gradient flow by scaling negative inputs to 0.5x before squaring, eliminating dead neurons. Ablated at -0.002 to -0.003 bpb in SOTA submissions [PR #549, PR #414 on openai/parameter-golf].
2. **11 layers** (from 9) -- deeper model, standard in all top-3 SOTA submissions.
3. **3x MLP** (from 2x) -- wider MLP (hidden dim 1536 vs 1024), standard in top-3 SOTA.

**References**:
- LeakyReLU²: First used in [modded-nanogpt PR #414](https://github.com/KellerJordan/modded-nanogpt), adopted by parameter-golf SOTA submissions [PR #549](https://github.com/openai/parameter-golf/pull/549), [PR #1019](https://github.com/openai/parameter-golf/pull/1019).
- 11L + 3x MLP: Standard config in all top-3 leaderboard entries (val_bpb 1.1147-1.1570).

**Changes**: 1-line activation swap + env var overrides.

**Output**:
- RTX 5090 (1 GPU): val_loss=2.34969424, **val_bpb=1.39162128**
- RTX 5090 (4 GPU): val_bpb=1.3440 (from prior run, no file preserved)
- 8xH100: val_loss=2.38980686, **val_bpb=1.41537824**

---

### R1-2: BigramHash 3072 Embedding

**Idea**: Add token co-occurrence signal at the embedding layer via a hash-based bigram table. Instead of a full vocab² bigram matrix (prohibitively expensive), hash each consecutive token pair into a fixed-size embedding table using polynomial XOR hashing. This gives the model access to local context patterns (which token pairs tend to co-occur) at the embedding level, before any attention computation.

**References**:
- BigramHash embedding: Introduced in parameter-golf [PR #414](https://github.com/openai/parameter-golf/pull/414). Extended with trigram hash in [PR #1019](https://github.com/openai/parameter-golf/pull/1019). Present in all top-2 SOTA submissions.
- Hash function design: Polynomial multiply-XOR with coprime constants (36313, 27191) for collision minimization. No published derivation -- empirically chosen.

**Key design choices**:
- Additive to token embedding (not concatenated) -- no dimension increase
- Zero-initialized with learnable scale (0.05) -- starts contributing nothing, learns to contribute
- Separate embedding dim (128) with projection to model dim (512)
- 3072 hash buckets with modulus 3071 (last bucket reserved as sentinel for position 0)

**Changes**: New BigramHashEmbedding class (~50 lines), added to forward before RMS norm.

**Output**:
- RTX 5090 (1 GPU): val_loss=2.34850334, **val_bpb=1.39091596**
- 8xH100: val_loss=2.38245012, **val_bpb=1.41102116**

---

### R1-3: XSA (Cross-Sequence Attention)

**Idea**: After standard attention computes the output, remove the component parallel to the current token's own value vector using Gram-Schmidt orthogonalization. The intuition: the attention output already "knows" the token's own value via the direct V path; by projecting it out, the residual output carries only information gathered from *other* tokens, reducing redundancy.

**References**:
- XSA: Introduced in parameter-golf [PR #198](https://github.com/openai/parameter-golf/pull/198) as "Efficient Partial XSA". Extended to all layers in [PR #1019](https://github.com/openai/parameter-golf/pull/1019). Ablated at -0.003 bpb.
- Mathematically equivalent to a per-head Gram-Schmidt orthogonalization step, applied after the softmax-weighted value aggregation.

**Key design choices**:
- GQA-aware: reshapes `[B,T,H,D]` -> `[B,T,Hkv,group,D]` to broadcast normalized V across query head groups without repeat_interleave
- Zero extra parameters
- Applied only on last N decoder layers (default N=4) -- early layers benefit less

**Changes**: New `_xsa_efficient()` method on attention class (~15 lines). Note: requires attention output in `[B,T,H,D]` format (baseline uses `[B,H,T,D]` -- may need layout adjustment).

**Output**:
- RTX 5090 (1 GPU): val_loss=2.33852809, **val_bpb=1.38500806**
- 8xH100: val_loss=2.37942100, **val_bpb=1.40922714**

---

### R1-4: U-Net Skip Connections

**Idea**: Split the transformer into encoder (first half) and decoder (second half) with learnable skip connections between them, following the U-Net topology. Skip connections allow gradients to flow directly from decoder to early encoder layers, and allow the decoder to access lower-level representations. The LIFO stack creates symmetric connections: encoder layer 0 <-> last decoder layer.

**References**:
- U-Net for transformers: Used in all top-3 parameter-golf SOTA submissions. The pattern originated in the [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt) where it was shown to improve convergence without additional parameters.
- U-Net architecture: Originally Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015).

**Key design choices**:
- 11 layers -> 5 encoder + 6 decoder (integer division)
- 5 skip connections (min of encoder, decoder counts)
- Per-channel learned scale vector `[512]` per skip, initialized to ones
- Additive: `x = x + skip_weight * skip_activation`

**Changes**: Split forward loop into encoder/decoder, add skip stack (~30 lines).

**Output**: N/A -- U-Net was already present in the baseline. No separate ablation needed.

---

### R1-5: Value Residual Propagation

**Idea**: Propagate the first layer's raw value vectors to all subsequent layers via a learnable sigmoid gate. This gives deeper layers direct access to the initial token-level value representation before it has been mixed by attention in intermediate layers. The gate starts at 50% mixing (sigmoid(0)=0.5), allowing training to find the optimal blend.

**References**:
- Value residual: Present in both top-2 SOTA parameter-golf submissions. SOTA #1 uses sigmoid gate (`v + sigmoid(a)*v0`), SOTA #2 uses unconstrained lambda mix (`l0*v0 + l1*v`). Both originated from [PR #414](https://github.com/openai/parameter-golf/pull/414).
- Related concept: Value embeddings (Zhou et al. 2024), used in the [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt), add a second embedding table to V. Value residual is the dynamic version -- it propagates actual V activations rather than a fixed embedding.

**Key design choices**:
- Sigmoid gate (SOTA #1 variant) -- simpler, one parameter per layer
- v0 captured from layer 0 after value projection, before RoPE
- v0 is frozen after capture (not updated during decoder pass)
- Shape: `[B, T, 4, 64]` (per KV head)

**Changes**: Store v0 from first layer, add vrl_alpha parameter to attention, mix in forward (~15 lines).

**Output**:
- RTX 5090 (1 GPU): CRASHED (exit code 1, likely OOM with 11L/3x + value residual on single GPU)
- RTX 5090 (4 GPU): val_loss=2.24570176, **val_bpb=1.33003108**
- 8xH100 (full screen): val_loss=2.35224871, **val_bpb=1.39313418**
- 8xH100 (benchmark rerun): val_loss=2.35753090, **val_bpb=1.39626259**

---

## Round 2: Novel Designs

All R2 experiments build on the R1-5 baseline (11L/3x + LeakyReLU² + BigramHash + XSA + Value Residual).

### R2-1: FAN Periodic MLP

**Idea**: Replace the standard MLP expansion with a Fourier Analysis Network (FAN) layer that dedicates 25% of hidden dimensions to periodic (sin/cos) features and 75% to standard LeakyReLU² features. The hypothesis: language data contains latent periodic structure (positional cycles, syntactic patterns) that purely monotonic activations cannot efficiently represent. Unlike the failed sin² experiment (R1 Phase 1), this approach ADDS periodic features alongside standard features rather than replacing them entirely.

**References**:
- Dong et al. "FAN: Fourier Analysis Networks" (arXiv 2410.02675, ICLR 2025) -- 14.65% OOD loss reduction, 25% fewer parameters.
- Yu et al. "FANformer: Improving Large Language Models Through Effective Periodicity Modeling" (arXiv 2502.21309, Feb 2025) -- 31% parameter efficiency, 20% fewer training tokens.
- Dec 2024 analysis (arXiv 2512.14873) -- only sine (not cosine) contributes; gain comes from gradient shape near zero.

**Key design choices**:
- 25% Fourier ratio: shared W_p for both sin and cos (saves one projection vs separate)
- 75% standard: LeakyReLU(0.5)² (proven best activation from Phase 1)
- Actually 12.5% fewer params than standard MLP (W_p is shared)
- Concatenation, not addition: `[cos(p) || sin(p) || leaky_relu(W_bar*x)²]`

**Changes**: New FAN_MLP class (~30 lines). Spec: `round2-specs.md#r2-1`

**Output**:
- 8xH100: val_loss=2.30122306, **val_bpb=1.36291392** (delta vs R1-5: -0.030)

---

### R2-2: DML-Gated MLP + Barlow Twins Loss

**Idea**: Design an MLP layer inspired by Double Machine Learning (Chernozhukov et al. 2018) with two parallel pathways -- a "nuisance" pathway that captures predictable structure and a "target" pathway that captures residual causal signal. Element-wise gating (like SwiGLU) combines them, while a Barlow Twins decorrelation loss explicitly enforces orthogonality between the two pathways' representations.

**References**:
- Chernozhukov et al. "Double/debiased machine learning for treatment and structural parameters" (Econometrica, 2018) -- the DML framework.
- Zbontar et al. "Barlow Twins: Self-Supervised Learning via Redundancy Reduction" (ICML 2021) -- the decorrelation loss.
- Shazeer "GLU Variants Improve Transformer" (arXiv 2002.05202, 2020) -- SwiGLU as dual-pathway prior art.
- Zhang et al. "ND-LoRA: Neural Diversity Regularizes Hallucinations in Language Models" (arXiv 2510.20690, 2025) -- Barlow Twins on parallel streams reduces hallucinations 14.6%.
- Ahn et al. "PODNN: Parallel Orthogonal Deep Neural Network" (Neural Networks, 2021) -- parallel streams + Gram-Schmidt.

**Key design choices**:
- 3-matrix structure (W_nuisance, W_target, W_out) -- budget-neutral with SwiGLU
- Nuisance pathway uses LeakyReLU(0.5)² as gating function
- Target pathway is linear (no activation in gate branch, like bilinear FFN)
- Barlow Twins loss weight l=0.01 (auxiliary, not dominant)

**Changes**: New DML_GatedMLP class + aux loss in training loop (~40 lines). Spec: `round2-specs.md#r2-2`

**Output**:
- 8xH100 (full screen): val_loss=2.28022813, **val_bpb=1.35047954** (delta vs R1-5: -0.043)
- 8xH100 (benchmark rerun): val_loss=2.28344824, **val_bpb=1.35238667**

---

### R2-3: DML with Gram-Schmidt Orthogonalization

**Idea**: Same dual-pathway concept as R2-2, but instead of a training-time loss, enforce orthogonality structurally in the forward pass using Gram-Schmidt projection (identical math to XSA in attention). The target pathway's output has the nuisance component explicitly removed before the combiner. This is a stronger constraint than Barlow Twins -- it guarantees orthogonality at every forward pass, not just on average over training.

**References**: Same as R2-2, plus:
- Gram-Schmidt orthogonalization: the same technique used by XSA (R1-3) in attention, applied here to MLP pathways.

**Key design choices**:
- Asymmetric widths: nuisance pathway is narrow (384d), target is wider (768d)
- 25% fewer params than standard MLP -- could reallocate saved params
- No auxiliary loss needed -- orthogonality is structural
- Concatenation + projection combiner (not multiplicative gating)

**Changes**: New DML_OrthMLP class (~35 lines). Spec: `round2-specs.md#r2-3`

**Output**:
- 8xH100: val_loss=2.32215574, **val_bpb=1.37531143** (delta vs R1-5: -0.018)

---

### R2-4: FAN + DML-Gated Combo

**Idea**: Combine periodic FAN features (R2-1) with DML dual-pathway gating (R2-2). The value pathway produces periodic + standard features, the gate pathway controls information flow, and Barlow Twins enforces decorrelation between gate and value. This is the most architecturally novel MLP variant -- if it works, it represents a genuinely new contribution combining three ideas (periodicity, causal orthogonalization, gated mixing).

**References**: Combination of R2-1 and R2-2 references.

**Changes**: New FAN_DML_MLP class (~55 lines). Spec: `round2-specs.md#r2-4`

**Output**:
- 8xH100: val_loss=2.29990283, **val_bpb=1.36213200** (delta vs R1-5: -0.031)

---

### R2-5: Token Dropout (10%)

**Idea**: Drop random tokens from the input sequence during training as a causal intervention on the information flow. This regularizes along the *sequence* dimension (orthogonal to standard dropout which regularizes along the *feature* dimension), forcing the model to learn robust causal dependencies rather than relying on every context position being present. As a bonus, shorter sequences mean faster per-step computation -- more training steps in the 10-minute window.

**References**:
- Conceptually related to: Srivastava et al. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (JMLR, 2014) -- but applied to sequence positions, not features.
- Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" (NAACL, 2019) -- token masking for MLM, but different objective (we drop, not mask-and-predict).
- Causal intervention interpretation: Pearl "Causality" (Cambridge, 2009) -- do(remove token_i) as an intervention.

**Key design choices**:
- Shared mask across batch (same positions dropped for all sequences) -- avoids variable-length padding
- Position 0 always kept -- model needs at least one token
- RoPE positions shift (intentional -- positional augmentation)
- 10% rate: balances regularization vs information loss

**Changes**: `token_dropout()` function in training loop (~10 lines). Spec: `round2-specs.md#r2-5`

**Output**:
- 8xH100: val_loss=2.20255423, **val_bpb=1.30447668** (delta vs R1-5: -0.089)

---

### R2-7: Token Dropout + Rho-1 Selective Loss

**Idea**: Combine input-side causal selection (token dropout -- which context to learn WITH) and output-side causal selection (Rho-1 -- which tokens to learn FROM). Token dropout removes 10% of input tokens; Rho-1 skips loss on the 20% easiest-to-predict output tokens. Together, the model trains on a curated subset: robust context (via dropout) predicting hard targets (via Rho-1).

**References**:
- R2-5 references (token dropout)
- Lin et al. "Rho-1: Not All Tokens Are What You Need" (NeurIPS 2024, best paper runner-up) -- selective token loss using reference model scoring.
- Our simplified Rho-1: uses logit margin (top1 - top2) as zero-cost difficulty proxy instead of a separate reference model.

**Changes**: `selective_loss()` function + R2-5 token dropout (~25 lines). Spec: `round2-specs.md#r2-7`

**Output**: Not screened (deprioritized in favor of R2-11 corrupted context which covers similar ground).

---

### R2-8: Graduated Token Dropout (20%->0%)

**Idea**: Apply heavy token dropout early in training (when the model is learning general features and benefits from regularization), then linearly decay to zero (when the model is fine-tuning precise predictions and benefits from clean data). This follows the curriculum learning principle: "easy task first, hard task later" -- where "easy" means "with augmentation" and "hard" means "exact prediction on full context."

**References**:
- Bengio et al. "Curriculum Learning" (ICML 2009) -- progressive task difficulty.
- R2-5 references (token dropout)

**Changes**: Step-dependent drop_rate in token_dropout (~15 lines). Spec: `round2-specs.md#r2-8`

**Output**:
- 8xH100: val_loss=2.21549639, **val_bpb=1.31214176** (delta vs R1-5: -0.081)

---

### R2-9: Cross-Layer Barlow Twins

**Idea**: Force each transformer block to produce representations that are decorrelated from its neighbors, preventing redundant computation across layers. The Barlow Twins loss penalizes off-diagonal elements of the cross-correlation matrix between adjacent layers' outputs.

**References**:
- Zbontar et al. "Barlow Twins" (ICML 2021) -- cross-correlation decorrelation loss.
- Zhang et al. "ND-LoRA" (arXiv 2510.20690, 2025) -- Barlow Twins between parallel streams.

**Key design choices**:
- Adjacent pairs only (not all layer combinations) -- 10 pairs for 11 layers
- Subsample features (128 of 512 dims) to reduce cost from O(512²) to O(128²)
- l = 0.005 (lighter than R2-2's 0.01, since this affects all layers)

**Changes**: Store layer outputs, compute pairwise BT loss in training loop (~20 lines). Spec: `round2-specs.md#r2-9`

**Output**: Not screened (deprioritized).

---

### R2-11: Corrupted Context Training (10%)

**Idea**: Bridge the train/inference gap (exposure bias) by occasionally replacing ground truth tokens with the model's own predictions during training. Standard teacher forcing always shows perfect context; at inference the model sees its own (possibly wrong) predictions. By training on corrupted context, the model learns to predict correctly even from imperfect inputs -- making it more robust at generation time.

**References**:
- Bengio et al. "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks" (NeurIPS 2015) -- the original exposure bias mitigation.
- Ranzato et al. "Sequence Level Training with Recurrent Neural Networks" (ICLR 2016) -- REINFORCE-based approach.
- Parameter-golf SOTA #1 uses AR self-generation for GPTQ calibration -- same idea applied post-training.

**Key design choices**:
- Extra forward pass (no grad) for predictions -- 30-40% compute overhead
- Replace with argmax predictions (not sampled) -- deterministic corruption
- Targets remain original ground truth -- model must correct for corrupted context
- Position 0 never corrupted

**Changes**: `corrupted_context()` function in training loop (~15 lines). Spec: `round2-specs.md#r2-11`

**Output**:
- 8xH100 (full screen): val_loss=2.19569671, **val_bpb=1.30041527** (delta vs R1-5: -0.093)
- 8xH100 (benchmark rerun): val_loss=2.19644709, **val_bpb=1.30085969**

---

### R2-12: Graduated Corruption (0%->20%->0%)

**Idea**: Apply corrupted context training on a sine schedule -- no corruption at start (learn basics from clean data), peak corruption at midpoint (learn robustness), zero corruption at end (fine-tune on clean data). The sine curve ensures smooth transitions and concentrates corruption when the model is mature enough to learn from it.

**References**: R2-11 references + curriculum learning (Bengio et al. 2009).

**Changes**: Step-dependent rate in corrupted_context (~20 lines). Spec: `round2-specs.md#r2-12`

**Output**:
- 8xH100: val_loss=2.20017294, **val_bpb=1.30306634** (delta vs R1-5: -0.090)

---

### R2-13: CausalWide MLP (Multi-Bank Orthogonal Decomposition)

**Idea**: Replace depth with structured width. Instead of 11 narrow layers (3x MLP, hidden=1536), use 8 wider layers (~4.7x MLP, hidden~2400) with the same total parameter budget (~26.5M). The wide MLP is internally structured into three orthogonal banks:

1. **Bank A ("Memory")** -- Pure linear projection, no activation. Acts as an associative lookup table.
2. **Bank B ("Feature")** -- LeakyReLU(0.5)² nonlinear features.
3. **Bank C ("Residual")** -- LeakyReLU(0.5)² features, then Gram-Schmidt orthogonalized against Banks A and B.

Barlow Twins loss enforces cross-bank decorrelation (3 pairs: A-B, A-C, B-C).

**References**:
- Width vs depth: Ternary U-Net submission uses 768d/10L -> outperforms 512d/25L.
- Gram-Schmidt orthogonalization: Same technique as XSA (R1-3), applied within MLP.
- Barlow Twins: Zbontar et al. (ICML 2021).

**Config**: `MLP_TYPE=causal_wide NUM_LAYERS=8 MLP_MULT=5` (8L x 5x ~ 27.0M)

**Changes**: New CausalWideMLP class (~75 lines). Spec: `round2-specs.md`

**Output**:
- 8xH100: val_loss=2.32922724, **val_bpb=1.37949957** (delta vs R1-5: -0.014)

---

### R2-14: DML-CausalWide MLP (Nested Causal Decomposition)

**Idea**: Two-level causal structure combining CausalWide (R2-13) and DML (R2-2). Level 1 decomposes the MLP into 3 orthogonal banks (memory, feature, residual). Level 2 applies DML dual-pathway gating (nuisance gate x target value) *within* each bank. Barlow Twins operates at both levels.

**References**:
- All R2-2 references (DML, Barlow Twins, ND-LoRA)
- All R2-13 references (CausalWide, Wide & Deep, superposition)
- Novel combination -- no published precedent for nested causal decomposition in transformer MLP layers.

**Key design choices**:
- Budget-neutral: sub_width = 2*hidden/9 per bank
- Bank A (Memory): bilinear gate (linear x linear)
- Bank B (Feature): LeakyReLU(0.5)² gate x linear value
- Bank C (Residual): Same as B, then Gram-Schmidt orthogonalized against A and B
- Two-level BT loss: 3 cross-bank pairs + 3 within-bank pairs = 6 decorrelation terms

**Config**: `MLP_TYPE=dml_causal_wide NUM_LAYERS=8 MLP_MULT=5`

**Changes**: New DML_CausalWideMLP class (~90 lines).

**Output**:
- 8xH100: val_loss=2.31385150, **val_bpb=1.37039319** (delta vs R1-5: -0.023)

---

### R2-15: Adaptive Causal Probing (PLANNED -- not yet implemented)

**Idea**: For tokens the model already predicts correctly, apply aggressive context dropout (30%) to stress-test which context is truly causally necessary. For hard tokens, keep full context. This is the inverse of Rho-1 -- instead of skipping easy tokens, make them harder.

**Key insight**: Standard token dropout (R2-5) drops context uniformly. But easy predictions don't need augmentation challenge -- they're already solved. Hard predictions need full context to learn from. Adaptive probing focuses the causal intervention where it has the most learning value.

**Recommended approach**: Single-pass variant using previous step's per-position loss as difficulty proxy. Zero compute overhead.

**Status**: Spec complete at `r2-15-adaptive-causal-probing-spec.md`. Not yet implemented. Waiting for benchmark results to prioritize.

**References**: Inverse of Rho-1 (NeurIPS 2024), Pearl's do-calculus, our R2-5 token dropout, our R2-11 corrupted context.

**Output**: Not implemented.

---

## Round 3: Quantization & Eval

### R3-1: GPTQ int6 (Full Hessian)

**Idea**: Apply GPTQ post-training quantization to compress model weights from bf16 to int6 (6 bits per weight). Full Hessian GPTQ uses the exact second-order error structure to optimally compensate quantization errors.

**References**:
- Frantar et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers" (arXiv 2210.17323, 2022).
- AR self-generation calibration: parameter-golf [PR #1019](https://github.com/openai/parameter-golf/pull/1019).

**Changes**: GPTQ quantization functions + AR calibration (~200 lines). Spec: `round3-specs.md#r3-1`

**Output**: Tested only within SOTA benchmark reproductions (see BM-sota1 below). Not applied to our own models yet.

---

### R3-slide: Sliding Window Evaluation (stride=64)

**Idea**: Evaluate with overlapping windows so every token (except the first stride positions of the first window) is scored with at least `seq_len - stride` tokens of context. Standard chunked eval gives early positions in each chunk minimal context, hurting their predictions.

**References**:
- Press et al. "Train Short, Test Long" (ICLR 2022).
- Standard practice in parameter-golf since [PR #198](https://github.com/openai/parameter-golf/pull/198).

**Changes**: Modified eval loop (~20 lines). Spec: `round3-specs.md#r3-2`

**Output**:
- 8xH100 (on R1-5 model): val_loss=2.35507790, **val_bpb=1.39480979** (delta vs R1-5 standard: **+0.002**)
- Sliding window eval **did not help** on our model. Standard eval was actually slightly better. This may be because our model was trained with seq_len=1024 and already handles chunk boundaries well, or because the R1-5 model specifically learns position-aware patterns that sliding window disrupts.

---

### R3-ttt: Legal Score-First TTT

**Idea**: Fine-tune the model on validation data chunks sequentially, but always SCORE each chunk BEFORE training on it -- maintaining legality (the model hasn't seen the data it's being graded on). As the model adapts to the validation distribution, later chunks are scored by a model that has been fine-tuned on earlier chunks.

**References**:
- Sun et al. "Test-Time Training with Self-Supervision for Generalization under Distribution Shifts" (ICML 2020).
- Legal Score-First TTT: parameter-golf [PR #549](https://github.com/openai/parameter-golf/pull/549) -- ablated at -0.0025 bpb.

**Changes**: TTT evaluation function (~150 lines). Spec: `round3-specs.md#r3-3`

**Output**:
- 8xH100 (on R1-5 model): val_loss=2.19218011, **val_bpb=1.29833254** (delta vs R1-5 standard: **-0.095**)
- TTT is the single largest improvement found in this study. The TTT-specific eval showed val_bpb=1.30082248 before int8 roundtrip; the final int8+zlib roundtrip result is 1.29833254 (slightly better, possibly due to eval noise).
- Runtime: ~6137 seconds (102 minutes) for TTT eval. This is very slow but the competition scores offline.

---

## Benchmark vs SOTA

Six experiments run on 8xH100 to compare our best configs against reproduced SOTA submissions.

| Experiment | Config | val_loss | val_bpb | Notes |
|-----------|--------|----------|---------|-------|
| BM-ours-corrupt | R1-5 + R2-11 corrupted context | 2.19644709 | **1.30085969** | Our best |
| BM-ours-dml-corrupt | R1-5 + DML-Gated + corrupt | 2.21194439 | 1.31003806 | DML + corrupt combo |
| BM-sota1 (GPTQ+XSA+slide) | SOTA #1 repro: GPTQ int6 + XSA + sliding window | 2.24814718 | 1.33148292 | int6+GPTQ+sliding eval |
| BM-sota2 (LeakyReLU+TTT) | SOTA #2 repro: LeakyReLU + TTT | 2.27459949 | 1.34714950 | int6+GPTQ+sliding eval |
| BM-ours-dml-gated | R1-5 + R2-2 DML-Gated MLP | 2.28344824 | 1.35238667 | MLP variant only |
| BM-ours-r1-baseline | R1-5 baseline (no R2/R3) | 2.35753090 | 1.39626259 | Reference point |

**Key findings from benchmark**:
1. Our corrupted context training (1.3009) beats both SOTA reproductions (1.3315, 1.3471) by a significant margin, even though SOTA uses int6 GPTQ + sliding window and we use int8+zlib.
2. DML-Gated + corruption (1.3100) also beats both SOTA reproductions.
3. DML-Gated alone (1.3524) does not beat SOTA #1 (1.3315).
4. The R1-5 baseline (1.3963) confirms that our training improvements (R2) are the main source of gain, not just architecture.

**Important caveat**: The SOTA benchmark numbers (1.3315, 1.3471) use int6 GPTQ quantization + sliding window eval, which is a different eval pipeline than our int8+zlib. The SOTA submissions' actual leaderboard scores (1.1147, ~1.15) are achieved with additional techniques not reproduced here (longer training, multi-seed ensembles, etc.).

---

## Results Analysis

### All Experiments Ranked by val_bpb (best first)

Using the best available result for each experiment (H100 8GPU preferred, then benchmark, then 5090).

| Rank | ID | Name | Best val_bpb | Hardware | Category |
|------|-----|------|-------------|----------|----------|
| 1 | B | Baseline (relu², 9L, 2x) | 1.23333344* | H100 8GPU | Phase 1 baseline |
| 2 | P1-leaky | LeakyReLU(0.5)² (9L, 2x) | 1.23719691* | H100 8GPU | Phase 1 activation |
| 3 | R3-ttt | TTT on R1-5 model | 1.29833254 | H100 8GPU | Eval trick |
| 4 | R2-11 | Corrupted context (10%) | 1.30041527 | H100 8GPU | Data augmentation |
| 5 | BM-ours-corrupt | R2-11 benchmark rerun | 1.30085969 | H100 Benchmark | Data augmentation |
| 6 | R2-12 | Graduated corruption | 1.30306634 | H100 8GPU | Data augmentation |
| 7 | R2-5 | Token dropout (10%) | 1.30447668 | H100 8GPU | Data augmentation |
| 8 | BM-ours-dml-corrupt | DML + corrupt combo | 1.31003806 | H100 Benchmark | MLP + Data aug |
| 9 | R2-8 | Graduated token dropout | 1.31214176 | H100 8GPU | Data augmentation |
| 10 | R1-5 (5090 4GPU) | Full R1 stack | 1.33003108 | 5090 4GPU | Architecture |
| 11 | BM-sota1 | SOTA #1 repro | 1.33148292 | H100 Benchmark | External baseline |
| 12 | BM-sota2 | SOTA #2 repro | 1.34714950 | H100 Benchmark | External baseline |
| 13 | R2-2 | DML-Gated MLP | 1.35047954 | H100 8GPU | MLP variant |
| 14 | BM-ours-dml-gated | DML-Gated benchmark rerun | 1.35238667 | H100 Benchmark | MLP variant |
| 15 | R2-4 | FAN + DML-Gated combo | 1.36213200 | H100 8GPU | MLP variant |
| 16 | R2-1 | FAN Periodic MLP | 1.36291392 | H100 8GPU | MLP variant |
| 17 | R2-14 | DML-CausalWide | 1.37039319 | H100 8GPU | MLP variant |
| 18 | R2-3 | DML Gram-Schmidt | 1.37531143 | H100 8GPU | MLP variant |
| 19 | R2-13 | CausalWide MLP | 1.37949957 | H100 8GPU | MLP variant |
| 20 | R1-5 | Full R1 stack (H100) | 1.39313418 | H100 8GPU | Architecture |
| 21 | R3-slide | Sliding window eval | 1.39480979 | H100 8GPU | Eval trick |
| 22 | BM-ours-r1-baseline | R1-5 benchmark rerun | 1.39626259 | H100 Benchmark | Architecture |
| 23 | R1-3 (5090) | XSA stack | 1.38500806 | 5090 1GPU | Architecture |
| 24 | R1-2 (5090) | Bigram stack | 1.39091596 | 5090 1GPU | Architecture |
| 25 | R1-1 (5090) | LeakyReLU+11L+3x | 1.39162128 | 5090 1GPU | Architecture |
| 26 | R1-3 (H100) | XSA stack | 1.40922714 | H100 8GPU | Architecture |
| 27 | R1-2 (H100) | Bigram stack | 1.41102116 | H100 8GPU | Architecture |
| 28 | R1-1 (H100) | LeakyReLU+11L+3x | 1.41537824 | H100 8GPU | Architecture |

\* Phase 1 experiments use a different architecture (9L/2x, ~17M params) and are not directly comparable to R1+ experiments (11L/3x, ~27M params). The 9L/2x model gets ~9100 training steps vs ~7000 for 11L/3x in the same 10-minute window.

### Technique Category Analysis

**What worked (on 11L/3x architecture):**

1. **Data augmentation / regularization** -- Dominant category. All four variants produced large improvements:
   - R2-11 Corrupted context: -0.093 bpb (best training technique)
   - R2-12 Graduated corruption: -0.090 bpb
   - R2-5 Token dropout: -0.089 bpb
   - R2-8 Graduated token dropout: -0.081 bpb
   - All four are within 0.012 bpb of each other. Constant-rate variants slightly outperform graduated schedules.

2. **Test-time training (eval trick)**: -0.095 bpb. The single largest individual improvement, but slow (102 min eval).

3. **MLP architecture variants** -- Modest improvements:
   - R2-2 DML-Gated: -0.043 bpb (best MLP variant)
   - R2-4 FAN+DML: -0.031 bpb
   - R2-1 FAN: -0.030 bpb
   - R2-14 DML-CausalWide: -0.023 bpb
   - R2-3 DML-Orth: -0.018 bpb
   - R2-13 CausalWide: -0.014 bpb

**What did not help:**

1. **Sliding window eval** (R3-slide): +0.002 bpb -- slightly worse than standard eval on our model.
2. **sin² activation** (Phase 1): +0.057 bpb -- catastrophic.
3. **Softcap=20**: +0.003 to +0.005 bpb in all combinations.
4. **CausalWide width-for-depth trade** (R2-13): Only -0.014 bpb despite novel architecture -- the 8L config may not have enough depth.
5. **R1-1 through R1-3 on H100**: Worse than Phase 1 baseline in absolute bpb because the 11L/3x model is under-trained in 10 min. These techniques still improve incrementally over each other.

### Recommended Config for Final Submission

**Best known config**: R1-5 + R2-11 (corrupted context 10%) + R3-ttt (test-time training)

Estimated combined val_bpb: TTT on corrupted-context model should yield approximately **~1.29 bpb** (R3-ttt gave -0.095 on R1-5 baseline; R2-11 gave -0.093 on R1-5; effects may not be fully additive).

**Additional potential improvements not yet tested:**
- GPTQ int6 quantization on our model (SOTA uses this; could save ~0.01-0.02 bpb from better compression)
- Combining DML-Gated MLP (R2-2) with corrupted context (R2-11) -- partially tested as BM-ours-dml-corrupt (1.3100), but not with TTT
- Adaptive causal probing (R2-15) -- not yet implemented
- Multi-seed ensemble averaging

### Gap Analysis vs Competition SOTA (1.1147)

| | val_bpb | Gap to SOTA |
|---|---------|-------------|
| Competition SOTA | 1.1147 | -- |
| Our best (R2-11 + int8+zlib) | 1.3004 | +0.1857 |
| Our best with TTT (R3-ttt on R1-5) | 1.2983 | +0.1836 |
| Estimated best combo (R2-11 + TTT) | ~1.29 | ~+0.175 |

The remaining gap of ~0.175 bpb is substantial. Key differences vs competition SOTA:
1. **Training efficiency**: SOTA submissions are highly optimized for H100 throughput (custom CUDA kernels, sequence packing, etc.). Our model gets ~7000 steps vs SOTA's ~10000+ in the same wallclock.
2. **Quantization**: SOTA uses int6 GPTQ (6-bit) vs our int8+zlib (8-bit). Better quantization = more params in budget = better model.
3. **Hyperparameter tuning**: SOTA submissions have hundreds of runs of tuning. We ran each config once.
4. **Architecture refinements**: SOTA submissions include many small optimizations (EMA, MTP heads, sequence length 2048, etc.) not tested here.
5. **Multi-seed ensembles**: SOTA may average across seeds.

Closing this gap requires addressing training throughput and quantization before further architectural innovation.
