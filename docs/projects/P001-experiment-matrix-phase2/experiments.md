# Experiment Log: Phase 2

## Summary

| ID | Name | Status | val_bpb (5090 1GPU) | val_bpb (5090 4GPU) | val_bpb (H100 8GPU) | Delta vs baseline |
|----|------|--------|--------------------|--------------------|--------------------|----|
| B | Baseline (relu², 9L, 2x MLP) | Done | 2.5173 | — | 1.2372 | — |
| R1-1 | LeakyReLU(0.5)² + 11L + 3x MLP | Done | 1.3916 | 1.3440 | _running_ | — |
| R1-2 | R1-1 + BigramHash 3072 | Done | 1.3909 | — | _running_ | — |
| R1-3 | R1-2 + XSA (last 4 layers) | Done | 1.3850 | — | _running_ | — |
| R1-4 | N/A (U-Net already in baseline) | N/A | — | — | — | — |
| R1-5 | R1-3 + Value residual | Done | — | 1.3300 | _running_ | — |
| R2-1 | FAN periodic MLP | Running | — | — | _running_ | — |
| R2-2 | DML-Gated MLP + Barlow Twins | Running | — | — | _running_ | — |
| R2-3 | DML Gram-Schmidt MLP | Running | — | — | _running_ | — |
| R2-4 | FAN + DML-Gated combo | Running | — | — | _running_ | — |
| R2-5 | Token dropout (10%) | Running | — | — | _running_ | — |
| R2-8 | Graduated token dropout (20%→0%) | Running | — | — | _running_ | — |
| R2-11 | Corrupted context (10%) | Running | — | — | _running_ | — |
| R2-12 | Graduated corruption (0%→20%→0%) | Running | — | — | _running_ | — |
| R2-13 | CausalWide MLP (8L×5x, 3-bank) | Running | — | — | _running_ | — |
| R2-14 | DML-CausalWide (8L×5x, nested) | Running | — | — | _running_ | — |
| R2-15 | Adaptive causal probing | Planned | — | — | — | — |
| R3-slide | Sliding window eval (stride=64) | Running | — | — | _running_ | — |
| R3-ttt | Legal Score-First TTT | Running | — | — | _running_ | — |

**Note**: All experiments now running on 8xH100 SXM (competition-standard config, grad_accum=1).
Previous 5090 results used 1 GPU (grad_accum=8) or 4 GPU (grad_accum=2) — fewer optimizer steps per wallclock.

---

## Phase 1 Results (Prior Art — Our Own)

### B: Baseline
- **Config**: relu² activation, 9 layers, 2x MLP, 512d, 1024 vocab, softcap=30
- **RTX 5090 (50 steps)**: val_bpb=2.5173, step_avg=618ms
- **8xH100 (10 min)**: val_bpb=1.2372, ~9100 steps, step_avg=65ms

### Phase 1 Activation Screen (8xH100, 10 min, 1 seed)
| Experiment | val_bpb | Delta |
|-----------|---------|-------|
| LeakyReLU(0.5)² | 1.2333 | -0.0039 |
| Baseline (relu²) | 1.2372 | — |
| LeakyReLU(0.5)² + softcap=20 | 1.2406 | +0.0034 |
| softcap=20 | 1.2426 | +0.0054 |
| SiLU | 1.2509 | +0.0137 |
| GELU | 1.2542 | +0.0170 |
| sin² | 1.2937 | +0.0565 |
| sin² + softcap=20 | 1.2971 | +0.0599 |

Logs: `results/h100_matrix_20260405/`

---

## Round 1: Proven Technique Stack

### R1-1: LeakyReLU(0.5)² + 11 Layers + 3x MLP

**Idea**: Stack three proven techniques from the current SOTA submissions:
1. **LeakyReLU(0.5)²** — preserves negative gradient flow by scaling negative inputs to 0.5x before squaring, eliminating dead neurons. Ablated at -0.002 to -0.003 bpb in SOTA submissions [PR #549, PR #414 on openai/parameter-golf].
2. **11 layers** (from 9) — deeper model, standard in all top-3 SOTA submissions.
3. **3x MLP** (from 2x) — wider MLP (hidden dim 1536 vs 1024), standard in top-3 SOTA.

**References**:
- LeakyReLU²: First used in [modded-nanogpt PR #414](https://github.com/KellerJordan/modded-nanogpt), adopted by parameter-golf SOTA submissions [PR #549](https://github.com/openai/parameter-golf/pull/549), [PR #1019](https://github.com/openai/parameter-golf/pull/1019).
- 11L + 3x MLP: Standard config in all top-3 leaderboard entries (val_bpb 1.1147–1.1570).

**Changes**: 1-line activation swap + env var overrides.

**Output**:
- RTX 5090: _pending_
- 8xH100: _pending_

---

### R1-2: BigramHash 3072 Embedding

**Idea**: Add token co-occurrence signal at the embedding layer via a hash-based bigram table. Instead of a full vocab² bigram matrix (prohibitively expensive), hash each consecutive token pair into a fixed-size embedding table using polynomial XOR hashing. This gives the model access to local context patterns (which token pairs tend to co-occur) at the embedding level, before any attention computation.

**References**:
- BigramHash embedding: Introduced in parameter-golf [PR #414](https://github.com/openai/parameter-golf/pull/414). Extended with trigram hash in [PR #1019](https://github.com/openai/parameter-golf/pull/1019). Present in all top-2 SOTA submissions.
- Hash function design: Polynomial multiply-XOR with coprime constants (36313, 27191) for collision minimization. No published derivation — empirically chosen.

**Key design choices**:
- Additive to token embedding (not concatenated) — no dimension increase
- Zero-initialized with learnable scale (0.05) — starts contributing nothing, learns to contribute
- Separate embedding dim (128) with projection to model dim (512)
- 3072 hash buckets with modulus 3071 (last bucket reserved as sentinel for position 0)

**Changes**: New BigramHashEmbedding class (~50 lines), added to forward before RMS norm.

**Output**:
- RTX 5090: _pending_
- 8xH100: _pending_

---

### R1-3: XSA (Cross-Sequence Attention)

**Idea**: After standard attention computes the output, remove the component parallel to the current token's own value vector using Gram-Schmidt orthogonalization. The intuition: the attention output already "knows" the token's own value via the direct V path; by projecting it out, the residual output carries only information gathered from *other* tokens, reducing redundancy.

**References**:
- XSA: Introduced in parameter-golf [PR #198](https://github.com/openai/parameter-golf/pull/198) as "Efficient Partial XSA". Extended to all layers in [PR #1019](https://github.com/openai/parameter-golf/pull/1019). Ablated at -0.003 bpb.
- Mathematically equivalent to a per-head Gram-Schmidt orthogonalization step, applied after the softmax-weighted value aggregation.

**Key design choices**:
- GQA-aware: reshapes `[B,T,H,D]` → `[B,T,Hkv,group,D]` to broadcast normalized V across query head groups without repeat_interleave
- Zero extra parameters
- Applied only on last N decoder layers (default N=4) — early layers benefit less

**Changes**: New `_xsa_efficient()` method on attention class (~15 lines). Note: requires attention output in `[B,T,H,D]` format (baseline uses `[B,H,T,D]` — may need layout adjustment).

**Output**:
- RTX 5090: _pending_
- 8xH100: _pending_

---

### R1-4: U-Net Skip Connections

**Idea**: Split the transformer into encoder (first half) and decoder (second half) with learnable skip connections between them, following the U-Net topology. Skip connections allow gradients to flow directly from decoder to early encoder layers, and allow the decoder to access lower-level representations. The LIFO stack creates symmetric connections: encoder layer 0 ↔ last decoder layer.

**References**:
- U-Net for transformers: Used in all top-3 parameter-golf SOTA submissions. The pattern originated in the [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt) where it was shown to improve convergence without additional parameters.
- U-Net architecture: Originally Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015).

**Key design choices**:
- 11 layers → 5 encoder + 6 decoder (integer division)
- 5 skip connections (min of encoder, decoder counts)
- Per-channel learned scale vector `[512]` per skip, initialized to ones
- Additive: `x = x + skip_weight * skip_activation`

**Changes**: Split forward loop into encoder/decoder, add skip stack (~30 lines).

**Output**:
- RTX 5090: _pending_
- 8xH100: _pending_

---

### R1-5: Value Residual Propagation

**Idea**: Propagate the first layer's raw value vectors to all subsequent layers via a learnable sigmoid gate. This gives deeper layers direct access to the initial token-level value representation before it has been mixed by attention in intermediate layers. The gate starts at 50% mixing (sigmoid(0)=0.5), allowing training to find the optimal blend.

**References**:
- Value residual: Present in both top-2 SOTA parameter-golf submissions. SOTA #1 uses sigmoid gate (`v + sigmoid(α)·v₀`), SOTA #2 uses unconstrained lambda mix (`λ₀·v₀ + λ₁·v`). Both originated from [PR #414](https://github.com/openai/parameter-golf/pull/414).
- Related concept: Value embeddings (Zhou et al. 2024), used in the [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt), add a second embedding table to V. Value residual is the dynamic version — it propagates actual V activations rather than a fixed embedding.

**Key design choices**:
- Sigmoid gate (SOTA #1 variant) — simpler, one parameter per layer
- v₀ captured from layer 0 after value projection, before RoPE
- v₀ is frozen after capture (not updated during decoder pass)
- Shape: `[B, T, 4, 64]` (per KV head)

**Changes**: Store v₀ from first layer, add vrl_alpha parameter to attention, mix in forward (~15 lines).

**Output**:
- RTX 5090: _pending_
- 8xH100: _pending_

---

## Round 2: Novel Designs

### R2-1: FAN Periodic MLP

**Idea**: Replace the standard MLP expansion with a Fourier Analysis Network (FAN) layer that dedicates 25% of hidden dimensions to periodic (sin/cos) features and 75% to standard LeakyReLU² features. The hypothesis: language data contains latent periodic structure (positional cycles, syntactic patterns) that purely monotonic activations cannot efficiently represent. Unlike the failed sin² experiment (R1 Phase 1), this approach ADDS periodic features alongside standard features rather than replacing them entirely.

**References**:
- Dong et al. "FAN: Fourier Analysis Networks" (arXiv 2410.02675, ICLR 2025) — 14.65% OOD loss reduction, 25% fewer parameters.
- Yu et al. "FANformer: Improving Large Language Models Through Effective Periodicity Modeling" (arXiv 2502.21309, Feb 2025) — 31% parameter efficiency, 20% fewer training tokens.
- Dec 2024 analysis (arXiv 2512.14873) — only sine (not cosine) contributes; gain comes from gradient shape near zero.

**Key design choices**:
- 25% Fourier ratio: shared W_p for both sin and cos (saves one projection vs separate)
- 75% standard: LeakyReLU(0.5)² (proven best activation from Phase 1)
- Actually 12.5% fewer params than standard MLP (W_p is shared)
- Concatenation, not addition: `[cos(p) || sin(p) || leaky_relu(W_bar·x)²]`

**Changes**: New FAN_MLP class (~30 lines). Spec: `round2-specs.md#r2-1`

**Output**: _pending_

---

### R2-2: DML-Gated MLP + Barlow Twins Loss

**Idea**: Design an MLP layer inspired by Double Machine Learning (Chernozhukov et al. 2018) with two parallel pathways — a "nuisance" pathway that captures predictable structure and a "target" pathway that captures residual causal signal. Element-wise gating (like SwiGLU) combines them, while a Barlow Twins decorrelation loss explicitly enforces orthogonality between the two pathways' representations.

**References**:
- Chernozhukov et al. "Double/debiased machine learning for treatment and structural parameters" (Econometrica, 2018) — the DML framework.
- Zbontar et al. "Barlow Twins: Self-Supervised Learning via Redundancy Reduction" (ICML 2021) — the decorrelation loss.
- Shazeer "GLU Variants Improve Transformer" (arXiv 2002.05202, 2020) — SwiGLU as dual-pathway prior art.
- Zhang et al. "ND-LoRA: Neural Diversity Regularizes Hallucinations in Language Models" (arXiv 2510.20690, 2025) — Barlow Twins on parallel streams reduces hallucinations 14.6%.
- Ahn et al. "PODNN: Parallel Orthogonal Deep Neural Network" (Neural Networks, 2021) — parallel streams + Gram-Schmidt.

**Key design choices**:
- 3-matrix structure (W_nuisance, W_target, W_out) — budget-neutral with SwiGLU
- Nuisance pathway uses LeakyReLU(0.5)² as gating function
- Target pathway is linear (no activation in gate branch, like bilinear FFN)
- Barlow Twins loss weight λ=0.01 (auxiliary, not dominant)

**Changes**: New DML_GatedMLP class + aux loss in training loop (~40 lines). Spec: `round2-specs.md#r2-2`

**Output**: _pending_

---

### R2-3: DML with Gram-Schmidt Orthogonalization

**Idea**: Same dual-pathway concept as R2-2, but instead of a training-time loss, enforce orthogonality structurally in the forward pass using Gram-Schmidt projection (identical math to XSA in attention). The target pathway's output has the nuisance component explicitly removed before the combiner. This is a stronger constraint than Barlow Twins — it guarantees orthogonality at every forward pass, not just on average over training.

**References**: Same as R2-2, plus:
- Gram-Schmidt orthogonalization: the same technique used by XSA (R1-3) in attention, applied here to MLP pathways.

**Key design choices**:
- Asymmetric widths: nuisance pathway is narrow (384d), target is wider (768d)
- 25% fewer params than standard MLP — could reallocate saved params
- No auxiliary loss needed — orthogonality is structural
- Concatenation + projection combiner (not multiplicative gating)

**Changes**: New DML_OrthMLP class (~35 lines). Spec: `round2-specs.md#r2-3`

**Output**: _pending_

---

### R2-4: FAN + DML-Gated Combo

**Idea**: Combine periodic FAN features (R2-1) with DML dual-pathway gating (R2-2). The value pathway produces periodic + standard features, the gate pathway controls information flow, and Barlow Twins enforces decorrelation between gate and value. This is the most architecturally novel MLP variant — if it works, it represents a genuinely new contribution combining three ideas (periodicity, causal orthogonalization, gated mixing).

**References**: Combination of R2-1 and R2-2 references.

**Changes**: New FAN_DML_MLP class (~55 lines). Spec: `round2-specs.md#r2-4`

**Output**: _pending_

---

### R2-5: Token Dropout (10%)

**Idea**: Drop random tokens from the input sequence during training as a causal intervention on the information flow. This regularizes along the *sequence* dimension (orthogonal to standard dropout which regularizes along the *feature* dimension), forcing the model to learn robust causal dependencies rather than relying on every context position being present. As a bonus, shorter sequences mean faster per-step computation — more training steps in the 10-minute window.

**References**:
- Conceptually related to: Srivastava et al. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (JMLR, 2014) — but applied to sequence positions, not features.
- Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" (NAACL, 2019) — token masking for MLM, but different objective (we drop, not mask-and-predict).
- Causal intervention interpretation: Pearl "Causality" (Cambridge, 2009) — do(remove token_i) as an intervention.

**Key design choices**:
- Shared mask across batch (same positions dropped for all sequences) — avoids variable-length padding
- Position 0 always kept — model needs at least one token
- RoPE positions shift (intentional — positional augmentation)
- 10% rate: balances regularization vs information loss

**Changes**: `token_dropout()` function in training loop (~10 lines). Spec: `round2-specs.md#r2-5`

**Output**: _pending_

---

### R2-7: Token Dropout + Rho-1 Selective Loss

**Idea**: Combine input-side causal selection (token dropout — which context to learn WITH) and output-side causal selection (Rho-1 — which tokens to learn FROM). Token dropout removes 10% of input tokens; Rho-1 skips loss on the 20% easiest-to-predict output tokens. Together, the model trains on a curated subset: robust context (via dropout) predicting hard targets (via Rho-1).

**References**:
- R2-5 references (token dropout)
- Lin et al. "Rho-1: Not All Tokens Are What You Need" (NeurIPS 2024, best paper runner-up) — selective token loss using reference model scoring.
- Our simplified Rho-1: uses logit margin (top1 - top2) as zero-cost difficulty proxy instead of a separate reference model.

**Changes**: `selective_loss()` function + R2-5 token dropout (~25 lines). Spec: `round2-specs.md#r2-7`

**Output**: _pending_

---

### R2-8: Graduated Token Dropout (20%→0%)

**Idea**: Apply heavy token dropout early in training (when the model is learning general features and benefits from regularization), then linearly decay to zero (when the model is fine-tuning precise predictions and benefits from clean data). This follows the curriculum learning principle: "easy task first, hard task later" — where "easy" means "with augmentation" and "hard" means "exact prediction on full context."

**References**:
- Bengio et al. "Curriculum Learning" (ICML 2009) — progressive task difficulty.
- R2-5 references (token dropout)

**Changes**: Step-dependent drop_rate in token_dropout (~15 lines). Spec: `round2-specs.md#r2-8`

**Output**: _pending_

---

### R2-9: Cross-Layer Barlow Twins

**Idea**: Force each transformer block to produce representations that are decorrelated from its neighbors, preventing redundant computation across layers. The Barlow Twins loss penalizes off-diagonal elements of the cross-correlation matrix between adjacent layers' outputs. If layer 5 and layer 6 produce highly correlated outputs, the loss pushes them apart — forcing each layer to contribute unique information to the representation.

**References**:
- Zbontar et al. "Barlow Twins" (ICML 2021) — cross-correlation decorrelation loss.
- Zhang et al. "ND-LoRA" (arXiv 2510.20690, 2025) — Barlow Twins between parallel streams (we extend to sequential layers).

**Key design choices**:
- Adjacent pairs only (not all layer combinations) — 10 pairs for 11 layers
- Subsample features (128 of 512 dims) to reduce cost from O(512²) to O(128²)
- λ = 0.005 (lighter than R2-2's 0.01, since this affects all layers)

**Changes**: Store layer outputs, compute pairwise BT loss in training loop (~20 lines). Spec: `round2-specs.md#r2-9`

**Output**: _pending_

---

### R2-11: Corrupted Context Training (10%)

**Idea**: Bridge the train/inference gap (exposure bias) by occasionally replacing ground truth tokens with the model's own predictions during training. Standard teacher forcing always shows perfect context; at inference the model sees its own (possibly wrong) predictions. By training on corrupted context, the model learns to predict correctly even from imperfect inputs — making it more robust at generation time.

**References**:
- Bengio et al. "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks" (NeurIPS 2015) — the original exposure bias mitigation.
- Ranzato et al. "Sequence Level Training with Recurrent Neural Networks" (ICLR 2016) — REINFORCE-based approach.
- Parameter-golf SOTA #1 uses AR self-generation for GPTQ calibration — same idea applied post-training.

**Key design choices**:
- Extra forward pass (no grad) for predictions — 30-40% compute overhead
- Replace with argmax predictions (not sampled) — deterministic corruption
- Targets remain original ground truth — model must correct for corrupted context
- Position 0 never corrupted

**Changes**: `corrupted_context()` function in training loop (~15 lines). Spec: `round2-specs.md#r2-11`

**Output**: _pending_

---

### R2-12: Graduated Corruption (0%→20%→0%)

**Idea**: Apply corrupted context training on a sine schedule — no corruption at start (learn basics from clean data), peak corruption at midpoint (learn robustness), zero corruption at end (fine-tune on clean data). The sine curve ensures smooth transitions and concentrates corruption when the model is mature enough to learn from it.

**References**: R2-11 references + curriculum learning (Bengio et al. 2009).

**Changes**: Step-dependent rate in corrupted_context (~20 lines). Spec: `round2-specs.md#r2-12`

**Output**: _pending_

---

### R2-13: CausalWide MLP (Multi-Bank Orthogonal Decomposition)

**Idea**: Replace depth with structured width. Instead of 11 narrow layers (3x MLP, hidden=1536), use 8 wider layers (~4.7x MLP, hidden≈2400) with the same total parameter budget (~26.5M). The wide MLP is internally structured into three orthogonal banks:

1. **Bank A ("Memory")** — Pure linear projection, no activation. Acts as an associative lookup table that memorizes token co-occurrence patterns. This is what extra depth would provide via multi-hop attention.
2. **Bank B ("Feature")** — LeakyReLU(0.5)² nonlinear features. Standard compositional feature extraction.
3. **Bank C ("Residual")** — LeakyReLU(0.5)² features, then Gram-Schmidt orthogonalized against Banks A and B. Captures whatever the other two banks miss — the "causal residual" after removing predictable structure.

Barlow Twins loss enforces cross-bank decorrelation (3 pairs: A-B, A-C, B-C).

The hypothesis: explicit causal disentangling (nuisance removal via Gram-Schmidt, redundancy prevention via Barlow Twins) in a wide layer can substitute for the implicit feature refinement that multiple narrow layers provide. Width gives memory capacity; causal structure gives depth-like abstraction.

**References**:
- Width vs depth: Ternary U-Net submission uses 768d/10L (wider+shallower) → outperforms 512d/25L because more training steps per wallclock.
- Wide and Deep Learning: Cheng et al. "Wide & Deep Learning for Recommender Systems" (DLRS 2016) — separate wide (memorization) and deep (generalization) components.
- Gram-Schmidt orthogonalization: Same technique as XSA (R1-3), applied within MLP instead of attention.
- Barlow Twins: Zbontar et al. (ICML 2021) — cross-correlation decorrelation.
- DML orthogonalization: Chernozhukov et al. (2018) — nuisance/target decomposition.

**Literature context** (from research sweep):
- HuggingFace 70M study: 4L×768 (ultra-wide shallow) FAILS — but 8L is safely above the 4L danger zone.
- Superposition hypothesis (Elhage et al. 2022): Wide networks prefer polysemantic (entangled) features even when monosemantic solutions exist. Explicit orthogonalization may be needed to break this preference — width alone won't disentangle.
- Transformers spontaneously learn factored orthogonal subspaces (arXiv 2602.02385, 2025) — our Gram-Schmidt accelerates what gradient descent already tends toward.
- No published paper tests orthogonalization-replacing-depth in LMs. This is novel territory.

**Config**: `MLP_TYPE=causal_wide NUM_LAYERS=8 MLP_MULT=5` (8L × 5x ≈ 27.0M, matching 11L × 3x budget)

**Changes**: New CausalWideMLP class (~75 lines). Spec: `round2-specs.md`

**Output**: _pending_

---

### R2-15: Adaptive Causal Probing (PLANNED — not yet implemented)

**Idea**: For tokens the model already predicts correctly, apply aggressive context dropout (30%) to stress-test which context is truly causally necessary. For hard tokens, keep full context. This is the inverse of Rho-1 — instead of skipping easy tokens, make them harder.

**Key insight**: Standard token dropout (R2-5) drops context uniformly. But easy predictions don't need augmentation challenge — they're already solved. Hard predictions need full context to learn from. Adaptive probing focuses the causal intervention where it has the most learning value.

**Recommended approach**: Single-pass variant using previous step's per-position loss as difficulty proxy. Zero compute overhead — same number of forward passes as standard training.

**Status**: Spec complete at `r2-15-adaptive-causal-probing-spec.md`. Not yet implemented. Waiting for benchmark results to prioritize.

**References**: Inverse of Rho-1 (NeurIPS 2024), Pearl's do-calculus, our R2-5 token dropout, our R2-11 corrupted context.

**Output**: _not yet implemented_

---

### R2-14: DML-CausalWide MLP (Nested Causal Decomposition)

**Idea**: Two-level causal structure combining CausalWide (R2-13) and DML (R2-2). Level 1 decomposes the MLP into 3 orthogonal banks (memory, feature, residual). Level 2 applies DML dual-pathway gating (nuisance gate x target value) *within* each bank. Barlow Twins operates at both levels — cross-bank decorrelation (L1) and within-bank nuisance/target decorrelation (L2). This is the most architecturally novel variant, testing whether nested causal decomposition provides deeper feature disentangling than either technique alone.

**References**:
- All R2-2 references (DML, Barlow Twins, ND-LoRA)
- All R2-13 references (CausalWide, Wide & Deep, superposition)
- Novel combination — no published precedent for nested causal decomposition in transformer MLP layers.

**Key design choices**:
- Budget-neutral: sub_width = 2*hidden/9 per bank (6 projection matrices + 1 combiner = same param count as standard 2-matrix MLP)
- Bank A (Memory): bilinear gate (linear×linear — pure associative lookup with second-order interactions)
- Bank B (Feature): LeakyReLU(0.5)² gate × linear value
- Bank C (Residual): Same as B, then Gram-Schmidt orthogonalized against A and B
- Two-level BT loss: 3 cross-bank pairs + 3 within-bank pairs = 6 decorrelation terms, averaged

**Config**: `MLP_TYPE=dml_causal_wide NUM_LAYERS=8 MLP_MULT=5` (same budget as 11L×3x)

**Changes**: New DML_CausalWideMLP class (~90 lines). 8/8 DoD tests pass.

**Output**: _pending_

---

## Round 3: Quantization & Eval

### R3-1: GPTQ int6 (Full Hessian)

**Idea**: Apply GPTQ post-training quantization to compress model weights from bf16 to int6 (6 bits per weight). Full Hessian GPTQ uses the exact second-order error structure to optimally compensate quantization errors — when column i is quantized with error δ, the remaining columns are adjusted by `δ × H⁻¹[i, :]` to minimize total output error. Calibration data is generated autoregressively by the model itself (no train/val data access required by competition rules).

**References**:
- Frantar et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers" (arXiv 2210.17323, 2022).
- AR self-generation calibration: parameter-golf [PR #1019](https://github.com/openai/parameter-golf/pull/1019).

**Changes**: GPTQ quantization functions + AR calibration (~200 lines). Spec: `round3-specs.md#r3-1`

**Output**: _pending_

---

### R3-2: Sliding Window Evaluation (stride=64)

**Idea**: Evaluate with overlapping windows so every token (except the first stride positions of the first window) is scored with at least `seq_len - stride` tokens of context. Standard chunked eval gives early positions in each chunk minimal context, hurting their predictions. Sliding window fixes this at the cost of 16x more eval forward passes (for stride=64, seq_len=1024).

**References**:
- Press et al. "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (ICLR 2022).
- Standard practice in parameter-golf since [PR #198](https://github.com/openai/parameter-golf/pull/198).

**Changes**: Modified eval loop (~20 lines). Spec: `round3-specs.md#r3-2`

**Output**: _pending_

---

### R3-3: Legal Score-First TTT

**Idea**: Fine-tune the model on validation data chunks sequentially, but always SCORE each chunk BEFORE training on it — maintaining legality (the model hasn't seen the data it's being graded on). As the model adapts to the validation distribution, later chunks are scored by a model that has been fine-tuned on earlier chunks, improving predictions through online adaptation.

**References**:
- Sun et al. "Test-Time Training with Self-Supervision for Generalization under Distribution Shifts" (ICML 2020).
- Legal Score-First TTT: parameter-golf [PR #549](https://github.com/openai/parameter-golf/pull/549) — ablated at -0.0025 bpb.

**Changes**: TTT evaluation function (~150 lines). Spec: `round3-specs.md#r3-3`

**Output**: _pending_
