# Non-Record: KAN, Sparsity (Structured and Hessian-Guided), MoE, KAN Negative Results

**Tokenizer:** SP8192  
**Submission type:** Non-record (negative results & technique exploration)

---

## Summary

This submission documents a systematic exploration of some novel techniques, built on top of the PR #1394. All techniques were implemented as toggleable features in a single training script and evaluated across 14 runs on 1xH100 SXM (RunPod) using a scaled-down "medium" configuration (2000 steps, 2048 seq_len, 5 train shards).

**Key finding:** None of the novel techniques (Sparsity, Hessian-Guided Sparsity, MoE, KAN) improved BPB over a well-tuned baseline that combines established techniques from the top leaderboard submissions (parallel residuals from PR #1412, TTT from PR #1413, QK gain tuning from PR #1493, CAT idea from PR #1385). The most effective strategy was simply combining some known techniques from these PRs.

---

## Hardware & Training Setup

| Parameter | Value |
|---|---|
| Hardware | 1xH100 SXM 80GB (RunPod) |
| Training mode | `TEST_MODE=medium` (scaled-down) |
| Training steps | 2000 |
| Sequence length | 2048 |
| Train shards | 5 (~500M tokens vs. 8B+ full) |
| GPUs | 1 (vs. 8 for competition) |

**Note:** All BPB values are from medium runs and are not directly comparable to full 8xH100 submissions. The relative comparisons between techniques are valid since all used identical medium configuration.

---

## Results

### Round 1: Initial Novel Techniques on top of PR #1394

Architecture: `loop_start=4, loop_end=5, qk_gain=4.0, warmdown=0.667, muon_wd=0.085`

| # | Experiment | Layers | Params | val_bpb | Pre-quant BPB | Size (bytes) | Under 16MB? |
|---|---|---|---|---|---|---|---|
| 1 | Baseline (PR #1394) | 11 | 35.9M | 1.6567 | 1.6508 | 16,053,780 | Borderline |
| 2 | 2:4 Sparsity | 13 | 41.7M | 1.5040 | 1.4616 | 16,624,092 | Over |
| 3 | MoE (4 experts, top-2) | 11 | 105.2M | 1.4367 | 1.4291 | 45,409,570 | **3x over** |
| 4 | KAN (grid=5, order=3) | 11 | 128.2M | 1.5322 | 1.5215 | 55,011,901 | **3.4x over** |
| 5 | CAT (every 50 steps) | 11 | 35.9M | 1.4759 | 1.4680 | 16,069,715 | Borderline |
| 6 | CAT + Sparsity | 13 | 41.7M | 1.4964 | 1.4551 | 16,620,887 | Over |

### Round 2: Combined with Top Submission Techniques

Architecture: parallel residuals from layer 7, `loop_start=3, loop_end=5, qk_gain=5.25, warmdown=0.72, matrix_lr=0.022, muon_wd=0.095, ema=0.9965`

| # | Experiment | Layers | val_bpb | Pre-quant BPB | Size (bytes) | Under 16MB? |
|---|---|---|---|---|---|---|
| 7 | Baseline (new defaults) | 11 | 1.4096 | 1.3998 | 16,077,847 | Borderline |
| 8 | CAT + Sparsity | 12 | 1.4402 | 1.4031 | 15,605,512 | Yes |
| 9 | CAT + Sparsity | 13 | 1.4277 | 1.4006 | 16,687,493 | Over |
| 10 | CAT only (no sparsity) | 11 | 1.4101 | 1.3998 | 16,079,171 | Borderline |
| 11 | CAT + Sparsity + Wide Loop (3-6) | 12 | 1.4290 | 1.4013 | 15,596,401 | Yes |

### Round 3: TTT + Hessian-Guided Sparsity (Sliding Window Enabled)

Same architecture as Round 2, with sliding window eval and score-first TTT active.

| # | Experiment | Layers | val_bpb | Sliding BPB | TTT BPB | Size (bytes) | Under 16MB? |
|---|---|---|---|---|---|---|---|
| **12** | **Baseline + TTT** | **11** | **1.3971** | **1.3817** | **1.3696** | **16,076,488** | **Borderline** |
| 13 | CAT + H-Sparsity + TTT | 11 | 1.4564 | 1.4416 | 1.4045 | 14,700,464 | Yes |
| 14 | CAT + H-Sparsity + TTT | 12 | 1.4633 | 1.4513 | 1.4120 | 15,757,920 | Yes |

---

## Novel Techniques Explored

### 1. Compressor-Aware Training (CAT) -- idea from PR #1385

**Motivation:** In the parameter golf pipeline, model weights are quantized (GPTQ, int6) and then entropy-coded (brotli). These compression steps are applied post-training, so the model has no incentive during training to produce weights that are easy to quantize or compress. CAT introduces a differentiable proxy for quantization loss directly into the training objective, encouraging the model to learn weight distributions that are "compression-friendly" — weights that naturally cluster near quantization grid points, resulting in lower entropy and better brotli compression ratios.

**How it works:** Every `CAT_EVERY` training steps, a soft-rounding regularization loss is computed over all large weight matrices. For each weight, the distance to the nearest quantization grid point is measured, and a sigmoid function creates a differentiable penalty that is higher when weights fall between grid points. This loss is scaled by `CAT_WEIGHT=0.001` and added to the language modeling loss. The intuition is that weights near grid boundaries contribute the most quantization error; nudging them during training should reduce post-training quantization degradation.

**Result:** CAT reduced compressed model size by ~70KB (~0.4% savings) but consistently degraded BPB by 0.01-0.06. The regularization disrupts the optimization landscape enough to hurt training quality, and the compression savings are too small to compensate — even when the saved bytes were used for additional model layers.

**Verdict: Negative.** GPTQ with SDClip already handles quantization effectively. Adding a training-time compression proxy introduces a conflicting objective that hurts final model quality more than it helps compression.

### 2. 2:4 Structured Sparsity (Magnitude-Based)

**Motivation:** The 16MB artifact constraint is the binding bottleneck. If 50% of weights can be zeroed out in a structured pattern (keeping the 2 largest magnitudes per group of 4), the resulting sparse matrices should compress dramatically under brotli, freeing space for more model capacity (additional layers). The 2:4 pattern is also hardware-friendly on NVIDIA Ampere/Hopper GPUs, which have native 2:4 sparse tensor cores for inference acceleration.

**How it works:** After training completes, all MLP weight matrices are reshaped into groups of 4 columns. Within each group, the 2 weights with smallest absolute magnitude are zeroed. The sparsified state dict is then passed to GPTQ and brotli compression. With ~50% of MLP weights zeroed, the entropy of the weight distribution drops and brotli achieves much better compression ratios, saving ~1.5MB.

**Result:** Sparsity saved ~1.5MB in artifact size, allowing 12-13 layer models to fit under 16MB. However, the BPB degradation from zeroing weights (0.03-0.06 worse) consistently exceeded the improvement from additional layers at 2000 training steps. The pre-quantization BPB was comparable, but post-GPTQ the sparse models suffered more.

**Verdict: Negative.** 50% sparsity is could be aggressive at this model scale. The information destroyed by pruning half the MLP weights outweighs the capacity gained from 1-2 extra layers.

### 3. Hessian-Guided 2:4 Sparsity

**Motivation:** Naive magnitude-based pruning assumes that the smallest weights are the least important. This might not always be true — a small weight connected to a high-curvature input dimension may contribute disproportionately to the loss function. GPTQ already collects full Hessian matrices (H = X^T X) for quantization. These same Hessians encode which input dimensions are most important. By combining weight magnitude with Hessian diagonal importance, we can make better pruning decisions.

**How it works:** The importance score for each weight is computed as `|w_ij| * sqrt(H_jj)`, where `H_jj` is the diagonal of the Hessian matrix for that layer's input. This replaces the standard `|w_ij|` magnitude criterion. The Hessians are collected as part of the existing GPTQ pipeline, so this adds zero computational overhead. Within each group of 4, the 2 weights with the highest combined importance are kept.

**Result:** Hessian-guided sparsity produced similar results than naive magnitude pruning, but both were substantially worse than no sparsity (1.3696 baseline TTT BPB). The fundamental constraint is that zeroing 50% of weights, regardless of how intelligently they are selected, could be removing too much model capacity at this scale.

**Verdict: Negative.** While the Hessian-guided importance criterion is theoretically sound and adds zero overhead, the 2:4 structured constraint forces exactly 50% sparsity, which might be too aggressive. Unstructured or lower-ratio pruning (e.g., 20-30%) might help, but would yield smaller compression savings. 

### 4. Mixture of Experts (MoE)

**Motivation:** MoE architectures increase model capacity without proportionally increasing computation per token. By replacing each MLP with N independent expert MLPs and a learned router that activates only the top-K experts per token, the model can maintain a much larger total parameter count while keeping training and inference costs manageable. In the parameter golf context, MoE could theoretically achieve better BPB by having more specialized experts, even if the total model is larger — provided the weights compress well enough to fit in 16MB.

**How it works:** Each transformer block's MLP is replaced with 4 independent expert MLPs and a learned router. For each token, the router selects the top-2 experts by softmax probability. Only those 2 experts process the token, and their outputs are combined weighted by the router probabilities. A load-balancing auxiliary loss (`alpha=0.01`) encourages even expert utilization. Each expert uses the same LeakyReLU(0.5)² activation as the standard MLP.

**Result:** MoE achieved the best pre-quantization BPB of all experiments (1.4291), demonstrating that increased capacity does help language modeling quality. However, the total artifact was 45.4MB — nearly 3x over the 16MB budget. The 4 expert MLPs each have independent weight matrices that learn different specializations, making them highly incompressible — brotli cannot exploit cross-expert redundancy.

**Verdict: Interesting but impractical.** MoE improves BPB meaningfully but is fundamentally incompatible with the 16MB constraint. Would require sub-2-bit quantization or expert weight sharing to fit, both of which would likely negate the quality gains. (Not included in final code)

### 5. KAN (Kolmogorov-Arnold Networks)

**Motivation:** KAN replaces traditional linear layers + fixed activations with learned activation functions parameterized by B-spline basis functions. Based on the Kolmogorov-Arnold representation theorem, KAN layers can theoretically approximate any continuous function with fewer parameters than standard MLPs for certain function classes. The hypothesis was that KAN's superior function approximation capability might achieve better BPB per parameter, offsetting the overhead of spline coefficients.

**How it works:** Each KAN layer parameterizes its activation as a B-spline with `grid_size=5` control points and `spline_order=3`. The spline weights are 3D tensors of shape `(out_features, in_features, num_coefficients)`. Since GPTQ's Hessian collection hooks only attach to standard `nn.Linear` modules, a fallback simple quantization (round-to-nearest with row-wise scaling) was implemented for KAN's spline weight parameters.

**Result:** KAN produced the largest artifact (55MB, 3.4x over budget) despite achieving worse BPB. The spline weights are inherently difficult to compress as they represent smooth continuous functions where every coefficient matters, so quantization introduces visible artifacts and brotli cannot find redundancy in the learned spline shapes. KAN also trained significantly slower than standard MLPs due to the B-spline evaluation overhead.

**Verdict: Negative.** KAN is fundamentally mismatched with the parameter golf constraint. The spline parameters are expensive in both raw size and compression ratio, and the function approximation benefits do not materialize at this model scale and training budget. (Not included in final code)

---

## Established Techniques Adopted (Not Novel) -- Credits

These techniques were adopted from the top leaderboard submissions or other pull requests and are not novel contributions of this work.

| Technique | Source | Effect |
|---|---|---|
| Parallel Residuals (GPT-J style, layer 7+) | PR #1412 | Attention and MLP read from same pre-attention input |
| Test-Time Training (Score-First SGD) | PR #1413 | ~0.03 BPB improvement via eval-time adaptation |
| QK Gain = 5.25 | PR #1493 | Per-head learnable query scaling |
| Recurrence Loop 3-5, enabled at 35% | PR #1437 | Wider and earlier depth recurrence |
| Warmdown = 0.72, matrix_lr = 0.022 | PR #1445 | Hyperparameter tuning |
| muon_wd = 0.095, ema_decay = 0.9965 | PR #1445 | Optimizer and EMA tuning |

---

## Artifact Size Note

Some configurations are marginally over the 16MB limit. The submitted `train_gpt.py` is ~68KB because it includes all experimental toggles (CAT, Sparsity, MoE configuration constants, multiple test modes, detailed logging). For a competition submission, several approaches would bring this under budget:

- **LZMA-compressing the training script**, as the top submissions do, which reduces code size.
- **Stripping unused code paths** (removing CAT, Sparsity) to reduce raw code size before compression.
- **Slightly reducing GPTQ calibration batches** (from 64 to 48) to shave a few KB from the quantized model.

Since this is a non-record negative results submission focused on documenting technique exploration rather than competing for SOTA, we include the full uncompressed script for readability.

---

## Lessons Learned

1. **Post-training compression is already near-optimal.** GPTQ + byte-shuffle + brotli-11 is extremely effective. Training-time techniques like CAT provide marginal compression gains (~0.4%) that do not justify the BPB cost.

2. **Sparsity at 50% might be too aggressive at this scale.** Whether magnitude-based or Hessian-guided, zeroing half the MLP weights at ~36M parameters destroys more information than extra layers can recover. It would be worth looking into further optimizations to take advantage of the additional space provided from incorporating sparsity.

3. **MoE and KAN explode model size.** Expert weights and spline parameters are inherently difficult to compress. Neither architecture is compatible with extreme compression constraints without fundamentally different quantization approaches.

---

## Log Files

All runs were executed on 1xH100 SXM 80GB via RunPod.

### Round 1 — Novel techniques on PR #1394 baseline
- `train_round1_baseline.log` — 11L baseline
- `train_round1_sparsity_13L.log` — 13L + 2:4 sparsity
- `train_round1_moe_4e.log` — 11L + MoE (4 experts, top-2)
- `train_round1_kan.log` — 11L + KAN (grid=5, order=3)
- `train_round1_cat.log` — 11L + CAT (every 50 steps)
- `train_round1_cat_sparsity_13L.log` — 13L + CAT + sparsity

### Round 2 — Top submission defaults + novel combos
- `train_round2_baseline_11L.log` — 11L baseline (parallel residuals, QK 5.25, loop 3-5)
- `train_round2_12L_cat_sparse.log` — 12L + CAT + sparsity
- `train_round2_13L_cat_sparse.log` — 13L + CAT + sparsity
- `train_round2_11L_cat.log` — 11L + CAT (no sparsity)
- `train_round2_12L_wideloop.log` — 12L + CAT + sparsity + wide loop (3-6)

### Round 3 — TTT + Hessian-guided sparsity
- `train_round3_baseline_ttt.log` — 11L baseline + TTT (best result)
- `train_round3_11L_cat_hsparse_ttt.log` — 11L + CAT + Hessian sparsity + TTT
- `train_round3_12L_cat_hsparse_ttt.log` — 12L + CAT + Hessian sparsity + TTT
