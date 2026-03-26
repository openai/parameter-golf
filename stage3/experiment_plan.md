# Stage 3 Experiment Plan (Revised 2026-03-24)

## Design Principle

Stage 2_1 owns the community playbook: LeakyReLU^2, EMA, MuonWD, XSA, GPTQ, Partial RoPE, LN Scale, curriculum.

Stage 3 tests **original hypotheses from cross-domain mechanism transfer**. Each idea comes from a different field and attacks a surface nobody in the pgolf community has touched.

| Slot | Idea | Surface | Source Domain |
|------|------|---------|---------------|
| H1 | Z-loss regularization | Loss function (logit magnitudes) | PaLM / large-scale LM stabilization |
| H2 | Adaptive Newton-Schulz steps | Optimizer internals (computation budget) | Control theory (gain scheduling) |
| H3 | Nuclear norm regularization | Weight structure (effective rank) | Rate-distortion theory / transform coding |
| H4 | Stochastic weight perturbation | Landscape geometry (minima flatness) | Langevin dynamics / channel coding |
| H5 | Gradient centralization | Gradient preprocessing | Computer vision (ICCV 2020, Yong et al.) |
| H6 | H1+H3 compound | Cross-surface | — |

Zero overlap with stage2_1. Zero overlap with community PRs. Five distinct surfaces.

## 1. Objective

**Current**: 1.1631 BPB. **Frontier**: 1.1158. **Gap**: 0.047.

Stage 3 is a discovery screen, not a gap-closing effort. We're looking for novel mechanisms that could compound with stage2_1 winners.

## 2. Hypothesis Details

### H1: Z-Loss Regularization (PaLM)

**Mechanism**: Add `z_loss_weight * mean(logsumexp(logits)^2)` to the training loss.

Penalizes the log-partition function directly. When logits drift to large magnitudes, the partition function grows exponentially. Z-loss keeps it in check via a soft, differentiable penalty.

**Why it's distinct**: Logit softcapping (cap=30, already present) is a hard clamp — it truncates but doesn't shape the distribution. Label smoothing (not present) changes the target distribution. Z-loss penalizes the model's confidence about its OWN output distribution, regardless of the target. It's a regularizer on the output entropy, not the targets.

**Key question**: Does z-loss add anything on top of softcapping? If softcap already prevents logit explosion, z-loss may be redundant. If z-loss wins, it proves the soft penalty is better than the hard clamp.

### H2: Adaptive Newton-Schulz Steps (Control Theory)

**Mechanism**: Schedule Muon's Newton-Schulz iteration count:
- First 20% of training: 7 steps (noisier gradients need better orthogonalization)
- Middle 60%: 5 steps (standard)
- Last 20%: 3 steps (weights converging, cheaper ortho OK)

From gain scheduling in control theory: a PID controller's gains are retuned as the plant moves through different operating regimes. Early training is a different regime from late training.

**Key question**: Is the quality of the polar decomposition approximation important, or is 5 steps already in the "good enough" regime everywhere? If 7 early helps, the gradient structure early in training is fundamentally different. If 3 late doesn't hurt, we get free throughput.

### H3: Nuclear Norm Regularization (Rate-Distortion)

**Mechanism**: Add `nuclear_norm_weight * mean(sum_of_singular_values)` penalty for each matrix parameter.

From transform coding: when you compress a signal, concentrating energy in fewer coefficients (basis functions) minimizes distortion at a given rate. The nuclear norm (sum of singular values) measures the effective rank. Penalizing it pushes weight matrices toward lower rank — fewer significant singular values — which makes quantization less destructive because there are fewer important values to preserve.

**Key question**: Is the quantization loss caused by weight magnitude (WD fixes this) or weight rank/structure (nuclear norm fixes this)? They target different aspects of the same problem. If nuclear norm wins, it proves STRUCTURE matters, not just MAGNITUDE.

**Caveat**: SVD is expensive. Computed only on the first micro-step to limit overhead. If ms/step increases >10%, the throughput cost may negate the quality gain.

### H4: Stochastic Weight Perturbation (Langevin Dynamics)

**Mechanism**: After each optimizer step, add Gaussian noise to matrix weights: `p += randn_like(p) * lr * scale`. Scale=0.01 by default.

From Langevin dynamics: SGD is already noisy (mini-batch gradient variance), but the noise is a side effect of batching, not intentional. Langevin dynamics adds EXPLICIT noise scaled to explore the posterior over weights. Sharp minima are destabilized (loss increases with perturbation), flat minima are robust (loss barely changes). Over training, this biases convergence toward flatter regions.

Also analogous to channel coding: adding noise during encoding makes the signal robust to the noisy channel (quantization).

**Key question**: Does SGD's implicit noise already find flat enough minima? If explicit noise helps, the landscape has meaningfully different local geometries that matter for quantization. If it hurts, SGD is already optimal for exploration.

### H5: Gradient Centralization (ICCV 2020)

**Mechanism**: Before each optimizer step, subtract the mean of each gradient tensor: `p.grad -= p.grad.mean(dim=non_batch_dims)`. Applied only to 2D matrix params.

From "Gradient Centralization: A New Optimization Technique for Deep Neural Networks" (Yong et al., ICCV 2020). Proven to improve convergence speed and generalization in image classification. The DC component of the gradient represents a global shift all neurons want to make. Removing it forces the optimizer to focus on the differential structure between neurons.

**Key question**: Does the mean gradient carry useful signal in LMs? In images, the mean often reflects lighting/contrast shifts that are nuisance variables. In LMs, it might reflect global weight scale adjustments that ARE useful. This is the core domain-transfer question.

**Zero new hyperparameters.** It's on or off.

### H6: Z-Loss + Nuclear Norm Compound

Two regularizers from opposite ends of the forward pass: z-loss shapes the output distribution, nuclear norm shapes the weight matrices. Mechanistically independent — tests whether novel regularizers from different surfaces compose.

## 3. Screen Layout

```
GPU 0: R0A  (control)
GPU 1: R0B  (control repeat)
GPU 2: H1   (LOSS: z-loss)
GPU 3: H2   (OPT: adaptive NS steps)
GPU 4: H3   (REG: nuclear norm)
GPU 5: H4   (LANDSCAPE: weight perturbation)
GPU 6: H5   (GRAD: gradient centralization)
GPU 7: H6   (CROSS: z-loss + nuclear norm)
```

180s each, 1 GPU each, all in parallel. ~4 min total.

## 4. Decision Rules

| Outcome | Action |
|---------|--------|
| Candidate worse than R0A by > 2x noise floor | Kill |
| Candidate better than R0A by > 2x noise floor | Promote to 600s validation |
| H6 > max(H1, H3) | Cross-surface compounds work. Compound stage3 winners with stage2_1 winners. |
| H6 < max(H1, H3) | Over-regularization. Keep solos only. |
| H3 ms/step > 110% of R0A | Nuclear norm SVD too expensive. Try cheaper proxy (trace norm). |
| All lose | These surfaces are tapped out at this scale. Focus on stage2_1 community techniques. |
| All lose but delta is < noise floor | Screen too short. Extend to 600s for promising ideas. |

## 5. If Winners Found

Any stage3 winner becomes a compound partner for stage2_1 winners. The compound hypothesis:

```
stage2_1 best (e.g., LeakyReLU^2 + MuonWD + EMA) + stage3 best (e.g., z-loss)
```

This is the highest-information compound because it combines the best community technique with the best novel technique on orthogonal surfaces.

## 6. What Each Outcome Teaches

| If H1 wins | Z-loss proves loss-function regularization is an untapped surface. Try other loss modifications: focal loss, poly loss, entropy penalty. |
|------------|---|
| If H1 loses | Logit softcapping subsumes z-loss. The loss-function surface is defended by the existing architecture. |
| If H2 wins | Newton-Schulz quality matters early AND cheaper NS is OK late. Optimizer internals are tunable. |
| If H2 loses | 5 steps is universally correct. NS convergence is not the bottleneck. |
| If H3 wins | Weight STRUCTURE (rank) matters for quantization, not just magnitude. Opens spectral regularization. |
| If H3 loses | Quantization loss is about outlier values, not rank. Weight decay is the right tool. |
| If H4 wins | Landscape geometry matters. Flat minima help quantization. Opens SAM-like ideas. |
| If H4 loses | SGD implicit noise is sufficient. Explicit exploration hurts more than helps. |
| If H5 wins | Gradient centralization transfers from CV to LM. Opens gradient preprocessing as a surface. |
| If H5 loses | The mean gradient carries useful signal in LMs. CV and LM gradients are fundamentally different. |
