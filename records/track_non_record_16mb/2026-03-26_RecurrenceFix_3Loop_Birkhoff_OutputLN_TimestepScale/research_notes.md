# Research Notes: Theoretical Foundations for Depth Recurrence Stabilization

This document provides citations and theoretical grounding for the three techniques developed to stabilize depth recurrence in parameter-shared transformers within the 16MB parameter-golf competition.

## 1. The Three Failure Modes of Depth Recurrence

Depth recurrence (weight-shared looping) has been attempted 15+ times in this competition with no SOTA result. Three failure modes explain why.

### 1a. Quantization Error Amplification

When the same weight matrix $W$ is applied $k$ times in a forward pass, post-training quantization error $\epsilon$ from int8 rounding compounds across iterations. PR #363 measured ~900× amplification over 3 cycles.

GPTQ quantizes layer-by-layer, compensating downstream weights for each layer's rounding error (Frantar et al., 2023, §3). With shared weights, this compensation is impossible. The same quantized matrix must serve all iterations. Errors from early iterations propagate uncompensated through later ones.

> Frantar, E., Ashkboos, S., Hoefler, T. & Alistarh, D. (2023). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." ICLR 2023. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)

### 1b. Per-Iteration Identity Collapse

Shared weights produce identical transformations for identical inputs. Without per-iteration conditioning, all loop iterations compute the same function. This collapses depth recurrence to a single effective pass.

Dehghani et al. (2019, §2.1) addressed this with sinusoidal timestep embeddings added at each recurrence step. Xu & Sato (2025) formalized the limitation: without timestep encoding, looped transformers have strict approximation rate bounds (Lemma 4.1).

> Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J. & Kaiser, L. (2019). "Universal Transformers." ICLR 2019. [arXiv:1807.03819](https://arxiv.org/abs/1807.03819)

### 1c. Residual Magnitude Erasure

Standard pre-norm applies RMSNorm before every sub-layer, projecting inputs to unit magnitude. With shared weights, all loop iterations receive normalized (magnitude-erased) inputs. The network cannot distinguish iteration 1 from iteration 3. This makes identity collapse worse — pre-norm removes the magnitude signal that shared weights would need to behave differently per iteration.

Xiong et al. (2020, Theorem 1) showed pre-norm yields well-behaved gradients $O(d\sqrt{\ln d / L})$ but did not analyze the cost of erasing magnitude for weight-shared architectures. Run C' confirmed: with Birkhoff mixing but standard pre-norm, all mixing alphas collapsed to ~0.48 (uniform) and MLP scale dropped to 0.2–0.3.

> Xiong, R. et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML 2020. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)

## 2. Technique 1: Birkhoff-Constrained Residual Mixing

**Problem.** Unconstrained residual mixing (learned 2-vector weighting of residual and skip streams) has unbounded spectral norm. Xie et al. (2025, §3.1) demonstrated this at scale: unconstrained Hyper-Connections (Zhu et al., 2025, §2) produce signal gain exceeding 3000× in a 27B-parameter model. With looping, this gain compounds exponentially across iterations.

**Solution.** Constrain mixing to the Birkhoff polytope $B_n$ — the set of $n \times n$ doubly stochastic matrices. By the Birkhoff-von Neumann theorem (Birkhoff, 1946), every such matrix is a convex combination of permutation matrices. For $n=2$ streams (residual $x$ and skip $x_0$), $B_2$ has exactly 2 vertices ($I$ and swap). Any doubly stochastic mixing is $\alpha \cdot x + (1-\alpha) \cdot x_0$ with $\alpha \in [0, 1]$. The implementation parameterizes $\alpha = \sigma(\text{logit})$ — an exact parameterization of $B_2$ with spectral norm $\leq 1$ by construction.

This follows the HC → mHC → mHC-lite simplification chain. mHC (Xie et al., 2025, §4.1) uses iterative Sinkhorn-Knopp projection. mHC-lite (Yang & Gao, 2026, Theorem 3.1, §3.2) bypasses Sinkhorn via explicit convex combinations of permutations. The $n=2$ case is the minimal instance where a single sigmoid suffices.

**Result.** Q-gap reduced from +0.0024 (unconstrained, Run B') to +0.0019 (Birkhoff + peri-norm, Run C) at screening. Exponential 3-loop blowup eliminated (Run F: Q-gap +0.0019 at 3 loops).

> Zhu, D. et al. (2025). "Hyper-Connections." ICLR 2025. [arXiv:2409.19606](https://arxiv.org/abs/2409.19606)
> Xie, Z. et al. (2025). "mHC: Manifold-Constrained Hyper-Connections." [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)
> Yang, Y. & Gao, J. (2026). "mHC-lite: You Don't Need 20 Sinkhorn-Knopp Iterations." [arXiv:2601.05732](https://arxiv.org/abs/2601.05732)
> Birkhoff, G. (1946). "Three observations on linear algebra." Univ. Nac. Tucumán Rev. A, 5:147–151.

## 3. Technique 2: Output-LN (Peri-LN Variant for Recurrence)

**Problem.** Pre-norm erases input magnitude before every sub-layer, making all loop iterations indistinguishable to shared weights (§1c).

**Failed attempt.** Removing MLP input norm entirely — as in MoEUT (Csordas et al., 2024, §2.4), which places normalization only before sigmoid/softmax gates — caused NaN at step 2 with leaky_relu². MoEUT uses ReLU on the main data path, which has bounded gradient. The quadratic activation $\text{leaky\_relu}(x)^2$ has no implicit magnitude limiter.

**Fix.** Move normalization from MLP input to MLP output: $x + \text{Norm}(\text{MLP}(x))$. The MLP receives raw (unnormalized) activations, so weight matrices see different magnitude distributions across loop iterations. This produces different outputs per iteration. RMSNorm on the output bounds the contribution to the residual stream.

**Relation to Peri-LN.** Kim et al. (2025) define full Peri-LN as dual normalization: $x + \text{Norm}(\text{Module}(\text{Norm}(x)))$. The implementation omits the input norm, which the paper terms "Output-LN" (§3.2). Output-LN was chosen over full Peri-LN because the input norm would erase magnitude — exactly the signal shared weights need to differentiate loop iterations. Proposition 3.1 in Kim et al. analyzes the full Peri-LN scheme (dual norm). The gradient bounds it establishes do not directly cover Output-LN alone, though the output norm's damping factor $\|a\|$ is present in both variants.

**Result.** Run C (peri+birkhoff) vs Run C' (birkhoff only): −0.007 BPB. Alpha learned meaningful gradient (0.37→0.70 across layers) vs collapsed (0.45→0.50 uniform). Peri-norm is the main factor.

> Kim, J. et al. (2025). "Peri-LN: Revisiting Normalization Layer in the Transformer Architecture." ICML 2025. [arXiv:2502.02732](https://arxiv.org/abs/2502.02732)
> Csordas, R., Irie, K., Schmidhuber, J., Potts, C. & Manning, C. (2024). "MoEUT: Mixture-of-Experts Universal Transformers." NeurIPS 2024. [arXiv:2405.16039](https://arxiv.org/abs/2405.16039)

## 4. Technique 3: Capped Timestep Scaling

**Problem.** Without per-iteration conditioning, looped transformers have strict approximation limitations. Xu & Sato (2025, Lemma 4.1) prove that weight-tied feed-forward networks cannot drive approximation error to zero for varying target sequences. With timestep encoding, the expressivity gap closes completely (Theorem 4.2).

**Solution.** Learned per-iteration scale vectors $\gamma_{\text{attn}}^{(t)}, \gamma_{\text{mlp}}^{(t)}$ for attention and MLP residual contributions, clamped to $[-M, +M]$. This is a simplified FiLM conditioning (Perez et al., 2018) — scale-only, no shift — applied per loop iteration. Parameter cost: $2 \times \text{eff\_layers} \times 512 \approx 8\text{KB}$.

**Choice of cap $M = 4.0$.** This is an empirical choice, not theoretically derived. Three reasons: (1) Gammas are multiplicative modifiers on residual contributions. Uncapped, they could reintroduce the spectral norm amplification that Birkhoff mixing prevents. (2) $M = 4$ is large enough for meaningful per-iteration differentiation — one iteration's MLP contribution can be 4× another's — while small enough to keep downstream activations at similar scales across iterations. (3) Values in $[-4, 4]$ have ~0.001 precision in float16, preserving fine-grained specialization through the quantization passthrough. No ablation over cap values (e.g., 2.0 vs 4.0 vs 8.0 vs uncapped) was performed. This is an open question for future work. Screening Run D used uncapped timestep scaling at 2000 steps without issues, but full-scale runs only tested $M = 4$.

**Surprising finding.** Timestep scaling has near-zero effect on pre-quantization BPB (Run H vs I: 1.2578 vs 1.2580) but reduces quantization gap by 26–30% (H vs I: +0.0126 → +0.0088; J vs K: +0.0103 → +0.0076). The mechanism: capped gammas are stored as float16 passthrough parameters that bypass int8 quantization entirely. They provide per-iteration specialization that survives the quantization pipeline. In short, timestep scaling helps quantization, not training.

**Result.** Run K (best): post-quant 1.2659 BPB, Q-gap +0.0076 — vs prior 3-loop attempts that failed catastrophically (PR #363 measured ~900× quantization amplification over 3 cycles).

> Xu, K. & Sato, I. (2025). "On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding." ICML 2025. [arXiv:2410.01405](https://arxiv.org/abs/2410.01405)
> Perez, E., Strub, F., de Vries, H., Dumoulin, V. & Courville, A. (2018). "FiLM: Visual Reasoning with a General Conditioning Layer." AAAI 2018. [arXiv:1709.07871](https://arxiv.org/abs/1709.07871)

## 5. Supporting Technique: Prelude-Recurrent-Coda Architecture

First and last transformer layers perform fundamentally different functions — input encoding and output prediction — compared to middle layers that do iterative refinement. Forcing boundary layers into shared weights compromises both functions. Geiping et al. (2025) demonstrated this at scale with Huginn 3.5B: 2 prelude + 4 shared (×32 loops) + 2 coda layers, achieving 132 effective depth from 3.5B parameters.

**Result.** Run E (1+3×2+1, all fixes) vs Run D (4×2, all fixes): −0.016 BPB — the largest single architectural improvement in the ablation. Boundary layers need unique parameters.

> Geiping, J. et al. (2025). "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach." [arXiv:2502.05171](https://arxiv.org/abs/2502.05171)

## 6. Combined Effect and Key Results

### Screening (2000 steps, 1×H100)

| Run | Config | Post-Q BPB | Q-Gap | Δ vs B' (bare) |
|-----|--------|-----------|-------|-----------------|
| B' | 4×2 bare recurrence | 1.3637 | +0.0024 | — |
| C' | 4×2 + birkhoff only | 1.3660 | +0.0024 | +0.002 |
| C | 4×2 + peri + birkhoff | 1.3587 | +0.0020 | −0.005 |
| D | 4×2 + peri + birk + timestep | 1.3584 | +0.0019 | −0.005 |
| E | 1+3×2+1 all fixes | 1.3428 | +0.0019 | −0.021 |
| F | 1+2×3+1 all (3 loops) | 1.3622 | +0.0019 | −0.002 |

Birkhoff alone hurts (C' > B'). Peri-norm is the main factor (C − C' = −0.007). Prelude-coda is the largest single win (E − D = −0.016). Three loops are viable for the first time (F: Q-gap +0.0019, not exponential).

### Full-Scale (600s, 8×H100)

| Run | Config | Eff. Layers | Pre-Q BPB | Post-Q BPB | Q-Gap |
|-----|--------|-------------|-----------|------------|-------|
| H | 1+4×2+1 peri+birk | 10 | 1.2578 | 1.2704 | +0.0126 |
| I | 1+4×2+1 peri+birk+ts(cap4) | 10 | 1.2580 | 1.2668 | +0.0088 |
| J | 1+4×3+1 peri+birk (3 loops) | 14 | 1.2567 | 1.2670 | +0.0103 |
| **K** | **1+4×3+1 peri+birk+ts(cap4)** | **14** | **1.2583** | **1.2659** | **+0.0076** |

**Headline result.** Run K achieves 14 effective layers from 6 unique blocks with Q-gap +0.0076. This is the first viable 3-loop depth recurrence in competition history, vs prior results showing catastrophic failure at 3+ loops. Timestep scaling reduces Q-gap by 26–30% on both 2-loop and 3-loop configurations. It helps quantization, not training.

## 7. Technique 4: FiLM Bias (Per-Iteration Shift Vectors)

**Problem.** Capped timestep scaling (§4) provides per-iteration scale vectors $\gamma^{(t)}$ but no shift. Standard FiLM conditioning (Perez et al., 2018) uses both scale and shift: $\text{FiLM}(x) = \gamma \odot x + \beta$. The missing $\beta$ limits per-iteration expressivity — scaling alone cannot shift the operating point of downstream layers.

**Solution.** Add per-iteration bias vectors $\beta_{\text{attn}}^{(t)}, \beta_{\text{mlp}}^{(t)}$ alongside existing gammas. Initialized to zeros (no effect at initialization), not clamped (unlike gammas which are capped at ±4.0), stored as FP16 passthrough parameters that bypass int8 quantization. Parameter cost: $2 \times \text{eff\_layers} \times 512 \approx 8\text{KB}$ additional.

**Result.** FiLM bias gives a consistent −0.003 post-Q BPB improvement at both loop counts:

| Comparison | Without bias | With bias | Delta |
|------------|-------------|-----------|-------|
| 2 loops (s2_I vs s3_N) | 1.2668 | 1.2641 | −0.0027 |
| 3 loops (s2_K vs s3_O) | 1.2659 | 1.2625 | −0.0034 |

The effect is independent of loop count (~0.003 in both cases), confirming that bias provides additive benefit on top of gammas. No throughput penalty: step_avg is 42.48ms (s3_N) vs 41.92ms (s2_I) at 2 loops, and 58.18ms (s3_O) vs 59.28ms (s2_K) at 3 loops. Negligible artifact overhead (+0.03MB).

> Perez, E., Strub, F., de Vries, H., Dumoulin, V. & Courville, A. (2018). "FiLM: Visual Reasoning with a General Conditioning Layer." AAAI 2018. [arXiv:1709.07871](https://arxiv.org/abs/1709.07871)

## 8. Attention-Only Sharing: Validating Per-Iteration MLP Differentiation

**Hypothesis.** If shared weights are the bottleneck, which component benefits more from being unique per iteration — attention or MLP? ALBERT (Lan et al., 2020, §4.4) found that sharing attention parameters across layers has negligible effect on downstream tasks, while sharing FFN parameters causes most of the degradation. This suggests that attention weights learn position-agnostic patterns, while FFN weights need layer-specific specialization.

**Experiment.** s3_L uses attention-only sharing: 4 `SharedAttnLayer` modules (shared across loop iterations) paired with 8 `UniqueMLP` modules (one per virtual position per loop). This gives each iteration distinct feedforward capacity while reusing attention weights.

**Result.** s3_L achieves **1.2406 post-Q BPB** — the best result in the entire ablation series — beating full sharing (s2_I: 1.2668) by −0.026 BPB. This is a massive improvement, larger than any other single technique.

| Metric | s2_I (full share) | s3_L (attn-only share) | Delta |
|--------|-------------------|------------------------|-------|
| Post-Q BPB | 1.2668 | 1.2406 | −0.0262 |
| Q-gap | 0.0088 | 0.0073 | −0.0015 |
| Params | 11.55M | 15.75M | +4.20M |
| Artifact | 10.77MB | 14.65MB | +3.88MB |
| step_avg | 41.92ms | 42.60ms | +0.68ms |

**Diagnostics confirm per-iteration specialization.** Unique MLPs develop aggressive per-position scales (148–260 range vs 157–177 for full sharing). Shared attention alphas differentiate more (0.45–0.78 vs 0.43–0.61), suggesting that unique MLPs enable the shared attention to learn more distinct mixing behaviors.

**Abandoned.** Despite the BPB win, attention-only sharing is impractical for competition use:
1. **Artifact cost:** 14.65MB leaves only ~1.35MB headroom — insufficient for integrating SOTA features.
2. **torch.compile limitation:** The 3-loop variant (s3_M, 12 UniqueMLP modules) crashes `torch.compile(fullgraph=True)` during AOT autograd tracing with `RuntimeError: tensor does not have a device`. The 2-loop variant (8 modules) compiles fine. The model works without compile (verified via smoke test), but the throughput penalty (~3× slower) makes it uncompetitive.

**Takeaway.** The concept — per-iteration MLP differentiation — is validated. The implementation — full unique MLP copies — is too expensive. A cheaper mechanism is needed.

> Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P. & Soricut, R. (2020). "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations." ICLR 2020. [arXiv:1909.11942](https://arxiv.org/abs/1909.11942)

## 9. Toward Cheap Per-Iteration Specialization

s3_L proved that per-iteration MLP differentiation is essential (−0.026 BPB). But unique MLPs cost ~4MB in artifact size. The question is: can we achieve most of the differentiation at a fraction of the parameter cost?

### Key insight: control the input, not the weights

A unique MLP per iteration gives each loop a distinct feedforward function $f_v(x)$. But the same effect can be approximated by giving the shared MLP a distinct *input* per iteration: $f(\text{transform}_v(x))$. If the per-iteration transform is cheap, the total parameter cost drops dramatically.

### Evidence from literature

**MoEUT** (Csordás et al., 2024, §2.4): For Mixture-of-Experts Universal Transformers, "peri-layernorm" (normalization placement around sub-layers) is critical for competitive performance. The paper finds that normalization controls what the shared weights see, which is more important than the weights themselves being unique. This aligns with the Output-LN finding (§3): normalization placement is the key lever for recurrence.

**BitFit** (Ben-Zaken et al., 2022, §3): When fine-tuning BERT by training only bias terms, LayerNorm parameters change more than any other component — even more than attention or FFN biases. This suggests that normalization parameters have outsized influence on layer behavior, making them an efficient target for per-iteration specialization.

**Relaxed Recursive Transformers** (Bae et al., 2025, §3.2): Per-iteration LoRA adapters on shared transformer weights recover 99.7% of non-shared performance at 1/3 the parameters. The paper demonstrates that low-rank per-iteration corrections are sufficient — full unique copies are overkill.

### Planned parameter-efficient stack

| Component | Per-iteration cost | Total (14 virtual positions) | Role |
|-----------|-------------------|---------------------------|------|
| Unique input norms (attn_in + mlp_in) | 2 × 512 = 1024 params | 14 × 2 × 512 = 14,336 params = 28KB FP16 | Control what shared weights see |
| Depth embeddings | 512 params | 14 × 512 = 14KB FP16 | Positional identity per iteration |
| Timestep gammas | 2 × 512 = 1024 params | 14 × 2 × 512 = 28KB FP16 | Per-iteration scale |
| Timestep betas | 2 × 512 = 1024 params | 14 × 2 × 512 = 28KB FP16 | Per-iteration shift |
| **Total** | | **~110KB FP16 passthrough** | |

This leaves ~4.8MB headroom (from an estimated ~11.2MB artifact) for SOTA feature integration — vs only ~1.35MB with unique MLPs.

**Depth embedding subsumes Q/K bias.** Adding a depth embedding $e_v$ to the input before attention: $W_q(x + e_v) = W_q \cdot x + W_q \cdot e_v$. The second term acts as a learned per-iteration query/key bias, providing positional differentiation within the attention mechanism without additional parameters beyond the embedding itself.

> Csordás, R., Irie, K., Schmidhuber, J., Potts, C. & Manning, C. (2024). "MoEUT: Mixture-of-Experts Universal Transformers." NeurIPS 2024. [arXiv:2405.16039](https://arxiv.org/abs/2405.16039)
> Ben-Zaken, E., Goldberg, Y. & Ravfogel, S. (2022). "BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models." ACL 2022. [arXiv:2106.10199](https://arxiv.org/abs/2106.10199)
> Bae, S., Ko, J., Song, H. & Yun, S.-Y. (2025). "Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA." ICLR 2025. [arXiv:2410.20672](https://arxiv.org/abs/2410.20672)

## 10. Series 4 Results: Learned Depth Embeddings and Unique Input Norms

**Hypothesis.** Per-iteration MLP differentiation is critical (s3_L: −0.026 BPB), but unique MLPs are too expensive (~4MB). Cheap per-iteration input controls — learned depth embeddings and unique input norms — should recover much of the benefit at ~110KB total cost (§9).

**Experiment.** Four runs tested combinations of learned depth embeddings (512-dim vectors added to input before attention and MLP, initialized to zeros) and unique input norms (per-iteration RMSNorm parameters for attention and MLP inputs), all with FiLM bias enabled:

| Run | Config | Eff. Layers | Pre-Q BPB | Post-Q BPB | Q-Gap |
|-----|--------|-------------|-----------|------------|-------|
| P | 1+4×2+1 learned depth+norms+bias | 10 | 1.2579 | 1.2663 | +0.0084 |
| Q | 1+4×3+1 learned depth+norms+bias | 14 | 1.2574 | 1.2643 | +0.0069 |
| R | 1+4×3+1 learned depth only+bias | 14 | 1.2566 | 1.2639 | +0.0073 |
| S | 1+4×3+1 norms only+bias | 14 | 1.2560 | 1.2629 | +0.0069 |

**Baseline comparison.** FiLM bias alone (s3_O: 1.2625, Q-gap +0.0078) outperforms all Series 4 runs on post-Q BPB. The best Series 4 result (Run S: 1.2629) is +0.0004 worse than s3_O despite having additional parameters.

**Negative result: neither technique improves over FiLM bias alone.** This was unexpected given the §9 analysis. Three factors explain the failure:

### 10a. Learned Depth Embeddings Remained Near Zero

Depth embeddings were initialized to zeros and learned during training. After full-scale training (600s, 8×H100), embedding RMS values were 0.006–0.010 — essentially still near initialization. The embeddings did not learn meaningful per-iteration identity within the training budget. This contrasts with the theoretical expectation: Xu & Sato (2025, Theorem 4.2) prove that timestep encoding closes the expressivity gap, but their analysis assumes the encodings carry sufficient signal. Near-zero learned embeddings provide negligible signal.

The root cause is likely the interaction of initialization (zeros), learning rate, and training duration. With ~10k optimization steps and a cosine-decayed learning rate, small gradients on the depth embeddings never accumulated enough to produce meaningful values. This is a fundamental limitation of learned embeddings in short-training regimes.

### 10b. Throughput Penalty as Primary Harm Mechanism

The additional per-iteration parameters introduced 6–15% throughput overhead, costing 600–1700 training steps compared to the FiLM-bias-only baseline. In a wallclock-capped competition (600s), fewer steps means less training. The marginal specialization benefit of near-zero depth embeddings and barely-differentiated norms was overwhelmed by the lost training steps.

### 10c. Unique Input Norms Failed to Differentiate

Unique per-iteration RMSNorm parameters were expected to control what the shared MLP sees at each iteration (§9, "control the input, not the weights"). In practice, the MLP gains barely moved from 1.0 across iterations, indicating that the norms did not learn meaningfully different scaling. The Output-LN architecture already provides magnitude differentiation by letting the MLP see unnormalized inputs (§3). Adding per-iteration input norms on top of this provided no additional differentiation — the mechanism was redundant with Output-LN.

### 10d. Positive Finding: Q-Gap Improvement

Despite hurting BPB, the additional per-iteration parameters did reduce quantization gap: Q-gap 0.0069–0.0073 across Series 4 runs, vs 0.0078 for FiLM bias alone (s3_O). The extra FP16 passthrough parameters provide more degrees of freedom that survive int8 quantization. However, this Q-gap benefit is insufficient to overcome the BPB regression from throughput loss.

### 10e. Implication: The MLP Needs Different Weights, Not Different Inputs

The 0.026 BPB gap between full sharing (s2_I: 1.2668) and unique MLPs (s3_L: 1.2406) cannot be closed by cheap per-iteration input controls. Runs P–S demonstrate that even with depth embeddings, unique norms, and FiLM conditioning combined, the shared MLP produces nearly identical outputs across iterations. The MLP genuinely needs different weights per iteration — not just different inputs — to achieve the specialization that s3_L demonstrated.

### 10f. Next Test: Sinusoidal Depth Encodings

The natural next experiment is replacing learned depth embeddings with **sinusoidal depth encodings** following the Universal Transformer (Dehghani et al., 2019, §2.1). Sinusoidal encodings address all three failure modes of learned embeddings:

1. **Full-strength from step 0:** Fixed sinusoidal patterns provide immediate per-iteration identity without needing to be learned. No dependence on learning rate, initialization, or training duration.
2. **Zero parameter cost:** Computed analytically, not stored. Zero artifact overhead.
3. **Zero throughput overhead:** No additional parameters to backpropagate through. No training step penalty.

The Universal Transformer adds sinusoidal timestep embeddings $T_t$ at each recurrence step, where $T_t$ uses the same sinusoidal formula as positional encodings but indexed by iteration count rather than sequence position. This provides orthogonal identity signals across iterations with bounded magnitude.

> Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J. & Kaiser, L. (2019). "Universal Transformers." ICLR 2019. [arXiv:1807.03819](https://arxiv.org/abs/1807.03819)

## 11. Series 5: Sinusoidal Depth Encoding

**Experiment.** Run T replaces learned depth embeddings (Series 4) with deterministic sinusoidal depth encodings following the Universal Transformer (Dehghani et al., 2019, §2.1). The encoding uses the same sinusoidal formula as positional encodings but indexed by iteration count rather than sequence position: $\text{enc}(v, i) = \sin(v / b^{i/d})$ where $v$ is the virtual layer index, $b = 10000$ is the base frequency, $i$ is the dimension index, and $d$ is the model dimension. These are computed analytically at initialization and stored as a non-persistent buffer — zero learnable parameters, zero artifact cost.

| Run | Config | Eff. Layers | Pre-Q BPB | Post-Q BPB | Q-Gap | Steps | step_avg | Artifact |
|-----|--------|-------------|-----------|------------|-------|-------|----------|----------|
| T | 1+4×3+1 sinusoidal depth+bias | 14 | 1.2551 | 1.2624 | +0.0073 | 10195 | 58.86ms | 10.73MB |

### 11a. Comparison vs FiLM Bias Alone (s3_O)

| Metric | s3_O (no depth) | s5_T (sinusoidal) | Delta |
|--------|-----------------|-------------------|-------|
| Post-Q BPB | 1.2625 | 1.2624 | −0.0001 |
| Pre-Q BPB | 1.2547 | 1.2551 | +0.0004 |
| Q-gap | +0.0078 | +0.0073 | −0.0005 |
| step_avg | 58.18ms | 58.86ms | +0.68ms |

BPB is essentially identical (−0.0001 post-Q), confirming that per-iteration identity signaling provides no meaningful BPB improvement when FiLM gammas and betas are already present — they already provide sufficient per-iteration differentiation. The Q-gap improves marginally (0.0078 → 0.0073), likely because the sinusoidal encoding adds a fixed per-iteration bias to the input that slightly reduces the variance seen by shared weights, making quantization more stable.

### 11b. Comparison vs Learned Depth Embeddings (s4_R)

| Metric | s4_R (learned depth) | s5_T (sinusoidal) | Delta |
|--------|---------------------|-------------------|-------|
| Post-Q BPB | 1.2639 | 1.2624 | −0.0015 |
| Pre-Q BPB | 1.2566 | 1.2551 | −0.0015 |
| Q-gap | +0.0073 | +0.0073 | 0.0000 |
| step_avg | 62.56ms | 58.86ms | −3.70ms |
| Steps | 9592 | 10195 | +603 |

Sinusoidal beats learned by 0.0015 BPB. The mechanism is entirely throughput: sinusoidal encoding has zero backpropagation cost (non-persistent buffer, no gradients), saving 3.70ms per step and enabling 603 additional training steps within the 600s wallclock cap. Q-gap is identical (0.0073), confirming that both provide equivalent per-iteration identity signal for quantization purposes — the difference is purely in training efficiency.

### 11c. Conclusion

Sinusoidal depth encoding is free and should be kept on as default. It doesn't help BPB but provides marginal Q-gap benefit and costs nothing. The model already gets per-iteration identity from FiLM gammas/betas — depth encoding is redundant for differentiation but harmless.

This resolves the depth encoding question from §10f. The validated technique stack for SOTA integration is: **Output-LN + Birkhoff mixing + FiLM scale+shift (gammas+betas) + sinusoidal depth encoding**. The best full-sharing configuration is s5_T (1.2624 post-Q BPB, Q-gap +0.0073, 10.73MB artifact).

> Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J. & Kaiser, L. (2019). "Universal Transformers." ICLR 2019. [arXiv:1807.03819](https://arxiv.org/abs/1807.03819)
