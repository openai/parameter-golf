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
