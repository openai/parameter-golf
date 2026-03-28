# PolyGLU: State-Conditional Activation Routing in Transformer Feed-Forward Networks

**Author**: Daniel Nobrega Medeiros (independent researcher)
**arXiv**: [2603.13347v1](https://arxiv.org/abs/2603.13347v1) (cs.LG, March 2026)
**Code**: https://github.com/danielxmed/PolyGLU
**Base model**: https://huggingface.co/tylerxdurden/PolyChromaticLM-1.0-base-0.6B
**Instruct model**: https://huggingface.co/tylerxdurden/PolyChromaticLM-1.0-instruct-0.6B

---

## Abstract

Biological neural systems employ diverse neurotransmitters — glutamate, GABA, dopamine, acetylcholine — to implement distinct signal-processing modalities within shared neural circuits. In contrast, modern transformers apply a single fixed activation function across all feed-forward neurons. We introduce **PolyGLU** (Polychromatic Gated Linear Unit), a drop-in replacement for SwiGLU that enables each FFN neuron to dynamically route among K=4 activation functions via a differentiable mechanism combining learned static preferences with input-conditioned gating, trained end-to-end with Gumbel-Softmax.

We train PolychromaticLM, a 597M-parameter transformer, on ~10B tokens using a single NVIDIA A100 GPU. Our key finding is **emergent routing behavior**: without any explicit sparsity loss or entropy regularization, the routing mechanism converges to near-deterministic activation selections (mean dynamic entropy = 0.030% of maximum), with a striking depth-dependent specialization pattern — early layers prefer GELU while deep layers strongly favor Tanh. Three layers maintain elevated routing entropy, suggesting computational flexibility points.

The routing architecture adds only **0.23% parameter overhead** (~1.4M parameters) and proves fully robust to supervised fine-tuning: routing entropy remains constant at ln(4) throughout 13,067 SFT steps. On standard benchmarks, PolychromaticLM achieves 62–89% of Qwen3-0.6B-Base performance despite training on 3,600× fewer tokens. All code, weights, and training infrastructure are released under Apache 2.0.

---

## 1. Introduction

Biological neural systems do not rely on a single signaling mechanism. Instead, they employ a diverse repertoire of neurotransmitters — each conferring distinct computational properties on the circuits they modulate. Glutamate provides fast excitatory transmission, GABA mediates inhibition, dopamine gates reward-modulated learning, and acetylcholine regulates attentional allocation. This diversity is a fundamental architectural feature that enables the brain to implement qualitatively different computations within a shared neural substrate.

Modern transformer architectures apply a single fixed activation function uniformly across all feed-forward neurons. The evolution from ReLU to GELU to SwiGLU has improved performance, but the fundamental assumption remains: **one activation function is optimal for all neurons at all depths for all inputs.**

We challenge this assumption with **PolyGLU**, a drop-in SwiGLU replacement that allows each FFN neuron to dynamically select among K=4 candidate activation functions — ReLU, Tanh, SiLU, and GELU — through a differentiable routing mechanism. Each neuron maintains a learned static preference over the activation palette, modulated by a lightweight gating network conditioned on the current hidden state. Routing decisions are made differentiable via Gumbel-Softmax with temperature annealing.

Our primary contribution is not the mechanism itself, but the **emergent behavior** it produces. When we train PolychromaticLM on ~10B tokens, we observe:

1. **Spontaneous routing convergence.** Without any explicit sparsity loss, entropy penalty, or load-balancing regularizer, the routing mechanism converges to near-deterministic selections (mean dynamic entropy = 0.030% of maximum). The model discovers that sparse, committed activation routing is preferable to soft mixing.

2. **Depth-dependent specialization.** A clear activation gradient emerges across the 28 transformer layers: early layers predominantly select GELU (probabilistic gating), while deep layers strongly favor Tanh (bounded compression). This learned specialization suggests that different network depths require different nonlinear transformations.

3. **Fine-tuning robustness.** During supervised fine-tuning on mathematical reasoning data (13,067 steps), routing entropy remains exactly constant at ln(4), indicating that the routing architecture cleanly separates "how to compute" from "what to compute."

All of this is achieved with only 0.23% parameter overhead (~1.4M routing parameters out of 597M total). The entire project was conducted on a single NVIDIA A100 80GB GPU at a total cost of ~$346.

---

## 2. Related Work

**Activation Functions.** The history traces from ReLU to GELU to SwiGLU — now the dominant FFN activation in LLaMA and Qwen. PolyGLU generalizes this line by replacing the fixed activation with a learned, input-conditioned selection among multiple candidates.

**Mixture of Experts.** Sparsely-activated models route entire tokens to different expert sub-networks. PolyGLU operates at a finer granularity: it routes **individual neurons to activation functions**. Notably, MoE models require explicit load-balancing losses to prevent routing collapse, whereas PolyGLU achieves near-deterministic routing **without any auxiliary loss**.

**Gumbel-Softmax.** Enables gradient-based optimization through discrete categorical choices via a continuous relaxation. We use it with temperature annealing from τ=1.0 (exploration) to τ=0.1 (commitment).

**Adaptive Computation.** PolyGLU differs from prior work in that routing is per-neuron and per-layer rather than global, and combines both static learned preferences and dynamic input conditioning.

---

## 3. Method

### 3.1 Background: SwiGLU

The SwiGLU feed-forward block computes:

```
SwiGLU(x) = [SiLU(x @ W_gate)] ⊙ (x @ W_up)
```

followed by a down-projection W_down. The SiLU activation is applied identically to every neuron in every layer.

### 3.2 PolyGLU Formulation

PolyGLU generalizes SwiGLU by replacing the fixed SiLU with a learned mixture of K activation functions:

```
PolyGLU(x) = [Σ_{k=1}^{K} g_k · σ_k(x @ W_gate)] ⊙ (x @ W_up)
```

where σ_k are the candidate activation functions and g_k are per-neuron routing weights.

We use **K=4 activation functions**, chosen to span qualitatively different nonlinear behaviors:

| k | Function | Property             | Biological Analogy             |
|---|----------|----------------------|--------------------------------|
| 0 | ReLU     | Hard threshold       | Glutamate (excitatory)         |
| 1 | Tanh     | Symmetric compression| GABA (inhibitory)              |
| 2 | SiLU     | Self-gated           | Dopamine (modulatory)          |
| 3 | GELU     | Probabilistic gate   | Acetylcholine (attentional)    |

### 3.3 Routing Mechanism

The routing weights g_k are computed from two components:

**Static preferences.** Each neuron j (of d_ff total) maintains a learnable preference vector α_j ∈ R^K, initialized to zero (uniform prior). These encode baseline activation affinities.

**Dynamic gating.** A lightweight MLP processes the mean-pooled hidden state h̄ = mean(x, dim=seq):

```
f(h̄) = W2 · ReLU(W1 @ h̄ + b1) + b2
```

where W1 ∈ R^{32×d_model} and W2 ∈ R^{K×32}. Per-activation scaling factors β ∈ R^K (initialized to 1.0) modulate the dynamic signal.

**Combined routing.** The full routing logits combine both components:

```
ℓ_k = α_k + β_k · f(h̄)_k
g_k = GumbelSoftmax(ℓ, τ)_k
```

**Temperature annealing:**

```
τ(t) = max(0.1, 1.0 − 0.9 · t/t_total)
```

At τ=1.0 (training start), routing is nearly uniform (exploration). At τ=0.1 (training end), routing is near-deterministic (commitment).

### 3.4 Integration into the Transformer Block

Each transformer block follows a pre-norm residual structure:

```
x ← x + GQA(RMSNorm(x))
x ← x + PolyGLU(RMSNorm(x))
```

PolyGLU is a drop-in replacement: W_gate, W_up, and W_down retain their standard dimensions.

### 3.5 Parameter Overhead Analysis

Per PolyGLU layer:
- α ∈ R^{d_ff × K}: 4,096 × 4 = 16,384 parameters
- β ∈ R^K: 4 parameters
- Gate network: Linear(1024→32) + Linear(32→4): 32,768+32+128+4 = 32,932 parameters
- **Total per layer: ~49,320 parameters**
- **Total (28 layers): ~1.4M parameters (0.23% of 597M)**

---

## 4. Experimental Setup

### 4.1 Model Architecture

| Parameter | Value |
|---|---|
| Total parameters | 597,153,888 |
| of which routing | ~1.4M (0.23%) |
| Hidden dimension (d_model) | 1,024 |
| FFN intermediate (d_ff) | 4,096 |
| Layers | 28 |
| Query / KV heads | 16 / 8 (GQA) |
| Head dimension | 64 |
| Context length | 4,096 tokens |
| Vocabulary | 151,669 (Qwen3 tokenizer) |
| Position encoding | RoPE (θ=10,000) |
| Normalization | RMSNorm (pre-norm) + QK-Norm |
| FFN activation | PolyGLU (K=4) |
| Weight tying | Embedding ↔ output head |

Residual connections to W_o (attention output) and W_down (FFN output) are scaled by `1/√(2·n_layers)`. Weight initialization follows N(0, 0.02), with α initialized to zero (uniform prior) and β initialized to ones.

### 4.2 Pre-Training

- **Data**: ~10.24B tokens (70% math, 25% STEM, 5% code)
- **Optimizer**: AdamW (β1=0.9, β2=0.95, ε=1e-8, weight decay 0.1)
- **Learning rate**: Cosine decay, 2,000-step warmup, peak 1e-4
- **Batch**: 524,288 tokens per effective batch (micro_batch=16, seq=4096, grad_accum=8)
- **Steps**: 19,531 (~10.24B tokens total)
- **Infrastructure**: Single A100 80GB, DeepSpeed ZeRO Stage 0, BFloat16
- **Throughput**: ~11,800 tokens/sec, ~12.5 days wall time

### 4.3 Supervised Fine-Tuning

- Dataset: nvidia/Nemotron-Math-v2 (~347K problems)
- 1 epoch (13,067 steps), LR 2e-5, τ frozen at 0.1
- Loss on assistant tokens only, ~18 hours (~$29.50)

---

## 5. Results

### 5.1 Pre-Training Convergence

Training loss decreased from 12.13 to 1.31 over 19,531 steps (89% reduction).

### 5.2 Emergent Routing Behavior

**The central finding: routing converges to near-deterministic activation selections without any explicit regularization.**

#### 5.2.1 Near-Deterministic Convergence

Mean dynamic routing entropy at convergence: **4.1 × 10⁻⁴** (only 0.030% of maximum ln(4) ≈ 1.386).

Most layers achieve entropy below 10⁻⁴ (effectively one-hot selections). Three layers stand out:
- **Layer 9**: entropy 2.5 × 10⁻⁴ — modestly elevated
- **Layer 16**: entropy 1.5 × 10⁻³ — partially specialized
- **Layer 17**: entropy 9.6 × 10⁻³ — highest in network (increased during final phase)

#### 5.2.2 Layer-Wise Activation Specialization

A clear depth-dependent gradient emerges:

- **Early layers (0–2)**: GELU dominates (~35–40%), with Tanh (~15–25%) and SiLU (~15–20%) secondary
- **Middle layers (3–14)**: Gradual transition. GELU remains plurality, SiLU grows to ~15–25%
- **Deep layers (15–27)**: **Tanh surges to 50–65%**, becoming the dominant activation

This is striking — the symmetric, bounded activation (Tanh) becomes preferred for deep representational processing.

#### 5.2.3 The Weight Decay Bug and Its Consequences

At step ~10,000, static preferences α were discovered to be under weight decay (inadvertently grouped with 2D weight matrices). This was corrected via optimizer state transplant. Critical finding: **even while α was suppressed, the dynamic gate network alone achieved near-deterministic routing** (0.58% of max entropy). This proves the dynamic pathway is sufficient.

### 5.3 Benchmark Performance

| Benchmark | Base | Random | Qwen3-0.6B |
|---|---|---|---|
| HellaSwag | 28.51 | 25.00 | 41.10 |
| ARC-Easy | 41.04 | 25.00 | 65.60 |
| ARC-Challenge | 22.27 | 25.00 | 33.90 |
| PIQA | 58.87 | 50.00 | 70.00 |
| WinoGrande | 52.17 | 50.00 | 58.50 |
| BoolQ | 61.13 | 50.00 | 69.70 |

On 6 benchmarks with published Qwen3-0.6B scores, PolychromaticLM achieves **62–89% of Qwen3's performance** despite training on **3,600× fewer tokens**.

### 5.4 Domain Perplexity

| Domain | Share | Perplexity | Bits/Token |
|---|---|---|---|
| Math | 70% | 3.56 | 1.83 |
| Code | 5% | 7.08 | 2.82 |
| STEM | 25% | 31.93 | 5.00 |

### 5.5 Fine-Tuning Stability

Routing entropy remained **exactly at ln(4) ≈ 1.386 throughout all 13,067 SFT steps**. The routing architecture is a permanent structural feature that completely survives fine-tuning.

---

## 6. Analysis and Discussion

### 6.1 Why Does Routing Converge Without Regularization?

The language modeling loss itself provides sufficient signal. The gradient with respect to routing weights g_k is:

```
∂L/∂g_k = ∂L/∂PolyGLU · [σ_k(x @ W_gate) ⊙ (x @ W_up)]
```

Each activation produces qualitatively different gradient signals. Over many steps, the optimizer discovers that committing to a specific activation produces cleaner, more consistent gradients. Temperature annealing amplifies this via a positive feedback loop.

### 6.2 The Static-Dynamic Separation

The dynamic routing pathway alone is sufficient. Even while α was suppressed:
1. The gate network (a simple 2-layer MLP) has sufficient expressive power for confident routing
2. The routing signal is primarily **contextual** — the network learns which activation to use based on current input
3. The static component α is functionally subordinate — it provides a warm-starting bias

### 6.3 Implications for Activation Function Design

The GELU-early, Tanh-deep specialization challenges the assumption that a single activation is optimal across all layers. This suggests:
- **Heterogeneous fixed activations**: Train PolyGLU, observe converged pattern, then "distill" into a standard model with different fixed activations per layer (zero inference cost)
- **Activation search**: PolyGLU patterns as data-driven activation function selection
- **Tanh in deep layers**: The bounded, symmetric output may provide implicit regularization preventing activation magnitude growth

### 6.4 Limitations

- **No SwiGLU baseline**: Budget constraints prevented training a vanilla SwiGLU model for controlled comparison
- **Scale**: All experiments at ~600M params / ~10B tokens. Scaling behavior remains open
- **Activation palette**: K=4 chosen heuristically. Systematic exploration of palette size and composition could yield improvements
- **Inference efficiency**: PolyGLU computes all K activations before selecting. For deployment, converged routing can be frozen into a static mapping

---

## 7. Conclusion

PolyGLU demonstrates that the "one activation fits all" assumption in modern transformers is suboptimal. Different network depths benefit from different nonlinear transformations, and gradient-based optimization can discover these specialization patterns given the freedom to do so.

Key numbers:
- 0.23% parameter overhead
- 0.030% of maximum entropy at convergence (near-deterministic routing)
- 62–89% of Qwen3-0.6B performance on 3,600× fewer tokens
- Zero routing entropy drift during fine-tuning
- Total project cost: ~$346 on a single A100

---

## Critical Takeaways for Parameter Golf Adaptation

1. **The routing mechanism converges naturally** — no special loss terms needed
2. **Gumbel-Softmax temperature annealing is essential** — start exploratory (τ=1.0), end committed (τ=0.1)
3. **The dynamic gate network is the key component** — it's a tiny MLP (d_model→32→K) that provides input-conditioned routing
4. **Static preferences (α) are secondary** — the dynamic pathway alone suffices. For a tiny model, we could simplify to dynamic-only routing
5. **Different depths genuinely prefer different activations** — even with just 10-11 layers, there may be a benefit
6. **Weight decay should NOT be applied to α** — it suppresses specialization. Exempt routing params from weight decay
7. **The routing params should NOT use Muon optimizer** — they are small/scalar-like, use Adam with the scalar learning rate
