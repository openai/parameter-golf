# Mixture of Convolutions (MoC): Token-Adaptive Short Convolutions via Kernel Mixtures

## Overview

Short convolutions are highly effective in this regime, providing strong improvements at minimal parameter cost. However, standard short convolution applies the same kernel to every token, regardless of identity or context.

We introduce **Mixture of Convolutions (MoC)**, where each token constructs its own convolutional kernel as a mixture over a small shared set of basis kernels.

This enables **token-adaptive local operators** while preserving the parameter efficiency and training stability of standard short convolution.

---

## Motivation

Parameter efficient ways to incorporate local information is known to be important in this regime. Common techniques such as SmearGate and BigramHash incorporate local context in lightweight ways, and are widely used in strong baselines.

However, these methods are relatively limited: SmearGate performs a simple gated mixing with the previous token, and BigramHash injects local token identity only once at the input.

In contrast, short convolution provides a significantly more expressive local operator, applied at every layer and directly within QKV projections.

This suggests that stronger local operators are beneficial.

However, standard short convolution is still **static**: all tokens use the same kernel, regardless of identity or context.

We instead explore making these local operators **dynamic**.

---

## From Static to Dynamic Convolution

A natural approach is to generate a unique convolutional kernel for each token using a learned projection.

However, we found that this approach performs poorly in practice. We hypothesize that this parameterization is too expressive and difficult to optimize under the same training budget.

This suggests the need for an intermediate design between:

- fixed kernels (too rigid)
- fully generated kernels (too flexible)

---

## Method: Mixture of Convolutions (MoC)

MoC addresses this by introducing a small set of **basis kernels**, and predicting only how to mix them per token.

### Kernel Bank

We learn a set of `k` convolutional kernels:

- shape: `(k, dim, kernel_size)`
- shared across all tokens

---

### Token-wise Routing

For each token, we compute mixture weights:

    α_t = softmax(gate(z_t) / τ)

- `z_t` is the same hidden state used to generate QKV
- `τ` is a learned temperature controlling sharpness

---

### Dynamic Kernel Construction

Each token constructs its own kernel as a mixture over the basis:

    W_t = Σ_i α_t,i * K_i

This kernel is then applied using a standard causal convolution.

---

### Special Case: Standard Short Convolution

When `k = 1`, MoC reduces exactly to standard short convolution.

MoC is therefore a **strict generalization** of static short convolution.

---

## Interpretation

MoC can be viewed as a middle ground between:

- fixed local operators (standard convolution)
- fully generated operators (projection-based kernels)

By constraining kernels to lie in a shared basis, MoC provides:

- enough flexibility to adapt per token
- enough structure to remain stable during optimization

This balance appears to be critical for good performance.

---

## Results

All runs are trained for 10k steps under identical settings across three seeds.

MLP expansion is adjusted to keep models within the parameter budget: baseline uses 2.00×, short convolution (k=1) uses 1.99×, and MoC (k=8) uses 1.93×.

| Model | Seed | Pre-quant BPB ↓ | Post-quant BPB ↓ | Size (bytes) |
|-------|------|----------------:|-----------------:|-------------:|
| Baseline | 1337 | 1.2262 | 1.2328 | 15861272 |
| Short Conv (k=1) | 1337 | 1.2201 | 1.2261 | 15866404 |
| MoC (k=8) | 1337 | **1.2148** | **1.2213** | 15883078 |
| Baseline | 42 | 1.2276 | 1.2343 | 15856563 |
| Short Conv (k=1) | 42 | 1.2199 | 1.2263 | 15864705 |
| MoC (k=8) | 42 | **1.2171** | **1.2235** | 15884351 |
| Baseline | 2025 | 1.2253 | 1.2321 | 15853892 |
| Short Conv (k=1) | 2025 | 1.2202 | 1.2270 | 15863930 |
| MoC (k=8) | 2025 | **1.2147** | **1.2208** | 15878813 |
| **Average (Baseline)** | — | 1.2264 | 1.2331 | 15857242 |
| **Average (Short Conv)** | — | 1.2201 | 1.2264 | 15865013 |
| **Average (MoC)** | — | **1.2155** | **1.2219** | 15882081 |


Short convolution provides a large improvement over baseline, and MoC improves further by replacing the static kernel with a token-adaptive mixture over basis kernels.

---

## Practical Considerations
MoC is more expensive than standard short convolution due to per-token kernel composition, as it cannot utilize highly optimized existing short-conv implementations which assume static weights.

As a result, MoC is not currently competitive on the time-constrained leaderboard, but is evaluated here in the fixed-step setting.
