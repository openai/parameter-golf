# Mixture of Convolutions (MoC): Token-Adaptive Short Convolutions via Kernel Mixtures

## Overview

Short convolutions are highly effective in this regime, providing strong improvements at minimal parameter cost. However, standard short convolution applies the same kernel to every token, regardless of identity or context.

We introduce **Mixture of Convolutions (MoC)**, where each token constructs its own convolutional kernel as a mixture over a small shared set of basis kernels.

This enables **token-adaptive local operators** while preserving the parameter efficiency and training stability of standard short convolution.

---

## Motivation

Parameter-efficient local context is important in this regime. Common techniques such as SmearGate and BigramHash incorporate local information in lightweight ways and are widely used in strong baselines.

These methods show that even small amounts of local context can be valuable. Short convolution takes this further by providing a more expressive local operator that can be applied at every layer and directly within QKV projections.

However, standard short convolution is still static: all tokens use the same kernel regardless of identity or context.

MoC removes this restriction by making the convolutional kernel token-adaptive while keeping it constrained to a small learned basis.

---

## Method

MoC introduces a bank of learned convolutional basis kernels.

The kernel bank has shape:

```text
(k, dim, kernel_size)
```

where:

- `k` is the number of basis kernels
- `dim` is the channel dimension
- `kernel_size` is the convolution width

For each token, MoC computes mixture weights over this kernel bank:

```text
α_t = softmax(gate(z_t) / τ)
```

where:

- `z_t` is the hidden state used to generate QKV
- `gate(z_t)` produces token-wise routing logits
- `τ` is a learned temperature controlling routing sharpness

Each token then constructs its own convolutional kernel:

```text
W_t = Σ_i α_t,i · K_i
```

This dynamically constructed kernel is applied as a causal convolution.

## Pseudocode

```python
class MixtureOfConvolutions:
    # shared basis of convolution kernels
    # k = basis/kernel index, K = causal kernel position
    basis = learned_tensor(k, dim, K)

    # token-wise router over basis kernels
    gate = linear(z_dim, k)

    # learned temperature controls routing sharpness
    temperature = learned_scalar()

    def forward(x, z):
        # local causal windows around each token
        windows = causal_windows(x, K)          # [batch, time, dim, K]

        # each token chooses a mixture over basis kernels
        alpha = softmax(gate(z) / temperature)  # [batch, time, k]

        # compose one convolution kernel per token
        dynamic_kernel = einsum(
            "btk,kdK->btdK",
            alpha,     # token-wise mixture weights over k basis kernels
            basis      # k basis kernels, each with dim channels and K positions
        )

        # apply each token's dynamic local operator
        return sum(windows * dynamic_kernel, dim=-1)
```

---

## Interpretation

MoC can be viewed as a middle ground between:

- **Fixed kernels:** stable and efficient, but not adaptive
- **Fully generated kernels:** highly flexible, but harder to optimize
- **Mixture-based kernels:** adaptive while remaining constrained to a learned basis

Directly generating kernels from a projection performed poorly in practice, likely because the parameterization was too flexible under the same training budget.

By constraining each token’s kernel to lie in the span of a small shared basis, MoC provides token-level flexibility while preserving much of the stability of standard short convolution.

---

## Special Case

Standard short convolution is the `k = 1` case of MoC.

With a single basis kernel, every token receives the same kernel, so the mixture reduces exactly to static short convolution.

MoC is therefore a generalization of short convolution.

---

## Setup

- Fixed 10k training steps
- Same setup as the baseline
- MoC applied as a token-adaptive short convolution
- MLP expansion adjusted to stay within the parameter budget

MLP multipliers:

- Baseline: 2.00×
- Short Conv (`k = 1`): 1.99×
- MoC (`k = 8`): 1.93×

---

## Results

All runs use identical settings across three seeds, building off of the original naive baseline.

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

## Stronger Baseline Validation

To check whether MoC still helps beyond the naive baseline setting, I also ran a second validation experiment on a stronger stack.

This setup scales up the original naive baseline with several components from the current 10-minute-track SOTA-style configuration:

- 11 layers
- 4× MLP expansion for the baseline
- SP8192 vocabulary
- stronger training hyperparameters

For this experiment, MoC uses a **3.88× MLP expansion** to stay within the parameter budget. MoC is applied both before QKV and before the MLP, and only the first 192 dimensions are used for the routing gate.

| Model | Seed | Val BPB ↓ |
|-------|------|----------:|
| Strong baseline | 1337 | 1.1304 |
| MoC | 1337 | **1.1180** |
| Strong baseline | 42 | 1.1312 |
| MoC | 42 | **1.1196** |
| Strong baseline | 2025 | 1.1318 |
| MoC | 2025 | **1.1193** |
| **Average strong baseline** | — | 1.1311 |
| **Average MoC** | — | **1.1190** |

MoC continues to provide a large improvement on the stronger baseline, reducing average validation BPB from **1.1311** to **1.1190**.

This suggests that the benefit of token-adaptive local operators is not limited to the naive baseline. In this stronger setting, applying MoC before both QKV and the MLP gives a substantial fixed-step gain, though the added routing and dynamic convolution overhead still makes throughput an important practical concern.



---

## Practical Considerations
MoC is more expensive than standard short convolution due to per-token kernel composition, as it cannot utilize highly optimized existing short-conv implementations which assume static weights.

As a result, MoC is not currently competitive on the time-constrained leaderboard, but is evaluated here in the fixed-step setting.
