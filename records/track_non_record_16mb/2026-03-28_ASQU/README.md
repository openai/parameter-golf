# Asymmetric Squared Unit (ASQU): Increasing Capacity via Learned Per-Channel Activations

## Overview

Neural networks typically use a single shared activation function across all channels. This implicitly assumes that all neurons should exhibit the same nonlinear behavior.

We explore relaxing this assumption by introducing **ASQU (Asymmetric Squared Unit)**, a simple per-channel activation that allows each feature dimension to learn its own asymmetric response.

ASQU outperforms the current strong 10-minute-track activation baseline, fixed-slope LeakyReLU², across all three seeds in the fixed-step evaluation.

It is not used in the timed-track stack because the learned `β_i` gradient adds an extra kernel launch, and the resulting throughput cost was not justified under the 10-minute constraint.

---

## Motivation

Activation functions often apply the same nonlinear behavior across all channels.

In this setting, the strongest activation baseline was fixed-slope LeakyReLU², which improves over ReLU² by allowing negative inputs to contribute through a shared slope. However, that slope is still hard-coded and shared across all feature dimensions.

This assumes that all channels benefit from the same asymmetric response.

ASQU relaxes this assumption by allowing each channel to specialize its negative-branch behavior. Some channels may benefit from suppressing negative inputs, while others may benefit from responding to large inputs regardless of sign, or from allowing negative inputs to contribute with a different sign or magnitude.

---

## Method

ASQU builds on ReLU² by adding a learned per-channel scaling parameter for the negative branch, similar in spirit to PReLU.

```text
f_i(x) = x^2        if x > 0
f_i(x) = β_i x^2    if x ≤ 0
```

where:

- `β_i` is a learned parameter for channel `i`
- the positive branch matches ReLU²
- the negative branch is learned independently per channel

This gives ASQU a continuum of activation behaviors:

- `β_i ≈ 0`: ReLU²-like behavior, suppressing negative inputs
- `β_i > 0`: magnitude-sensitive behavior, where large negative inputs can activate positively
- `β_i < 0`: negative inputs produce modulated negative outputs

ASQU can be viewed as a squared PReLU-style activation: ReLU² provides the squared positive branch, while the learned `β_i` gives each channel control over its negative response.

---

## Pseudocode

```python
class ASQU:
    # one learned scalar per channel
    beta = learned_vector(dim)

    def forward(x):
        x2 = x ** 2
        return where(x > 0, x2, beta * x2)
```

---

## Setup

- Fixed 10k training steps
- Same setup as the baseline
- ASQU replaces the standard ReLU² activation
- Minimal parameter overhead from one learned `β_i` per channel

---

## Results

All runs use identical settings across three seeds, building off of the original naive baseline.

| Model | Seed | Pre-quant BPB ↓ | Post-quant BPB ↓ | Size (bytes) |
|-------|------|----------------:|-----------------:|-------------:|
| ReLU² | 1337 | 1.2262 | 1.2328 | 15861272 |
| LeakyReLU² (0.5) | 1337 | 1.2243 | 1.2315 | 15861749 |
| ASQU | 1337 | **1.2236** | **1.2296** | 15895013 |
| ReLU² | 42 | 1.2276 | 1.2343 | 15856563 |
| LeakyReLU² (0.5) | 42 | 1.2247 | 1.2315 | 15862578 |
| ASQU | 42 | **1.2240** | **1.2309** | 15894743 |
| ReLU² | 2025 | 1.2253 | 1.2321 | 15853892 |
| LeakyReLU² (0.5) | 2025 | 1.2234 | 1.2302 | 15858384 |
| ASQU | 2025 | **1.2225** | **1.2295** | 15892158 |
| **Average (ReLU²)** | — | 1.2264 | 1.2331 | 15857242 |
| **Average (LeakyReLU²)** | — | 1.2241 | 1.2311 | 15860870 |
| **Average (ASQU)** | — | **1.2234** | **1.2300** | 15893971 |

ASQU provides a consistent improvement over both ReLU² and fixed-slope LeakyReLU².

---

## Additional Experiments

### Beta Analysis

The learned `β_i` values typically have a mean around 0.5, though this depends on initialization. This helps explain why fixed-slope asymmetric activations such as LeakyReLU² are already strong baselines.

However, there is substantial variation across channels. Some `β_i` values become moderately negative, while others grow larger than 1. This suggests that different features benefit from distinct activation behavior that a single shared slope cannot capture.

### Learned Exponent

I also explored learning the activation exponent instead of fixing it to 2. This did not consistently improve final performance and introduced additional overhead, but it showed a consistent depth-dependent pattern:

- early layers: exponent ≈ 1.4
- middle layers: exponent ≈ 1.8
- late layers: exponent ≈ 2.2

This suggests that different layers may benefit from different degrees of nonlinearity, with deeper layers favoring sharper activations.

---

## Notes on Evaluation Setting

This PR evaluates ASQU under a fixed 10k step budget to isolate architectural effects from slight potential differences in data exposure. This gives a cleaner comparison when studying small changes such as activation functions.
