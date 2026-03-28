# Asymmetric Squared Unit (ASQU): Increasing Capacity via Learned Per-Channel Activations

## Overview

Neural networks typically use a single shared activation function across all channels. This implicitly assumes that all neurons should exhibit the same nonlinear behavior.

We explore relaxing this assumption by introducing **ASQU (Asymmetric Squared Unit)**, a simple per-channel activation that allows each feature dimension to learn its own asymmetric response.

---

## Motivation

Activation functions (non-gated) are almost always:

- fixed (e.g. ReLU, GELU), or  
- globally parameterized (e.g. PReLU with a single slope)  

In both cases, all channels share the same nonlinearity.

However, different features may benefit from different activation behavior. For example:

- some channels may benefit from strong suppression of negative inputs (ReLU-like behavior)  
- others may benefit from more magnitude-based responses, activating on large inputs regardless of sign  
- others may benefit from allowing negative inputs to contribute with inverted sign  

Static activations impose the same behavior across all neurons, which is unnecessarily restrictive.

By allowing the negative branch to be learned per channel, each neuron can specialize its response. In practice, this allows a continuum of behaviors:

- `β_i ≈ 0` → ReLU²-like (suppress negatives)  
- `β_i > 0` → magnitude-sensitive activation  
- `β_i < 0` → signed response to negative inputs  

ASQU enables this flexibility with minimal overhead.

---

## Method: ASQU

ASQU builds on the squared activation (ReLU²), introducing a learned per-channel scaling for the negative branch:
f(x) = x^2 if x > 0
f(x) = β_i x^2 if x ≤ 0

- `β_i` is a learned parameter for each channel  
- adds minimal parameter overhead  
- preserves the simplicity and stability of squared activations  

ASQU can be viewed as a per-channel generalization of asymmetric squared activations.

---

## Results

We evaluate ASQU under the same 10k step training setup as the baseline.

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

ASQU provides a consistent improvement over both ReLU² and other fixed-slope asymmetric activations.

---

## Additional Experiments

### Beta Analysis
Empirically, we observe that the mean value of β typically converges to roughly around 0.5 (though this also depends significantly on the initialization of beta), which helps explain the effectiveness of fixed-slope asymmetric activations such as LeakyReLU².

However, there is substantial variation across channels: some β values become moderately negative, while others grow larger than 1. This suggests that while a global slope is a strong baseline, different features benefit from distinct activation behaviors that a global, fixed parameterization cannot capture.


### Learned Exponent

We explored learning the exponent instead of fixing it to 2. While this did not consistently improve final performance and introduced additional overhead, it revealed a consistent depth-dependent pattern:

- early layers: exponent ≈ 1.4  
- middle layers: exponent ≈ 1.8  
- late layers: exponent ≈ 2.2  

This suggests that different layers may benefit from different degrees of nonlinearity, with deeper layers favoring sharper activations.

---
