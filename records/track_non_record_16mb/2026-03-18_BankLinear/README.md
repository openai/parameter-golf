# BankLinear: Compositional Weight Sharing via Learned + Random Basis Mixtures

## Overview

Under strict parameter budgets, allocating a unique set of weights per layer is inefficient. Standard transformers duplicate large weight matrices across depth, even though many layers learn structurally similar transformations.

Prior approaches attempt to address this through weight tying or recurrence, but these impose rigid constraints and typically underperform fully independent per-layer parameterization.

We instead pursue the same motivation — **parameter reuse across depth** — but with significantly greater flexibility.

---

## BankLinear

We introduce **BankLinear**, a linear layer that does not store weights per layer. Instead, each layer **constructs its weights as a mixture over a shared bank of matrices**.

We allocate:
- a **small set of learned basis matrices**
- a **larger set of fixed random projections**

In our configuration:
- 9 total layers  
- 3 learned basis matrices  
- 512 fixed random projections  

Each layer learns its own set of **mixing coefficients**, allowing it to compose a unique weight matrix:

  W^(l) = Σ_i α_i^(l) B_i

This replaces explicit per-layer weight storage with **compositional weight synthesis**.

---

## Motivation

This design can be viewed as a relaxed form of depth recurrence:

- Recurrence: reuse the *same* weights across layers  
- BankLinear: reuse a **shared basis**, but allow each layer to construct its own weights  

This avoids the rigidity of recurrence while retaining its parameter efficiency.

At the same time, it avoids the redundancy of fully independent layers by sharing structure across depth.

---

## Learned + Random Basis

A key component is the inclusion of a **large bank of fixed random projections**.

These are mixed using the same coefficients as the learned basis.

This serves two purposes:

- The **learned basis** captures reusable, important directions  
- The **random basis** provides a cheap, high-dimensional span  

Together, they allow the model to construct expressive weight matrices despite a small number of learned parameters.

---

## Initialization

Initialization of the mixing coefficients is important for stable performance.

A useful reference point is **depth recurrence**. If layers were assigned *hard* coefficients over the learned basis (e.g. blocks of layers using a single basis), this would recover a recurrent-depth transformer.

BankLinear instead uses a **soft relaxation** of this structure.

We initialize mixing coefficients with a **depth-aware profile**, where early, middle, and late layers are biased toward different learned bases, with smooth transitions between them. This encourages an initial division of labor across layers while still allowing full flexibility during training.

Without this initialization, performance degrades significantly.

---

## Integration

We apply BankLinear to attention projections:

- Q, K, V are constructed from shared banks  
- Each layer uses its own mixing coefficients  

This enables parameter sharing across depth while maintaining per-layer specialization.

---


## Results

All runs are trained for 10k steps under identical settings across three seeds.  
BankLinear replaces QKV projections, and saved parameters are reinvested into a larger MLP (2.65× vs 2.00× baseline).

| Model | Seed | Pre-quant BPB ↓ | Post-quant BPB ↓ | Size (bytes) |
|-------|------|----------------:|-----------------:|-------------:|
| Baseline | 1337 | 1.2262 | 1.2328 | 15861272 |
| BankLinear | 1337 | **1.2245** | **1.2311** | 15729669 |
| Baseline | 42 | 1.2276 | 1.2343 | 15856563 |
| BankLinear | 42 | **1.2239** | **1.2300** | 15734621 |
| Baseline | 2025 | 1.2253 | 1.2321 | 15853892 |
| BankLinear | 2025 | **1.2223** | **1.2280** | 15739602 |
| **Average (Baseline)** | — | 1.2264 | 1.2331 | 15857242 |
| **Average (BankLinear)** | — | **1.2236** | **1.2297** | 15734631 |


---

## Additional Experiments

We explored allocating parameters to layer-specific LoRA adapters, but found this to be less effective than reinvesting the saved capacity into larger MLP expansions.

We also experimented with increasing the granularity of BankLinear by learning mixing coefficients per attention head instead of per basis. This consistently degraded performance.

Applying BankLinear to output projections (i.e. layers that write directly into the residual stream) significantly degraded performance.

We hypothesize that this is due to the increased rigidity of the partially fixed parameterization. Errors in these projections directly pollute the residual stream and accumulate across layers. 
