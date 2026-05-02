# Hierarchical Shared Attention (HSA): Multi-Level Sharing Across Attention Heads

## Overview

Standard attention treats each head as independent, allocating separate parameters per head even though many heads learn similar or redundant features.

We introduce **Hierarchical Shared Attention (HSA)**, which explicitly models this redundancy by sharing features across heads at multiple levels. Instead of choosing between full sharing (MQA), grouped sharing (GQA), or independent heads (MHA), HSA combines all of them within a single hierarchy.

Rather than removing redundancy across attention heads, HSA models the shared structure that gives rise to it, allowing heads to reuse common features while still maintaining head-specific specialization.

This enables **structured parameter sharing across heads**, reducing redundant parameterization of shared features and KV-cache duplication while preserving expressivity.

---

## Motivation

Empirically, attention heads are not independent:

- many heads learn similar or overlapping patterns  
- pruning or merging heads often has limited impact (in models larger than these)  
- grouped-query attention (GQA) already exploits partial sharing  

However, existing approaches enforce a **single level of sharing**:

- MQA: all heads share the same features  
- GQA: heads share within fixed groups  
- MHA: no sharing  

This suggests a more flexible structure:

> some features should be shared across all heads, some across groups of heads, and some remain head-specific.

HSA is designed to model this structure explicitly.

---

## Method: Hierarchical Shared Attention

HSA constructs query/key/value projections using multiple levels of shared features.

Each level is defined by a pair `(g, d)`:

- `g`: number of groups (how many distinct feature sets exist at this level)  
- `d`: number of dimensions allocated to this level  

At each level:

- features are shared within groups of heads  
- smaller `g` → more sharing  
- larger `g` → more specialization  

The total head dimension is composed by concatenating features from all levels.

This allows different portions of the feature space to operate at different levels of sharing.

Shared features are modulated with a learned per-head scaling, allowing shared representations to specialize with minimal cost.

---

### Example Configuration

Consider the following configuration for key/value projections:

`kv_levels = [(1, 16), (2, 16), (4, 32)]`

which corresponds to 4 KV heads with a total head dimension of 64.

This decomposes the head features into three levels:

- `(1, 16)`  
  16 dimensions are shared across all heads (MQA-style)

- `(2, 16)`  
  16 dimensions are split into 2 groups, each shared across 2 heads (GQA-style)

- `(4, 32)`  
  32 dimensions are unique to each head (MHA-style)

The final head representation is formed by concatenating these components, so each head contains:

- shared features (global context)  
- group-shared features (partial specialization)  
- head-specific features (full specialization)  

In practice, we observe that more aggressive sharing is often effective for KV projections than for queries, though relatively little exploration has been done to determine optimal structures.
### Special Cases

HSA generalizes common attention variants:

- **MQA**: `(1, head_dim)` → all heads share features  
- **GQA**: `(g, head_dim)` → fixed grouping  
- **MHA**: `(num_heads, head_dim)` → no sharing  

HSA allows combining these simultaneously across different feature subspaces.

---

## Results

We evaluate HSA under 10k fixed-step training. MLP expansion is adjusted to maintain a consistent parameter budget (MLP mult = 2.27).

We use the following hierarchical configurations for attention projections:

`q_levels = [(2, 8), (4, 16), (8, 40)]`  
`kv_levels = [(1, 16), (2, 16), (4, 32)]`

| Model | Seed | Pre-quant BPB ↓ | Post-quant BPB ↓ | Size (bytes) |
|-------|------|----------------:|-----------------:|-------------:|
| Baseline | 1337 | 1.2262 | 1.2328 | 15861272 |
| HSA | 1337 | **1.2223** | **1.2285** | 15890606 |
| Baseline | 42 | 1.2276 | 1.2343 | 15856563 |
| HSA | 42 | **1.2223** | **1.2282** | 15895007 |
| Baseline | 2025 | 1.2253 | 1.2321 | 15853892 |
| HSA | 2025 | **1.2228** | **1.2295** | 15876915 |
| **Average (Baseline)** | — | 1.2264 | 1.2331 | 15857242 |
| **Average (HSA)** | — | **1.2225** | **1.2287** | 15887509 |

---

## Practical Considerations

- Reduces parameter count in QKV projections  
- Reduces KV-cache size due to shared feature representations across heads  
- Implementation is not currently optimized, and adds moderate overhead compared to standard projections
- Compatible with existing attention implementations  
---
