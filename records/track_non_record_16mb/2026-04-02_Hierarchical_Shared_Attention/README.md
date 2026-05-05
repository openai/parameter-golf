# Hierarchical Shared Attention (HSA): Multi-Level Sharing Across Attention Heads

## Overview

Standard attention treats each head as independent, allocating separate parameters per head even though many heads learn similar or redundant features.

Standard attention treats each head as independent, allocating separate parameters per head even though many heads learn similar or overlapping features. HSA instead decomposes head features into shared, group-shared, and head-specific components, allowing the model to reuse common structure while preserving specialization.

Rather than choosing between full sharing, grouped sharing, or independent heads, HSA combines these patterns within a single hierarchy. HSA treats the head dimension as a feature budget that can be split across multiple sharing granularities, rather than assigning the entire head to a single sharing pattern.

---

## Motivation

Empirically, attention heads are not fully independent:

- many heads learn similar or overlapping patterns
- pruning or merging heads often has limited impact in larger models
- grouped-query attention already exploits partial sharing

However, common attention variants enforce a single level of sharing:

- **MQA:** all heads share the same features
- **GQA:** heads share within fixed groups
- **MHA:** heads remain independent

This suggests a more flexible structure: some features can be shared across all heads, some can be shared across groups of heads, and some can remain head-specific.

HSA is designed to model this structure explicitly.

---

## Method

HSA constructs query, key, and value projections using multiple levels of shared features.

Each level is defined by a pair `(g, d)`:

- `g`: number of groups, or how many distinct feature sets exist at this level
- `d`: number of dimensions allocated to this level

At each level:

- features are shared within groups of heads
- smaller `g` means more sharing
- larger `g` means more specialization

The total head dimension is formed by concatenating features from all levels. This allows different portions of the feature space to operate at different sharing granularities.

Shared features are also modulated with learned per-head scaling, allowing shared representations to specialize with minimal additional cost.

---

## Pseudocode

```python
class HierarchicalSharedProjection:
    levels = [(num_groups, dim), ...]

    # one learned scale per head and per level dimension
    per_head_scales = learned_scales(num_heads, dim_for_each_level)

    def forward(x):
        proj = linear(x)
        pieces = []

        for groups, dim in levels:
            # read this level's projected features
            level = take_level_slice(proj, groups * dim)
            level = level.reshape(groups, dim)

            # expand group-shared features to heads
            group_id = head_to_group_map(groups)
            per_head = level[group_id]

            # lightweight per-head specialization
            per_head = per_head * per_head_scales[groups]

            pieces.append(per_head)

        return concat(pieces, dim=-1)
```

---

## Example Configuration

For an 8-head projection, a hierarchy could use:

```text
levels = [(1, 8), (2, 16), (4, 26), (8, 14)]
```

This corresponds to 8 heads with a total head dimension of 64.

The decomposition is:

- `(1, 8)`: 8 dimensions shared across all heads, MQA-style
- `(2, 16)`: 16 dimensions split into 2 groups, each shared across 4 heads
- `(4, 26)`: 26 dimensions split into 4 groups, each shared across 2 heads
- `(8, 14)`: 14 dimensions unique to each head, MHA-style

The final representation for each head is formed by concatenating these components, so each head contains:

- globally shared features for broad reuse
- group-shared features at multiple granularities
- head-specific features for full specialization

This example shows the main idea behind HSA: MQA-style sharing, GQA-style sharing, and MHA-style specialization can coexist inside the same head representation.

---

## Setup

- Fixed 10k training steps
- Same baseline comparison setup as the naive baseline
- HSA applied to attention projections
- MLP expansion adjusted to maintain the parameter budget
- MLP multiplier: 2.27

This PR uses the following hierarchical configurations:

```text
q_levels = [(2, 8), (4, 16), (8, 40)]
kv_levels = [(1, 16), (2, 16), (4, 32)]
```

---

## Results

All runs use identical settings across three seeds, building off of the original naive baseline.

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

On average, HSA improves both pre-quant and post-quant BPB while staying within the same size budget.

---
