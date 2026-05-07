# Cross-Layer Reciprocal Basis Sharing (CRBS)

**Author:** Muhammad Musab  
**Status:** In Progress  
**Target Score:** < 1.14 bpb

## Core Idea

Every MLP block in a transformer has an expansion layer (dim → k×dim) 
and a contraction layer (k×dim → dim). These two are geometrically 
related — one expands, the other collapses back. Instead of storing 
two fully independent weight matrices, we bind both to a single shared 
low-rank basis (U, V) with lightweight per-group diagonal scaling vectors.

Forward pass routes inputs sequentially:
x → xU → elementwise scale by d → xV

No dense weight matrices are ever materialized. Everything runs through 
the smaller basis matrices directly.

## Three Stacked Techniques

### 1. CRBS on MLP Layers
- Shared basis U ∈ R^{C×R}, V ∈ R^{R×C} at rank=128
- Per-group diagonal vectors d_up, d_down (kept FP16 at quantization)
- ~20-24x MLP parameter reduction per layer
- Freed bytes reinvested into wider dim (512→640) and more layers (10→12)

### 2. Encoder-Decoder Block Tying
- Baseline already has U-Net topology with skip connections
- Paired blocks (block i and block N-1-i) share the same U, V basis
- Each block keeps independent diagonal vectors
- Halves basis parameter cost across all paired blocks

### 3. Trigram Context Embedding
- Extends existing bigram hash embedding to also hash token triplets
- (t_{i-2}, t_{i-1}, t_i) hashed via XOR into same embedding table
- Near-zero extra parameter cost

## Compression Strategy
- Diagonal vectors (d_up, d_down): kept FP16 — tiny but critical
- Shared basis matrices (U, V): INT8 quantization
- Magnitude pruning on diagonal vectors during warmdown
- zstd level 22 final compression

## Expected Gains
- Model wider and deeper than baseline at same or smaller file size
- More principled quantization targeting than flat INT5/INT6
- Compounding gains from all three techniques together

## Results
*Training runs in progress — will update with bpb scores*
```
6. **Copy the URL** of this PR — it will look like:
```
https://github.com/openai/parameter-golf/pull/XXX
