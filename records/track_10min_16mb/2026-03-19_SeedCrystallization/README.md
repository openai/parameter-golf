# Seed Crystallization: ALBERT-style Weight Sharing for Parameter Golf

## Approach

Instead of storing 10 unique transformer blocks (the standard approach), we store **3 unique wide blocks** that are **recycled 4 times each** for 12 effective layers. This fundamentally changes the capacity-per-byte equation:

- **Current SOTA**: 10 unique blocks x 512 dim = ~18.9M unique params -> ~14.7MB compressed
- **This approach**: 3 unique blocks x 768 dim = ~13.2M unique params -> ~12.5MB compressed, but with **50% wider representations** and **20% more effective depth**

### Architecture

```
Embedding: 1024 x 768 (FP16 tied, ~1.5 MB)

Block A: dim=768, 12 heads, 6 KV heads, 2x MLP (relu^2)
Block B: dim=768, 12 heads, 6 KV heads, 2x MLP (relu^2)
Block C: dim=768, 12 heads, 6 KV heads, 2x MLP (relu^2)

Forward: A -> B -> C -> A -> B -> C -> A -> B -> C -> A -> B -> C  (12 effective layers)
         |--- encoder (6) ---|   |--- decoder (6) ---|
         U-Net skip connections across encoder/decoder halves
```

### Key Innovation: Per-Iteration Scalars

Each recycled application of a shared block gets its own learned `attn_scale`, `mlp_scale`, and `resid_mix` parameters. These are tiny (~37K params total) but break the symmetry between recycled iterations, allowing each application to specialize.

### Why This Should Work

1. **Width > unique depth for LMs**: Language model quality scales more with hidden dimension than with number of unique parameter sets. A 768-dim model with shared blocks should outperform a 512-dim model with unique blocks at the same storage budget.

2. **Compression multiplier**: Any improvement to the 3 shared blocks (better quantization, QAT, etc.) multiplies 4x since each block is reused 4 times.

3. **~3.9MB headroom**: The compressed model is ~12.5MB vs the 16MB cap, leaving room for future improvements (wider dim, 4th block, int4 quantization).

### Preserved from SOTA
- Sliding window evaluation (stride=64)
- FP16 tied embeddings with overtone spectral init
- Muon optimizer with decoupled weight decay
- Phase-transition residual mix initialization
- U-Net skip connections

## Parameter Budget

| Component | Params | Storage |
|-----------|--------|---------|
| Embedding 1024x768 (FP16) | 786K | ~1.5 MB |
| Block A (attn + MLP) | ~4.1M | ~3.8 MB |
| Block B | ~4.1M | ~3.8 MB |
| Block C | ~4.1M | ~3.8 MB |
| Per-iteration scalars (12 sets) | ~37K | negligible |
| Skip weights | ~5K | negligible |
| **Total** | **~13.2M** | **~12.5 MB** |

## Results

*Pending GPU runs*

## How to Run

```bash
# Training (8xH100)
torchrun --nproc_per_node=8 train_gpt.py

# Training (1xH100 for testing)
python train_gpt.py
```
