## 9L INT6 AR-GPTQ + XSA-all + BigramHash 3072×112 (val_bpb: 1.2005)

**val_bpb: 1.2005** (sliding window stride=64, seed 42) | **~14.08 MB** | 8×H100, 560s

### Approach

9-layer model forced by the 16MB artifact limit — 11L INT6 exceeds 16MB (~18MB), so layers were reduced to 9 to fit within the constraint. Combines AR self-generated GPTQ calibration, XSA on all layers, and BigramHash 3072×112 from the SOTA config (PR #1019), adapted for 9L.

### Key Differences vs PR #1019 (11L SOTA)

| Component | PR #1019 | This |
|-----------|----------|------|
| **Layers** | 11L | 9L (16MB constraint) |
| **Artifact size** | ~15.9MB | ~14.08MB |
| **VE layers** | 9,10 | 7,8 |
| **XSA_LAST_N** | 11 | 9 (all layers) |
| **val_bpb** | 1.1147 | 1.2005 |

### Results

| Seed | val_loss | val_bpb | Artifact |
|------|----------|---------|----------|
| **42** | 2.0269 | **1.2005** | ~14.08 MB |
| 1337 | pending | pending | - |
| 2024 | pending | pending | - |

Seeds 1337 and 2024 pending compute allocation.

### Architecture

- 9 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3.0× MLP expansion, LeakyReLU(0.5)² activation
- U-Net skip connections
- XSA on all 9 layers
- Partial RoPE (16 dims)
- LN Scale 1/sqrt(layer+1)
- Value Embeddings (dim=128, layers 7,8)
- BigramHash 3072-entry polynomial hash, dim=112
- Tied embeddings

### Training

- Parallel Muon optimizer (batched Newton-Schulz): lr=0.025, momentum=0.99, WD=0.04
- AdamW (embeddings): lr=0.035, (scalars): lr=0.025, WD=0.04
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- Wallclock cap: 560s (40s headroom for GPTQ within 600s budget)
- EMA decay=0.997 + SWA warmdown-only (30% blend, every 50 steps)
- Late QAT: INT6 fake-quantization when LR scale < 0.15

### Quantization

- Uniform INT6 per-row GPTQ for all weights
- 10-candidate clip search + column-wise error compensation
- AR self-generated calibration (64 seqs × 2048 tokens, temp=0.8, fixed seed)
- best-of zstd-22 / lzma-9 compression

### Hardware

8×H100, 560s wallclock, 22.3M parameters

### Notes

Single seed result (seed 42). Full 3-seed submission pending compute grant.
