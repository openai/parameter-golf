# EMA + XSA + SmearGate + BigramHash

**Team:** HCMUS Ninepls  
**Track:** Non-record (16MB)  
**val_bpb:** 1.4705 (T4 test run, expect ~1.17 on 8×H100)

## Techniques added over naive baseline

- **EMA** (decay=0.997): exponential moving average of weights applied
  before final eval. Smoother weights → better generalisation.
- **XSA** on last 3 layers: removes each token's self-value contribution
  from attention output via orthogonal projection. Zero extra params.
- **SmearGate**: learned sigmoid gate blending x[t] with x[t-1],
  allowing residual carry-over across positions.
- **BigramHash**: hash-based bigram context table (4096×128, ~3K params)
  added to token embeddings for cheap local context.
- **Sliding-window eval** (stride=64): every token scored with maximum
  available context instead of fixed chunk boundaries.
- **Int-8 PTQ + zlib**: per-row int-8 quantisation of weight matrices,
  compressed with zlib level 9. Artifact ~7.4MB well under 16MB limit.

## Architecture

- 9 transformer blocks, dim=512, 8 heads, 4 KV heads (GQA)
- Vocab 1024, seq_len 1024, tied embeddings
- U-Net skip connections (encoder/decoder split)
- Muon optimizer (matrix params) + AdamW (scalars/embeddings)
- RoPE with NTK-aware scaling

## Hardware

Tested on Kaggle T4 (1 GPU, reduced batch for verification).  
Full submission intended for 8×H100 SXM via torchrun --nproc_per_node=8.
