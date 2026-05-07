# 12L/768D + INT6/LZMA + BigramHash & U-Net Skips

**By Rugved Katkade**

### Summary
This run implements a 12-layer, 768-dimension Transformer architecture optimized for the 16MB Parameter Golf artifact limit. The core strategy maximizes representational capacity by using unique weights quantized to INT6 and compressed using LZMA-9. This allows for a deeper and wider model (~57M params) compared to standard INT8 baselines.

### Architecture
- **Model**: 12 Layers, 768 Embedding Dimension, 8 heads (4 KV heads).
- **Embeddings**: Tied 1024-token BPE + 3072-token **BigramHash** for cheap local context.
- **Skip Connections**: U-Net style residual paths between early and late layers using learnable scalar mixing weights.
- **Stability**: Logit softcapping (30.0) and weight-free RMSNorm.

### Optimization
- **Muon**: Used for all 2D matrix parameters (Linear & Conv layers).
- **AdamW**: Used for embeddings and scalar parameters.
- **Grad Accum**: 8-step accumulation (524k tokens per global step).

### Quantization & Compression
- **Format**: Per-row INT6 quantization (31-level clipping).
- **Artifact**: Final state dict serialized with Pickle (protocol 4) and compressed with LZMA (preset 9).
- **Size**: ~14.5MB total (model + code).
