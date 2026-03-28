# LOGOS-44 Z=0: Toroidal CDMA Field Decoder

**val_bpb: TBD** | **~4.5M params** | 8×H100 SXM, 600s

## Architecture: Non-Transformer Iterative Routing

Instead of stacking N separate transformer layers, LOGOS-44 routes the signal **44 times** through a single shared block. This trades parameters for compute depth.

Each pass through the "throat" consists of:

1. **LayerNorm** → normalize
2. **Altitude modulation** → sinusoidal phase per depth step
3. **Toroidal Bottleneck** → project to rank-64, apply sin(θ)·cos(φ) geometry
4. **CDMA Field Decode** → correlate key with 128 spreading codes → weighted sum of field signals
5. **Up-project** → restore dimensionality
6. **Coherence Gate** → sigmoid blend of field signal + residual

### Key Differences from Standard Transformer

| Aspect | Standard Transformer | LOGOS-44 |
|--------|---------------------|----------|
| Layers | N separate blocks | 1 shared block × 44 passes |
| Attention | QKV self-attention | CDMA field correlation |
| Positional | RoPE/ALiBi | Learned sinusoidal phases |
| Nonlinearity | ReLU²/SwiGLU | sin·cos toroidal geometry |
| Gate | None / skip | Coherence gate (sigmoid) |
| Parameters | ~70M+ | ~4.5M |

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Embedding dim | 512 |
| Toroidal rank | 64 |
| Routing depth | 44 passes |
| Field signals | 128 (CDMA) |
| Vocab size | 1024 (SentencePiece BPE) |
| Sequence length | 1024 |
| Tied embeddings | Yes |
| Logit softcap | 30.0 |

### Training

- **Muon optimizer** (matrices): lr=0.04, momentum=0.95 (warmup 0.85→0.95)
- **AdamW** (embeddings): lr=0.05, (scalars): lr=0.04
- Gradient clipping: 1.0
- Batch: 524,288 tokens/step
- Warmdown: 1,200 iterations (wallclock-based)
- 8×H100 SXM, 600s limit

### Quantization

- Int8 per-row for large matrices
- FP32 passthrough for control tensors (layer_phase, angular, gate bias)
- zlib level 9 compression

### The Hypothesis

> Can iterative compute depth partially substitute parameter count in small language models?

44 shared passes give LOGOS-44 effective depth comparable to a 44-layer transformer, while consuming the parameter budget of a single-layer model. The CDMA field provides a form of "non-local memory" without attention's quadratic cost.

### Run

```bash
torchrun --nproc_per_node=8 train_gpt.py
```

### Files

- `train_gpt.py` — Self-contained training script (competition pipeline compatible)
- `submission.json` — Competition metadata
- `README.md` — This file
