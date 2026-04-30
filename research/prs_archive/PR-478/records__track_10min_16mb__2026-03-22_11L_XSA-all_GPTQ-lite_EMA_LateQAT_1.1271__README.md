## Record: 11L XSA-all + GPTQ-lite + EMA + Late QAT (val_bpb: 1.12676, 3-seed mean)

**val_bpb: 1.12676** (3-seed mean, sliding window stride=64) | **~15.7 MB** | 8xH100 SXM, 600s

### Key Innovations

| Feature | Description | Impact |
|---------|-------------|--------|
| **XSA-all(11)** | Exclusive Self Attention on ALL 11 layers, not just last 4 | -0.002 BPB vs XSA(4) |
| **GPTQ-lite** | 5 clip percentiles per row, pick min MSE reconstruction | -0.0006 BPB (zero training cost) |
| **EMA(0.997)** | Exponential moving average every step | Smoother weights, better compression |
| **Late QAT@0.15** | Int6 STE fake-quantization when LR scale < 0.15 | Minimal quant gap without training noise |
| **Raw Binary + zstd22** | Custom binary serialization, no torch.save overhead | ~300KB savings vs torch.save |
| **No Pruning** | Int6-all fits without magnitude or row pruning | No quality loss from pruning |

### XSA-all: Our Unique Contribution

Standard XSA applies only to the last few layers. We found that applying XSA to ALL 11 layers
provides a consistent 0.002 BPB improvement. Early layers benefit from XSA by being forced to
encode novel contextual information rather than repeating self-value patterns.

| Config | val_bpb | Steps | ms/step |
|--------|---------|-------|---------|
| XSA-all(11) | **1.12676** | 6764 | 88.7 |
| XSA(4) | 1.13266 | 6998 | 85.7 |

Despite XSA-all being ~3ms/step slower, the quality gain outweighs the ~230 fewer training steps.

### Ablation: Backout Removal

Removing the Backout mechanism (which subtracts middle-layer output) improved results by 0.0035 BPB.
With LN Scale + XSA-all already managing information flow, Backout was redundant and slightly destructive.

| Config | val_bpb |
|--------|---------|
| With Backout | 1.1306 |
| **Without Backout** | **1.1271** |

### Architecture

- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3x MLP expansion (1536 hidden), relu-squared activation
- U-Net skip connections (5 encoder, 6 decoder)
- Exclusive Self Attention (XSA) on ALL 11 layers (GQA-aware, zero-alloc)
- Partial RoPE (16/64 dims) + NTK-aware scaling
- LN Scale Factor 1/√(layer_idx+1)
- Shared Value Embedding (dim=128, layers 9,10) with per-layer learned scales
- SmearGate + BigramHash (2048 buckets, dim=128)
- Tied embeddings, logit softcap=30.0

### Training

- FlashAttention 3 (Hopper-optimized), falls back to PyTorch SDPA if FA3 unavailable
- Muon optimizer (matrices): lr=0.025, momentum=0.99 (warmup 0.92→0.99 over 1500 steps), WD=0.04
- AdamW (embeddings): lr=0.035, (scalars): lr=0.025, WD=0.04
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 3500 iterations (wallclock-based)
- **EMA**: decay=0.997, every step (applied before quantization)
- **Tight SWA**: every 50 steps when scale<0.2
- **Late QAT**: Int6 STE fake-quantization when LR scale<0.15 (~step 6242)
- OrthoInit + muP-scaled output projections
- 20-step warmup with state reset

### Quantization

- **GPTQ-lite**: Per-row optimal clip percentile search (5 candidates: 0.999, 0.9995, 0.9999, 0.99999, 1.0) for int6
- Int6 per-row for ALL large weights (MLP + attention + bigram + VE)
- Int8 per-row for embeddings
- Control tensors in fp32
- **Raw binary serialization** + zstd level 22 compression (no torch.save overhead)

### Dependencies

- `zstandard` (for zstd level 22 compression; falls back to zlib if unavailable, but model may exceed 16MB with zlib)
- `flash_attn_3` (FlashAttention 3 for Hopper GPUs; falls back to PyTorch SDPA if unavailable, but ~3ms/step slower)
- See `requirements.txt`

### Results (3 seeds, BACKOUT_ENABLED=0)

| Seed | Steps | Sliding BPB (s64) | Size | ms/step |
|------|-------|--------------------|------|---------|
| 42 | 6764 | 1.12713 | 15.64 MB | 88.7 |
| 1337 | 6766 | 1.12648 | 15.62 MB | 88.7 |
| 2024 | 6764 | 1.12667 | 15.88 MB | 88.7 |
| **Mean** | **6765** | **1.12676** | **~15.7 MB** | **88.7** |

Standard deviation: 0.00034 BPB. All seeds well under 16MB limit.

### Statistical Significance vs Current SOTA (1.14276)

Mean improvement: 1.14276 - 1.12676 = 0.01600 nats (well above 0.005 threshold).
All 3 seeds individually beat SOTA by > 0.015 nats.

### Run Command

```bash
BACKOUT_ENABLED=0 MAX_WALLCLOCK_SECONDS=600 RUN_ID=v34 SEED=42 \
python3 -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py
```

### Setup

```bash
# Install dependencies (on RunPod template with PyTorch 2.9.1+cu128 pre-installed)
pip install -r requirements.txt

# Alternative manual install (if requirements.txt has issues):
# pip install zstandard
# pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291 --no-deps

# Download dataset (if not already present)
python3 data/cached_challenge_fineweb.py --variant sp1024
```
