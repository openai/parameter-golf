# 11L Gradient-Guided Quant + EMA + Sliding Eval

**val_bpb: 1.1396** (post int8+zstd quantization roundtrip, sliding window eval stride=64, full validation coverage)

## Run Command

```bash
# Setup
pip install sentencepiece zstandard
python3 data/cached_challenge_fineweb.py

# Train + evaluate (defaults baked into train_gpt.py)
torchrun --nproc_per_node=8 records/track_10min_16mb/2026-03-22_11L_GradQuant_EMA_SlidingEval/train_gpt.py
```

All hyperparameters are set as defaults in `train_gpt.py`. No env vars needed.

## Architecture

- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu^2 activation
- Encoder-decoder skip connections with learned weights
- SmearGate residual mixing
- NTK-aware RoPE positional encoding
- XSA (cross-sequence attention) on last 4 layers
- Orthogonal initialization
- Tied input/output embeddings

## Training

- Muon optimizer: matrix_lr=0.025, scalar_lr=0.025, momentum=0.99
- AdamW for embeddings/scalars: WD=0.04
- Momentum warmup: 0.92 -> 0.99 over 1500 steps
- Adaptive warmdown: 3000 iters (auto-capped to 55% of total steps for hardware robustness)
- Warmup: 20 steps
- seq_len=2048, batch=786K tokens
- grad_clip=0.3
- EMA: alpha=0.997, initialized from model init

## Compression

- Gradient-guided adaptive quantization: per-tensor bit assignment based on gradient sensitivity
  - Top 45% (highest gradient): int7 (127 values)
  - Middle 40%: int6 (63 values)
  - Bottom 15% (lowest gradient): int5 (31 values)
- zstd level 22 compression
- Artifact: 15,913,419 bytes (code: ~59KB, model: ~15.9MB)

## Evaluation

- Sliding window eval with stride=64, full validation set coverage (~121K windows/GPU)
- Per-token loss scoring: only the last 64 tokens of each 2048-token window are scored (full context)
- Post-quantization roundtrip: quantize -> decompress -> evaluate
- Eval time: ~6 min (runs after training completes)
