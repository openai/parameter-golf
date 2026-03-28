# Record: 11L SOTA Fork — LeakyReLU² + XSA + EMA + GPTQ-lite + SmearGate

**Target: val_bpb ≤ 1.12** | 8xH100 SXM, 600s

## Architecture

| Component | Config |
|---|---|
| Layers | 11 (5 encoder + 6 decoder, U-Net skip) |
| Dimensions | 512d, 8 heads (4 KV heads, GQA) |
| MLP | 3x expansion (1536 hidden), LeakyReLU²(0.5) |
| XSA | Last 4 layers (GQA-aware, orthogonal projection) |
| RoPE | Partial (16/64 dims) |
| SmearGate | Per-token gating with previous token |
| BigramHash | 2048 buckets, dim=128 |
| Tied embeddings | Yes, logit softcap=30.0 |
| LN Scale | 1/sqrt(layer_idx+1) per layer |

## Key Techniques

### Training
- **Muon optimizer**: lr=0.025, momentum 0.92→0.99 (warmup 1500 steps), WD=0.04
- **AdamW** (embeddings): lr=0.035, (scalars): lr=0.025, WD=0.04
- **Gradient clip**: 0.3
- **Batch**: 786,432 tokens/step, seq_len=2048
- **Warmdown**: 3500 iterations (cosine schedule)
- **OrthoInit**: Orthogonal initialization for all projection layers

### Weight Averaging
- **EMA**: decay=0.997, every step, GPU-side (avoids 32% throughput hit)
- **Tight SWA**: every 50 steps when LR scale < 0.2
- Final weights = blend of EMA and SWA averages

### QAT + Quantization
- **Late QAT**: STE int6 fake-quantization when LR scale < 0.15
- **GPTQ-lite**: Per-row optimal clip percentile search (5 candidates: 0.999, 0.9995, 0.9999, 0.99999, 1.0)
- **Int6** per-row for MLP + attention weights
- **Int8** per-row for embeddings
- Control tensors in fp32
- **zstd level 22** compression (or zlib-9 fallback)

### Evaluation
- **Sliding window** with stride=64 for better BPB

## Lineage

Built on the merged SOTA stack:
- PR #374 architecture (11L, XSA, SmearGate, BigramHash)
- PR #414 optimizations (EMA, GPTQ-lite, warmdown tuning, Late QAT)
- LeakyReLU² from modded-nanogpt speedrun findings

## Run Command

```bash
# Setup (once)
python3 data/cached_challenge_fineweb.py --variant sp1024

# Train + evaluate (default seed=1337)
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py

# With specific seed
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All hyperparameters are set as defaults in `train_gpt.py`. No env vars needed.

## Files

- `train_gpt.py` — Complete training + evaluation + quantization script (1195 lines)
- `README.md` — This file
- `submission.json` — Submission metadata
