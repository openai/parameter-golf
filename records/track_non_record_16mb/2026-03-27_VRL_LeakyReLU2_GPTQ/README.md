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
## Smoke test logs

Building SOTA model: 11L, 512d, 3x MLP, XSA last 4 layers, LeakyReLU²(slope=0.5)
Total parameters: 26,829,912

Starting training: batch=131,072 tokens, seq_len=2048, warmdown=3500
Step 1: Late QAT activated (lr_scale=0.0500)
step=25 | train_loss=5.3126 | lr_scale=0.0448 | step_time=1761.5ms | elapsed=43.3s | qat=ON
step=50 | train_loss=5.1735 | lr_scale=0.0402 | step_time=1265.8ms | elapsed=62.5s | qat=ON
step=75 | train_loss=5.0654 | lr_scale=0.0359 | step_time=1101.2ms | elapsed=81.8s | qat=ON
step=100 | train_loss=5.1499 | lr_scale=0.0319 | step_time=1020.6ms | elapsed=101.3s | qat=ON
step=125 | train_loss=4.9812 | lr_scale=0.0281 | step_time=774.9ms | elapsed=120.8s | qat=ON
^[[A^[[A^[[B^[[B^[[B^[[Bstep=150 | train_loss=4.9898 | lr_scale=0.0245 | step_time=776.1ms | elapsed=140.1s | qat=ON
step=175 | train_loss=5.0093 | lr_scale=0.0211 | step_time=780.2ms | elapsed=159.8s | qat=ON
step=200 | train_loss=4.9182 | lr_scale=0.0180 | step_time=779.7ms | elapsed=179.3s | qat=ON
step=225 | train_loss=4.8911 | lr_scale=0.0152 | step_time=779.2ms | elapsed=198.7s | qat=ON
step=250 | train_loss=4.9016 | lr_scale=0.0125 | step_time=780.6ms | elapsed=218.2s | qat=ON
step=275 | train_loss=4.9626 | lr_scale=0.0102 | step_time=778.0ms | elapsed=237.6s | qat=ON
step=300 | train_loss=4.8946 | lr_scale=0.0080 | step_time=778.3ms | elapsed=257.1s | qat=ON
step=325 | train_loss=4.8770 | lr_scale=0.0062 | step_time=781.4ms | elapsed=276.8s | qat=ON
step=350 | train_loss=4.8561 | lr_scale=0.0045 | step_time=785.8ms | elapsed=296.7s | qat=ON
step=375 | train_loss=4.8868 | lr_scale=0.0031 | step_time=788.6ms | elapsed=316.5s | qat=ON
step=400 | train_loss=4.8844 | lr_scale=0.0020 | step_time=790.2ms | elapsed=336.1s | qat=ON
step=425 | train_loss=4.8747 | lr_scale=0.0011 | step_time=788.0ms | elapsed=355.6s | qat=ON
step=450 | train_loss=4.8459 | lr_scale=0.0005 | step_time=782.9ms | elapsed=375.1s | qat=ON
step=475 | train_loss=4.8406 | lr_scale=0.0001 | step_time=781.0ms | elapsed=394.6s | qat=ON
step=500 | train_loss=4.8391 | lr_scale=0.0000 | step_time=782.5ms | elapsed=414.3s | qat=ON

Applying EMA weights...
Applying SWA (10 checkpoints)...
                                                                                                                                             Final (pre-quant): val_loss=4.9621 | val_bpb=2.9389
Quantizing (int6 MLP/attn, int8 embed, GPTQ-lite)...
Post-quant roundtrip: val_loss=4.9654 | val_bpb=2.9408                                                                                       Quantization gap: +0.0020 BPB                                              

============================================================
ARTIFACT SUMMARY
============================================================
Code size:              50,152 bytes
Model size:          5,549,512 bytes (zlib-9)
Total artifact:      5,599,664 bytes (5.60 MB)
16MB limit:         16,000,000 bytes
Headroom:           10,400,336 bytes
============================================================
final_val_bpb:    2.9408
final_val_loss:   4.9654
============================================================
Artifact fits within 16MB limit.
Saved: model_smoke2.bin

## Files

- `train_gpt.py` — Complete training + evaluation + quantization script (1195 lines)
- `README.md` — This file
- `submission.json` — Submission metadata
