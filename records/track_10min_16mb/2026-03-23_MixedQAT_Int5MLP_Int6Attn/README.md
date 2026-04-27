# Mixed-Precision QAT: Int5 MLP + Int6 Attention + Full SOTA Stack

**val_bpb: 1.14777536** (seed=42, 1 run; SOTA baseline = 1.1428)

Built on PR #162 by @unnir and the `2026-03-20_10L_Int5MLP_MuonWD04_SWA50` SOTA.

## Core Idea

The current SOTA uses **post-training quantization (PTQ)**: weights are trained in fp32/bf16
and only quantized (int5 for MLP, int6 for attention) at export time.  PTQ introduces a
quantization gap of roughly 0.016 bpb between the pre-quant and post-quant scores.

This submission adds **quantization-aware training (QAT)** with the Straight-Through Estimator
(STE) to close that gap.  Crucially, the STE clip ranges exactly match the export scheme:

| Layer type        | QAT clip range | Export scheme |
|-------------------|---------------|---------------|
| MLP fc / proj     | ±15 (int5)    | int5 per-row  |
| Attn Q/K/V/O      | ±31 (int6)    | int6 per-row  |
| BigramHash.proj   | ±31 (int6)    | int6 per-row  |
| tok_emb           | —             | fp16          |

The STE trick:
```
w_quantized = clamp(round(w / scale), -(clip+1), clip) * scale
w_ste       = w + (w_quantized - w).detach()
```
Forward pass sees `w_quantized` (simulated quantization noise);
backward pass receives real gradients through `w` (STE, no dead-gradient problem).

## Architecture (identical to SOTA)

- 10 transformer layers, dim 512, 8 heads, 4 KV heads (GQA)
- MLP 3× expansion (hidden 1536), ReLU² activation
- U-Net skip connections
- SmearGate — learned previous-token blending at embedding layer
- BigramHash(10240, dim=128) — hash-based bigram embedding table
- Orthogonal weight init with muP output-projection scaling
- Tied embeddings (vocab 1024)

## Training Hyperparameters (identical to SOTA)

| Parameter            | Value              |
|----------------------|--------------------|
| seq_len              | 2048               |
| batch                | 786 432 tokens     |
| warmdown_iters       | 3000               |
| matrix_lr (Muon)     | 0.02               |
| scalar_lr (AdamW)    | 0.02               |
| tied_embed_lr        | 0.03               |
| Muon momentum        | 0.99 (warmup 0.92→0.99 over 1500 steps) |
| Muon WD              | 0.04               |
| AdamW WD             | 0.04               |
| grad_clip            | 0.3                |
| SWA start_frac       | 0.4                |
| SWA every            | 50 steps           |
| Eval stride          | 64 (sliding window)|
| Compressor           | zstd level 22      |
| QAT                  | **enabled from step 0** |

## Why QAT + Mixed Int5/Int6 Should Help

- The 3rd-place entry (uniform int6 QAT, 11 layers, no SmearGate/BigramHash) = 1.1502
- The SOTA (mixed int5/int6 PTQ, SmearGate, BigramHash) = 1.1428
- Combining: QAT aligned to the *actual* export quantization scheme + full SOTA stack

The 3rd-place entry shows QAT reduces the quantization gap from ~0.015 to near zero.
The SOTA shows int5 MLP + SmearGate + BigramHash improve the base score by ~0.015.
Together these improvements are largely orthogonal: expected combined gain ≈ 0.007–0.010 bpb.

## How to Run

```bash
# Requires: pip install zstandard

# Setup (once, on RunPod or any CUDA machine)
python3 data/cached_challenge_fineweb.py --variant sp1024

# Train + evaluate (seed=42)
SEED=42 \
RUN_ID=mixedqat_seed42 \
torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-03-23_MixedQAT_Int5MLP_Int6Attn/train_gpt.py

# All hyperparameters are baked in as defaults; no env vars needed beyond paths.
```

## Expected Artifact Size

Same architecture as SOTA; artifact size should be ~15.8–15.9 MB (well within 16 MB).

## Results

| Metric | Value |
|--------|-------|
| val_bpb (final, sliding window) | **1.14777536** |
| val_loss | 1.93796548 |
| Steps completed | 6137 / 20000 (wallclock cap hit) |
| Model size (int6+zstd) | 15,901,560 bytes |
| Code size | 55,721 bytes |
| Total submission size | 15,957,281 bytes |
| Peak GPU memory | 23,856 MiB |
| SWA checkpoints averaged | 24 |

Run stopped at ~600 s wallclock, completing 6137 steps at ~97.77 ms/step on 8×H100.

## Files

- `train_gpt.py` — training script with mixed QAT
- `README.md` — this file
- `submission.json` — run metadata
- `train_seed42.log` — full training log (seed=42)
