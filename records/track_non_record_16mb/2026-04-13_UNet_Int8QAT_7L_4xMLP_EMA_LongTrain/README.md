# Non-Record V2: 7-Layer UNet + Int8 QAT + 4×MLP + EMA + Long Training

**U-Net Skip Connections + Muon Optimizer + Int8 QAT + Leaky ReLU Squared + EMA + 4-hour training on DGX Spark**

**Mean val_bpb: 1.39693** (3 seeds) | **Best seed: 1.39532841** | **Artifact: ~14.83 MB** | DGX Spark (1x GB10 Blackwell, 128GB unified)

> **Non-record submission** — trained on NVIDIA DGX Spark (single GB10 GPU) rather than 8xH100. V2 iteration over the [2026-04-07 v1 submission](../2026-04-07_UNet_Int8QAT_LeakyReLU2_Muon), improving by **0.27 BPB** (1.6656 → 1.3969) through a deeper 7-layer model, 4× MLP multiplier, longer 4-hour training budget, and 3-seed statistical validation. Entire project developed with AI-assisted coding.

## Results

| Metric | Seed 1337 | Seed 42 | Seed 314 |
|--------|-----------|---------|----------|
| val_bpb (int8+zlib roundtrip, exact) | 1.39982649 | 1.39564112 | 1.39532841 |
| val_loss | 2.36354840 | 2.35648158 | 2.35595358 |
| Steps completed | 1033 | 1041 | 1040 |
| Artifact size | 15,518,077 B | 15,559,281 B | 15,565,024 B |
| Training time | 4h wallclock cap | 4h wallclock cap | 4h wallclock cap |

**Cross-seed statistics**: mean 1.39693 BPB, range 0.00450 BPB (0.32% of mean), std dev ~0.00251 BPB. Tight reproducibility across seeds confirms the configuration is stable rather than seed-lucky.

## Improvement over V1

| Dimension | V1 (2026-04-07) | V2 (2026-04-13) |
|-----------|-----------------|-----------------|
| Layers | 9 (d=512) | 7 (d=512) |
| MLP multiplier | default | 4× |
| Training wallclock | ~80 min | 4 h |
| Steps reached | ~319 | ~1037 (mean) |
| Seeds | 3 (42, 314, 999) | 3 (1337, 42, 314) |
| Best val_bpb | 1.6632 | 1.39533 |
| Mean val_bpb | ~1.6656 | 1.39693 |
| Artifact size | ~8.77 MB | ~14.83 MB |

The 7-layer + 4× MLP trade (fewer blocks, wider MLP) plus a 3× longer wallclock drove the 0.27 BPB gain. Artifact size grew but stays well under the 16 MB cap.

## Architecture

- **7 transformer layers**, dim=512, 8 heads, 4 KV heads (GQA), head_dim=64
- **MLP multiplier 4×** (wider FFN than v1)
- **U-Net skip connections**: encoder/decoder halves with learned skip weights and per-block residual mixing from input embedding
- **Leaky ReLU squared** MLP activation (negative_slope=0.5, then squared)
- **Tied embeddings** with separate tied_embed_lr
- **Logit softcap** (tanh-based)
- **RoPE** positional encoding
- Vocab size 1024 (SentencePiece BPE)
- Sequence length 1024
- 20,725,304 parameters

## Training Hyperparameters

- MATRIX_LR: 0.08261619767374824
- SCALAR_LR: 0.014691154447587356
- TIED_EMBED_LR: 0.021552090970329115
- HEAD_LR: 0.0 (tied-only head path)
- MUON_MOMENTUM: 0.9382982028913158
- WARMDOWN_ITERS: 1558
- EMA_DECAY: 0.997 (EMA enabled)
- QAT_ENABLED: 1 (int8)
- 524,288 train tokens per step, 8 gradient accumulation steps

## Training Recipe

- **Muon optimizer** for matrix parameters with Newton-Schulz orthogonalization
- **Adam** for embeddings and scalar/vector parameters
- **Quantization-aware training (QAT)**: int8 straight-through estimator during forward pass
- **EMA** (exponential moving average, decay 0.997) of model weights, applied before final serialization
- **Warmdown** LR schedule based on wallclock time remaining
- **MAX_WALLCLOCK_SECONDS**: 14400 (4 hours) — each seed hit the cap at ~step 1037

## Quantization & Serialization

- **Int8 per-row quantization** for all weight matrices
- **Float16 passthrough** for small tensors and control parameters
- **zlib compression** on serialized checkpoint
- Roundtrip validation: decompress, dequantize, and re-evaluate to verify BPB integrity
- Mean artifact ~15.55 MB — well under the 16 MB cap

## Hardware

- **NVIDIA DGX Spark**: ARM64 (aarch64), GB10 GPU (Blackwell sm_121)
- 128 GB unified CPU+GPU memory (no discrete VRAM)
- CUDA 13.0
- Single GPU training via torchrun
- ~13.8 s/step, ~1037 steps in 4 h wallclock (identical across all 3 seeds)
- Peak GPU memory: ~25.4 GB allocated / ~25.9 GB reserved

## Development

This submission was developed with AI-assisted development using Claude for architecture exploration, hyperparameter tuning, and orchestration of the multi-seed training runs. All training and evaluation ran on DGX Spark hardware.

## Files

- `train_gpt.py` — self-contained training script (same config across all 3 seeds)
- `train_seed1337.log` / `train_seed42.log` / `train_seed314.log` — raw training logs
- `submission.json` — structured result manifest
- `requirements.txt` — pip dependencies
