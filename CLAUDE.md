# CLAUDE.md — JEPA-LM for Parameter Golf

## What This Is

A JEPA (Joint Embedding Predictive Architecture) language model for OpenAI's Parameter Golf challenge. This is a novel architecture — JEPA has never been tried for text compression in this competition. The submission lives in `train_gpt.py`.

## Quick Start

```bash
# 1. Download data (only needed once)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

# 2. Train on 8xH100 (competition run)
torchrun --standalone --nproc_per_node=8 train_gpt.py

# 3. Smoke test (single GPU, fast)
ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Architecture: JEPA-Augmented Transformer

The baseline U-Net transformer is reinterpreted as a JEPA system:

- **Encoder (layers 0–4):** Produces latent embeddings at the bottleneck
- **Decoder (layers 5–9):** Transforms latents into token predictions via skip connections
- **JEPA projection:** Projects bottleneck to 256-dim latent space
- **Latent predictor:** Predicts next-position latent embedding (MSE loss)
- **SIGReg:** Sketched Isotropic Gaussian Regularizer prevents representation collapse
- **Output:** Standard cross-entropy logits for BPB evaluation

Loss: `L = CE + 0.1 * JEPA_MSE + 0.03 * SIGReg` (constant weights; SIGReg as buffer for torch.compile compatibility)

Also includes LeakyReLU(0.5)^2 activation (proven -0.003 BPB improvement).

## JEPA Hyperparameters (env vars)

| Variable | Default | Description |
|----------|---------|-------------|
| `JEPA_DIM` | 256 | Latent projection dimension |
| `JEPA_ALPHA` | 0.1 | Weight of JEPA prediction loss |
| `SIGREG_WEIGHT` | 0.03 | Weight of SIGReg regularization |
| `SIGREG_NUM_PROJ` | 128 | Random projections for SIGReg |
| `SIGREG_SUBSAMPLE` | 4096 | Token subsample size for SIGReg |

All standard Parameter Golf hyperparameters (NUM_LAYERS, MODEL_DIM, etc.) still work.

## Constraints

- `train_gpt.py` must stay under 1500 lines (currently ~1213)
- Artifact (int8+zlib model + code) must be < 16,000,000 bytes (currently ~12.8MB)
- Training must complete in < 600 seconds on 8xH100
- Final metric: `final_int8_zlib_roundtrip val_bpb` in the output

## Submission

When ready, create a folder in `records/track_10min_16mb/` with:
- `train_gpt.py` (the self-contained training script)
- `README.md` (describe the approach)
- `submission.json` (scores from 3+ seed runs)
- Training logs
