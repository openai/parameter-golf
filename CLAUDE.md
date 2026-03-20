# CLAUDE.md

## Project Overview

OpenAI Parameter Golf: train the best language model that fits in a **16MB artifact** (code + compressed weights) and trains in **under 10 minutes on 8xH100 SXM GPUs**. Evaluation metric is **bits-per-byte (BPB)** on FineWeb validation (tokenizer-agnostic).

## Quick Commands

```bash
# Setup (local Mac with Apple Silicon)
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm

# Setup (remote CUDA — deps pre-installed in RunPod template)
pip install torch numpy sentencepiece huggingface-hub datasets tqdm

# Download data (1024-token SentencePiece vocab)
python3 data/cached_challenge_fineweb.py --variant sp1024              # full (80 shards, 8B tokens)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1  # small subset

# Train locally (MLX, Apple Silicon)
RUN_ID=test ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py

# Train on CUDA (single GPU)
RUN_ID=test torchrun --standalone --nproc_per_node=1 train_gpt.py

# Train on CUDA (8xH100, competition setting)
RUN_ID=test torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Architecture

- **Model**: GPT with encoder-decoder skip connections (U-Net style — first half stores activations, second half adds them back weighted by learnable `skip_weights`)
- **Attention**: Grouped Query Attention (GQA) with 8 heads / 4 KV heads, RoPE positional encoding, logit soft-capping
- **Optimizer**: Muon (Newton-Schulz orthogonalization) for matrix params + Adam for scalars/embeddings
- **Compression pipeline**: int8 post-training quantization (per-row for 2D, per-tensor otherwise) → zlib compression. Model is decompressed to higher precision at eval time.
- **Embeddings**: Tied input/output embeddings by default (`TIE_EMBEDDINGS=1`)

## Configuration

All hyperparameters are set via **environment variables** — there are no config files. See the `Hyperparameters` class at the top of each training script for the full list and defaults. Key ones:

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `./data/datasets/fineweb10B_sp1024` | Training data directory |
| `TOKENIZER_PATH` | `./data/tokenizers/fineweb_1024_bpe.model` | SentencePiece model |
| `VOCAB_SIZE` | `1024` | Vocabulary size |
| `NUM_LAYERS` | `9` | Transformer blocks |
| `MODEL_DIM` | `512` | Hidden dimension |
| `NUM_HEADS` / `NUM_KV_HEADS` | `8` / `4` | Attention / KV heads |
| `ITERATIONS` | `20000` | Training steps |
| `MAX_WALLCLOCK_SECONDS` | `600` | Wall-clock cap (0 = unlimited) |
| `VAL_LOSS_EVERY` | `1000` (CUDA) / `0` (MLX) | Steps between validation |
| `TRAIN_BATCH_TOKENS` | `524288` | Tokens per training step |

## Key Files

- `train_gpt.py` — Main CUDA training script (PyTorch + DDP via torchrun)
- `train_gpt_mlx.py` — Local MLX training script (Apple Silicon)
- `data/cached_challenge_fineweb.py` — Data download/preprocessing
- `records/` — Submitted runs (each in its own dated subfolder with README, submission.json, train log, and script)

## Submission Rules

- Artifact = code bytes + zlib-compressed int8 model bytes, must be < 16,000,000 bytes (decimal 16MB)
- New SOTA must beat previous by >= 0.005 nats at p < 0.01
- No network access during evaluation; artifact must be self-contained
- PRs add a folder under `records/track_10min_16mb/` or `records/track_non_record_16mb/`

## Style Notes

- Training scripts must stay under 1500 lines each
- `train_gpt.py` and `train_gpt_mlx.py` are starter scripts, not SOTA configs — keep them simple
- Competitive submissions go in `/records`, not in the root scripts
