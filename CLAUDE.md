# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

OpenAI's "Parameter Golf" challenge: train the best language model that fits in a **16MB artifact** (code + zlib-compressed int8 weights) and trains in **under 10 minutes on 8×H100s**. Scored by bits-per-byte (val_bpb) on a fixed FineWeb validation set. Inspired by NanoGPT Speedrunning but optimizing L(N) — lowest loss for fixed parameter count.

## Key Commands

### Data Download
```bash
# Download cached FineWeb with 1024-token vocab (default: full val + 80 train shards)
python3 data/cached_challenge_fineweb.py --variant sp1024
# Smaller subset for local iteration
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

### Training (CUDA, multi-GPU)
```bash
RUN_ID=my_run \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
Use `--nproc_per_node=1` for single GPU. Override `MAX_WALLCLOCK_SECONDS=0` to remove the 10-minute cap.

### Training (MLX, Apple Silicon)
```bash
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py
```

## Architecture

Two self-contained training scripts (hard cap: 1500 lines each):

- **`train_gpt.py`** (~1370 lines) — PyTorch/CUDA baseline. Includes: GPT model (GQA, RoPE, SwiGLU MLP, tied embeddings, logit softcap), Muon optimizer (Newton-Schulz orthogonalization), int8 quantization + zlib compression for artifact export, LoRA test-time training (TTT) at eval, DDP multi-GPU support. All hyperparameters are env-var configurable via the `Hyperparameters` class at the top.

- **`train_gpt_mlx.py`** (~1100 lines) — MLX port for local Apple Silicon iteration. Same model architecture, adapted optimizers, eager eval mode for 16GB machines.

- **`data/`** — Dataset download (`cached_challenge_fineweb.py`) and retokenization (`download_hf_docs_and_tokenize.py`). Shards are `.bin` files in `data/datasets/`, tokenizers in `data/tokenizers/`.

- **`records/`** — Submission history. Each record folder contains `train_gpt.py`, `submission.json`, `README.md`, and `train.log`. Two tracks: `track_10min_16mb/` (leaderboard) and `track_non_record_16mb/` (unlimited compute).

## Submission Rules

- New SOTA must beat existing by ≥0.005 nats at p < 0.01 (typically 3 run logs)
- Artifact size = code bytes + zlib-compressed model bytes ≤ 16,000,000 (decimal, not MiB)
- Submissions are PRs that add a folder under `records/` with: `README.md`, `submission.json`, `train_gpt.py`, `train.log`
- No external downloads or network calls allowed during evaluation
- Eval time limit: 10 minutes on 8×H100 (separate from training time)

## Key Hyperparameters (env vars)

All configured via environment variables. Key ones: `ITERATIONS`, `TRAIN_BATCH_TOKENS`, `TRAIN_SEQ_LEN`, `NUM_LAYERS`, `MODEL_DIM`, `NUM_HEADS`, `NUM_KV_HEADS`, `VOCAB_SIZE`, `VAL_LOSS_EVERY`, `MAX_WALLCLOCK_SECONDS`. See the `Hyperparameters` class in each script for the full list.

## Dependencies

Core: `torch`, `numpy`, `sentencepiece`, `tqdm`. MLX path adds `mlx`. Data scripts need `huggingface-hub`, `datasets`. See `requirements.txt`.
