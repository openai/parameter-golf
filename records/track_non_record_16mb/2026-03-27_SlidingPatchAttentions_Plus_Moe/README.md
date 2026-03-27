# Sliding Patch Attentions + MoE (2-layer compact run)

This folder contains a `non_record_16mb` submission built from the current `train_gpt.py`.

This run is not intended for the 8xH100 / 10-minute main leaderboard. It was trained on a single H100 for a 600 second wallclock cap and is submitted as a compact non-record result plus a code snapshot of the experimental branch.

## Summary

- Track: `non_record_16mb`
- Author: GRisk ([@BurguerJohn](https://github.com/BurguerJohn))
- Date: `2026-03-27`
- Hardware: `1x NVIDIA H100 80GB HBM3`
- Timed stop: `2869/20000` steps at `600011ms`
- Parameters: `4,198,928`
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=2 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 TIE_EMBEDDINGS=1`
- Training batch: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- Eval mode: flat validation over the full `fineweb_val_*` split
- Best timed validation before export: `val_loss:2.5074`, `val_bpb:1.4850`
- Final post-quant roundtrip metric: `val_loss:2.51455785`, `val_bpb:1.48926280`
- Submission size: `3938328` bytes
- Compressed model size: `3853521` bytes
- Logged code size: `84807` bytes

## What This Submission Actually Runs

The `train_gpt.py` in this folder experiments with a few ideas beyond the stock baseline:

- shifted patch self-attention inside a spatial router
- linearized global self-attention in the router path
- top-k MoE routing over sliding token windows
- block-history residual mixing
- encoder/decoder-style skip connections

However, the exact logged run submitted here uses `NUM_LAYERS=2`, and the log reports `moe_layers:0/2`. In other words, the router/MoE path is present in code, but it is not active in the measured run. The submitted score should therefore be interpreted as a compact baseline result from this experimental branch, not as a full MoE-enabled result.

## Reproducing The Run

Track-relevant command:

```bash
OMP_NUM_THREADS=1 \
RUN_ID=gpt_2026-03-27_17-45-21 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=2 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TIE_EMBEDDINGS=1 \
TIED_EMBED_LR=0.05 \
HEAD_LR=0.0 \
MATRIX_LR=0.04 \
SCALAR_LR=0.04 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=1024 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=1000 \
WARMUP_STEPS=20 \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
TORCH_COMPILE=auto \
python train_gpt.py
```

Notes:

- The run stops early on the wallclock cap rather than completing all `20000` iterations.
- Validation is flat, not sliding-window evaluation, for this specific run.
- The provided `train.log` contains the code snapshot followed by the training output from the run.

## Included Files

- `train_gpt.py`: training script used for the run
- `train.log`: captured run output
- `submission.json`: leaderboard metadata for this submission

## Submission Notes

This submission fits well under the `16,000,000` byte artifact limit after int8 + zlib export, but it does not claim a statistically significant win and it does not target the main 8xH100 record track. It is submitted to document the experimental code branch and its compact single-GPU result.
