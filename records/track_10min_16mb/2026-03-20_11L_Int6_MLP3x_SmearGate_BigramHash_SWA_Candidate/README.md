# 11L Int6 + SmearGate + Compact BigramHash + SWA

This folder is a candidate record implementation, not a measured submission yet.

It combines two strong public directions that looked compatible in code:

- The 11-layer int6 + weight-decay + sliding-window stack from PR `#179`
- The SmearGate + BigramHash + orthogonal init + SWA ideas from PR `#162`

## What Changed

- Keeps the 11-layer, 512-dim, GQA, `MLP_MULT=3.0` backbone.
- Trains at `TRAIN_SEQ_LEN=2048` and evaluates with sliding-window stride `64`.
- Keeps the main Muon weight-decay recipe with `WEIGHT_DECAY=0.038`.
- Adds `SmearGate` at the embedding layer.
- Adds a smaller `BigramHashEmbedding` by default: `BIGRAM_VOCAB_SIZE=2048`, `BIGRAM_DIM=96`.
- Uses orthogonal init for large matrices, with scaled projection weights.
- Enables SWA by default with `SWA_CHECKPOINTS=7`.
- Switches token/scalar optimizers to `AdamW` with `ADAM_WEIGHT_DECAY=0.01`.
- Keeps `tok_emb` and the last layer `c_k` in fp16 export by default:
  `MIXED_KEEP_FLOAT_PATTERNS=tok_emb,blocks.10.attn.c_k`

The bigram table is intentionally smaller than the one used in PR `#162`; the goal is to keep the extra context signal while staying inside the 16 MB artifact budget on top of the 11-layer model.

## Default Config

```bash
NUM_LAYERS=11
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=3.0
TRAIN_SEQ_LEN=2048
TRAIN_BATCH_TOKENS=524288
MATRIX_LR=0.025
SCALAR_LR=0.025
TIED_EMBED_LR=0.035
MUON_MOMENTUM=0.99
MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500
WARMDOWN_ITERS=3000
WEIGHT_DECAY=0.038
ADAM_WEIGHT_DECAY=0.01
GRAD_CLIP_NORM=0.3
EVAL_STRIDE=64
SWA_CHECKPOINTS=7
BIGRAM_VOCAB_SIZE=2048
BIGRAM_DIM=96
MIXED_KEEP_FLOAT_PATTERNS=tok_emb,blocks.10.attn.c_k
```

## Reproduction Command

```bash
RUN_ID=11l_bigram_candidate \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Budget Loop

`budget_autoresearch.py` is a small Karpathy-style search driver for this folder. It does not rewrite code; instead it mutates a curated set of env vars, launches one trial per config, parses the final post-quant metrics, ranks results by `final_int6_sliding_window_exact`, and stops when the soft dollar budget is reached.

It keeps each trial in its own workdir under `autotune_runs/` and writes an aggregate leaderboard to `autotune_runs/leaderboard.md`.

Recommended cheap search on a single H100:

```bash
python3 budget_autoresearch.py \
  --budget-dollars 12 \
  --hourly-rate 2.69 \
  --nproc-per-node 1 \
  --max-runs 20
```

The script uses a conservative single-GPU batch by default:

```bash
TRAIN_BATCH_TOKENS=262144
VAL_BATCH_SIZE=131072
VAL_LOSS_EVERY=0
MAX_WALLCLOCK_SECONDS=600
```

If your single GPU comfortably fits more, you can override any env var:

```bash
python3 budget_autoresearch.py \
  --budget-dollars 12 \
  --hourly-rate 2.69 \
  --nproc-per-node 1 \
  --extra-env TRAIN_BATCH_TOKENS=524288 \
  --extra-env VAL_BATCH_SIZE=262144
```

Useful modes:

```bash
python3 budget_autoresearch.py --dry-run
python3 budget_autoresearch.py --reset
```

## Before Submission

- Run at least 3 seeds on 8xH100 SXM.
- Verify `final_int6` artifact size stays under `16,000,000` bytes for every reported seed.
- Check that sliding-window eval time remains inside the extra 10-minute eval budget.
- Create a fresh `submission.json` with real measured results.
- Include only logs produced by this folder's script.
