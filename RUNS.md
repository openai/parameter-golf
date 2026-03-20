# RUNS

These are the 3 highest-value CUDA experiments for the current repo state. They isolate whether the new quantization path helps enough on artifact size / roundtrip `val_bpb` to justify carrying it forward.

## Common setup

Assumptions:
- repo lives at `/workspace/parameter-golf` on the remote box
- dataset/tokenizer were downloaded with `python3 data/cached_challenge_fineweb.py --variant sp1024`
- Python deps from the official Parameter Golf image are already present

Common env:

```bash
export REPO_DIR=/workspace/parameter-golf
export DATA_PATH=$REPO_DIR/data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=$REPO_DIR/data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024
export NPROC_PER_NODE=1   # use 8 for final 8xH100 timing later
export MAX_WALLCLOCK_SECONDS=600
export VAL_LOSS_EVERY=200
export TRAIN_LOG_EVERY=100
```

Each run writes a log under `logs/remote_runs/` and should end with:
- `Serialized model int8+zlib:`
- `Total submission size int8+zlib:`
- `final_int8_zlib_roundtrip val_loss:... val_bpb:...`

Primary comparison metrics:
1. `final_int8_zlib_roundtrip val_bpb` (lower is better)
2. `Total submission size int8+zlib` (must stay safely under 16,000,000 bytes)
3. `outlier_cols:` bytes inside the int8 payload log line
4. effective train speed (`step_avg`, whether the 10 min cap truncates training too early)

## Experiment 1: control / old behavior inside new code

Purpose: recover the pre-change per-row quantization behavior as the control.

```bash
cd "$REPO_DIR"
RUN_ID=control_per_row \
INT8_GROUP_SIZE=999999 \
INT8_OUTLIER_COLS=0 \
INT8_CLIP_PERCENTILE=99.99984 \
DATA_PATH="$DATA_PATH" TOKENIZER_PATH="$TOKENIZER_PATH" VOCAB_SIZE=$VOCAB_SIZE \
MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS VAL_LOSS_EVERY=$VAL_LOSS_EVERY TRAIN_LOG_EVERY=$TRAIN_LOG_EVERY \
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE train_gpt.py
```

Go if this reproduces the expected baseline neighborhood (~1.2 `val_bpb`, artifact under cap). Stop if the run is already badly off, because later comparisons will be noisy.

## Experiment 2: grouped scales only

Purpose: test whether per-row groupwise scales buy better size/quality tradeoff without extra outlier payload.

```bash
cd "$REPO_DIR"
RUN_ID=group64 \
INT8_GROUP_SIZE=64 \
INT8_OUTLIER_COLS=0 \
INT8_CLIP_PERCENTILE=99.99984 \
DATA_PATH="$DATA_PATH" TOKENIZER_PATH="$TOKENIZER_PATH" VOCAB_SIZE=$VOCAB_SIZE \
MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS VAL_LOSS_EVERY=$VAL_LOSS_EVERY TRAIN_LOG_EVERY=$TRAIN_LOG_EVERY \
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE train_gpt.py
```

Go if either:
- `val_bpb` improves by at least `0.002` vs control with submission size still < `15.8MB`, or
- submission size drops by at least `100KB` with `val_bpb` regression <= `0.001`

Stop if `val_bpb` worsens by > `0.003` and size is not meaningfully better.

## Experiment 3: grouped scales + saved outlier columns

Purpose: spend a little payload on the most sensitive columns to see if the quality recovered per extra byte is worth it.

```bash
cd "$REPO_DIR"
RUN_ID=group64_outliers8 \
INT8_GROUP_SIZE=64 \
INT8_OUTLIER_COLS=8 \
INT8_OUTLIER_MIN_ROWS=256 \
INT8_OUTLIER_NAME_PATTERNS=tok_emb,lm_head,c_q,c_k,c_v,proj,fc \
INT8_CLIP_PERCENTILE=99.99984 \
DATA_PATH="$DATA_PATH" TOKENIZER_PATH="$TOKENIZER_PATH" VOCAB_SIZE=$VOCAB_SIZE \
MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS VAL_LOSS_EVERY=$VAL_LOSS_EVERY TRAIN_LOG_EVERY=$TRAIN_LOG_EVERY \
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE train_gpt.py
```

Go if `val_bpb` beats Experiment 2 by at least `0.0015` while staying below `15.9MB` total submission size. Stop if `outlier_cols` adds noticeable bytes but `val_bpb` barely moves (`<0.0008`).

## Recommended order

1. `control_per_row`
2. `group64`
3. `group64_outliers8`

If `group64` wins clearly, skip more outlier sweeps and move straight to a small 8-GPU timing confirmation. If `group64_outliers8` wins, the next sweep should be only `INT8_OUTLIER_COLS in {4, 8, 16}`.

## Fast result extraction

```bash
cd "$REPO_DIR"
./scripts/run_remote_experiment.sh summary logs/remote_runs/*.log
```
