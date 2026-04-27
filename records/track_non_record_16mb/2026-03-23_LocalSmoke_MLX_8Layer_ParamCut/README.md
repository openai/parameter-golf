## Non-record submission: Local smoke baseline + 8-layer param cut (MLX)

This submission is a first end-to-end PR artifact for the challenge workflow:
- run baseline
- make one small, real model change
- report results and include reproducible files

Because this was executed on Apple Silicon MLX with a tiny local smoke shard, it is **not leaderboard-comparable** to the official 8xH100 full-validation track. It is intended as a reproducibility and iteration bootstrap.

### What changed

One architecture change:
- `NUM_LAYERS: 9 -> 8` (all other model defaults unchanged)

### Why this change

Reducing depth is one of the simplest levers under strict artifact and runtime limits. This first attempt checks the speed/size trade-off before trying more advanced ideas.

### Local setup used

- Hardware: Apple Silicon (MLX backend)
- Dataset: local smoke shard with valid challenge shard header format
  - train tokens: 2,000,000
  - val tokens: 1,000,000
- Tokenizer: `data/tokenizers/fineweb_1024_bpe.model`

### Commands

Baseline:
```bash
RUN_ID=mlx_smoke_baseline_local \
DATA_PATH=./data/datasets/fineweb10B_sp1024_smoke \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
ITERATIONS=60 \
TRAIN_BATCH_TOKENS=8192 \
GRAD_ACCUM_STEPS=1 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=131072 \
python3 train_gpt_mlx.py
```

Submission run (8 layers):
```bash
RUN_ID=mlx_smoke_layers8_local \
DATA_PATH=./data/datasets/fineweb10B_sp1024_smoke \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
ITERATIONS=60 \
TRAIN_BATCH_TOKENS=8192 \
GRAD_ACCUM_STEPS=1 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=131072 \
NUM_LAYERS=8 \
python3 train_gpt_mlx.py
```

### Results

Baseline (`logs_baseline.txt`):
- final `val_bpb`: `2.74767562`
- int8+zlib model bytes: `9,283,342`
- train time to step 60: `42,500ms`

8-layer run (`train.log`):
- final `val_bpb`: `2.75529448`
- int8+zlib model bytes: `8,451,732`
- train time to step 60: `35,198ms`

Observed trade-off in this smoke setup:
- faster training and smaller model artifact
- slight regression in held-out bpb

### Included files

- `train.log`: full log for the 8-layer submission run
- `logs_baseline.txt`: baseline comparison run log
- `submission.json`: metadata for this non-record run
- `train_gpt.py`: small launcher pinned to this run config

