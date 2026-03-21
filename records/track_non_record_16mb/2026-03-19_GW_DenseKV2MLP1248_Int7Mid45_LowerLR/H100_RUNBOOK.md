## Goal

Turn this non-record folder into a real `8xH100` leaderboard attempt without violating the challenge rules.

This runbook assumes:

- date context: March 19, 2026
- official merged leaderboard on `main` still shows `1.2244`
- stronger public non-val-only PRs exist, so in practice you should benchmark against those too
- this branch does **not** use val-only training
- this branch does **not** change the tokenizer or dataset

## Rule boundaries

Keep these fixed if the goal is a clean record attempt:

- do not set `TRAIN_ON_VAL`
- do not edit tokenizer or dataset generation
- keep artifact under `16,000,000` total bytes
- keep training under `600s` on `8xH100 SXM`
- keep evaluation under its separate `600s` cap
- collect enough seeds/logs to support the required significance test before making any record claim

## Environment

From a fresh `8xH100 SXM` machine:

```bash
cd /workspace
git clone https://github.com/gwskier11-design/parameter-golf.git
cd parameter-golf
git checkout codex/add-gw-kv2-int7-nonrecord
python3 data/cached_challenge_fineweb.py --variant sp1024
```

If you already have the upstream repo cloned, add this fork as a remote and fetch the branch instead:

```bash
git remote add george https://github.com/gwskier11-design/parameter-golf.git
git fetch george
git checkout george/codex/add-gw-kv2-int7-nonrecord
```

## Primary run

Run from the repo root:

```bash
RUN_ID=gw_dense_kv2_mlp1248_int7mid45_h100_seed1337 \
SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=2 \
MLP_HIDDEN=1248 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
WARMDOWN_ITERS=3600 \
SERIALIZE_TIED_EMBED_FP16=0 \
INT8_KEEP_FLOAT_MAX_NUMEL=65535 \
INT4_LAYERS=4,5 \
INT4_STEP=2 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=8 \
records/track_non_record_16mb/2026-03-19_GW_DenseKV2MLP1248_Int7Mid45_LowerLR/train_gpt.py \
2>&1 | tee \
records/track_non_record_16mb/2026-03-19_GW_DenseKV2MLP1248_Int7Mid45_LowerLR/train_h100_seed1337.log
```

## Repeat seeds

Use at least two more seeds before making any strong claim:

```bash
RUN_ID=gw_dense_kv2_mlp1248_int7mid45_h100_seed42 \
SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=2 \
MLP_HIDDEN=1248 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
WARMDOWN_ITERS=3600 \
SERIALIZE_TIED_EMBED_FP16=0 \
INT8_KEEP_FLOAT_MAX_NUMEL=65535 \
INT4_LAYERS=4,5 \
INT4_STEP=2 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=8 \
records/track_non_record_16mb/2026-03-19_GW_DenseKV2MLP1248_Int7Mid45_LowerLR/train_gpt.py \
2>&1 | tee \
records/track_non_record_16mb/2026-03-19_GW_DenseKV2MLP1248_Int7Mid45_LowerLR/train_h100_seed42.log
```

```bash
RUN_ID=gw_dense_kv2_mlp1248_int7mid45_h100_seed7 \
SEED=7 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=2 \
MLP_HIDDEN=1248 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
WARMDOWN_ITERS=3600 \
SERIALIZE_TIED_EMBED_FP16=0 \
INT8_KEEP_FLOAT_MAX_NUMEL=65535 \
INT4_LAYERS=4,5 \
INT4_STEP=2 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=8 \
records/track_non_record_16mb/2026-03-19_GW_DenseKV2MLP1248_Int7Mid45_LowerLR/train_gpt.py \
2>&1 | tee \
records/track_non_record_16mb/2026-03-19_GW_DenseKV2MLP1248_Int7Mid45_LowerLR/train_h100_seed7.log
```

## Fallbacks

If the primary branch misses on quality or size, try these next:

1. Full-int8 control:

```bash
RUN_ID=gw_dense_kv2_mlp1248_h100_seed1337 \
SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=2 \
MLP_HIDDEN=1248 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
WARMDOWN_ITERS=3600 \
SERIALIZE_TIED_EMBED_FP16=0 \
INT8_KEEP_FLOAT_MAX_NUMEL=65535 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=8 \
records/track_non_record_16mb/2026-03-19_GW_DenseKV2MLP1248_LowerLR/train_gpt.py
```

2. More aggressive size fallback:

```bash
RUN_ID=gw_dense_kv2_mlp1248_int6mid45_h100_seed1337 \
SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=2 \
MLP_HIDDEN=1248 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
WARMDOWN_ITERS=3600 \
SERIALIZE_TIED_EMBED_FP16=0 \
INT8_KEEP_FLOAT_MAX_NUMEL=65535 \
INT4_LAYERS=4,5 \
INT4_STEP=4 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=8 \
records/track_non_record_16mb/2026-03-19_GW_DenseKV2MLP1248_Int7Mid45_LowerLR/train_gpt.py
```

3. Public reference check:

```bash
torchrun --standalone --nproc_per_node=8 \
records/track_non_record_16mb/2026-03-19_GW_10L_MixedPrecision_Reference/train_gpt.py
```

## What to extract from each log

Use these commands after each run:

```bash
rg "stopping_early|step_avg:|Serialized model|Total submission size|final_int8_zlib_roundtrip_exact|peak memory" \
records/track_non_record_16mb/2026-03-19_GW_DenseKV2MLP1248_Int7Mid45_LowerLR/train_h100_seed1337.log
```

At minimum, verify:

- `final_int8_zlib_roundtrip_exact val_bpb:...`
- `Total submission size int8+zlib: ... bytes`
- the training stopped under the wallclock cap
- eval time is under the separate cap

## Promotion checklist

Only promote this folder into `records/track_10min_16mb` if all of the following are true:

1. Artifact bytes are strictly below `16,000,000`.
2. Training is reproducibly below `600s` on `8xH100 SXM`.
3. Evaluation is reproducibly below `600s`.
4. The score is good enough to justify a real record attempt.
5. You have enough seed logs to support the challenge significance rule.

If those conditions hold, then:

- copy this folder into `records/track_10min_16mb/<date>_GW_...`
- replace `train.log` and companion logs with the real H100 logs
- update `submission.json` with the measured H100 numbers
- update `README.md` so every claim is backed by those logs

## What not to change for the first H100 pass

Do not change these on the first real attempt:

- `TRAIN_ON_VAL`
- tokenizer
- dataset
- eval method
- extra helper scripts inside the counted submission path

The first goal is a faithful real-H100 measurement of the current clean candidate.
