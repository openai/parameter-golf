This non-record submission documents the merged staging-profile behavior on `8xH100` with LAWA enabled and sliding-window evaluation enabled.

It does **not** beat the current leaderboard SOTA (e.g. `Muon WD + 10 layer` at `val_bpb=1.1748`); the main purpose is reproducible validation of the `STAGING_PROFILE=1` recipe at production scale.

## Configuration summary

- Staging profile: `STAGING_PROFILE=1`
  - Enables LAWA (`LAWA_ENABLED=1`)
  - Enables sliding-window eval (`EVAL_STRIDE=512`)
  - Applies merged-baseline defaults (e.g. `NUM_LAYERS=10`, `WARMDOWN_ITERS=2500`, `TIED_EMBED_LR=0.10`, `MUON_WEIGHT_DECAY=0.02`, `ADAM_WEIGHT_DECAY=0.01`)
- Training
  - Dataset: `fineweb10B_sp1024` (SP-1024 tokenizer)
  - Batch/shape: `TRAIN_SEQ_LEN=1024`, `TRAIN_BATCH_TOKENS=524288`
  - Trainer: `train_gpt.py` snapshot + dependencies from the run
  - Hardware: `8xH100`, run stopped by `MAX_WALLCLOCK_SECONDS=600`

## Results (from `train.log`)

- Scored metric (int8+zlib roundtrip exact): `final_int8_zlib_roundtrip_exact val_bpb:1.18926428`
- Matching loss: `val_loss:2.00802292`
- Pre-quant eval at stop: `quick_metric ... val_bpb:1.21846263`
- TTT LoRA eval (reported in log): `final_int8_ttt_lora val_bpb:1.1863`
- Steps/wallclock
  - Stopped at `11436/20000` (wallclock cap): `train_time_ms:599955` (~`599.955s`)
- Artifact size (challenge cap): `Total submission size int8+zlib: 15292665 bytes`

## Command (repro)

```bash
STAGING_PROFILE=1 RUN_ID=prod_8xh100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included files

- `train_gpt.py`, `train_gpt_lawa.py`, `train_gpt_sliding.py` (trainer snapshot + dependencies)
- `train.log` (full remote training log)
- `submission.json` (leaderboard metadata)

