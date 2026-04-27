# In-Place TTT with Per-Document Reset

This is an experimental non-record submission based on the baseline `train_gpt.py`.

## Summary

This variant keeps baseline training unchanged and adds an extra final evaluation pass that performs in-place test-time training on `mlp.proj.weight`, resetting those fast weights at each document boundary detected by BOS markers in the validation stream.

The goal was to test whether direct adaptation of existing MLP projection weights can improve validation compression without changing the saved artifact.

## What Changed

- Baseline training path is unchanged.
- Normal final `final_int8_zlib_roundtrip` evaluation still runs first.
- An additional `final_int8_inplace_ttt` pass runs afterward when `INPLACE_TTT_ENABLE=1`.
- Fast weights are the per-block `mlp.proj.weight` tensors.
- Fast weights are reset at each validation document boundary.
- Each chunk is scored before adaptation, so updates only help later chunks within the same document.

## Key Settings

- `INPLACE_TTT_ENABLE=1`
- `INPLACE_TTT_TARGET=mlp.proj`
- `INPLACE_TTT_CHUNK_SIZE=1024`
- `INPLACE_TTT_EVAL_SEQ_LEN=1024`
- `INPLACE_TTT_LR=1e-3`

## Result

Smoke run result:

- `final_int8_zlib_roundtrip val_loss: 5.80955353`
- `final_int8_zlib_roundtrip val_bpb: 3.44074483`
- `final_int8_inplace_ttt val_loss: 5.73910452`
- `final_int8_inplace_ttt val_bpb: 3.39902097`
- `Total submission size int8+zlib: 5086138 bytes`

This shows the in-place TTT pass improved the metric on this smoke run, but the implementation is currently far too slow to be competitive for the 10-minute evaluation budget.

## Reproduction

Run from the repository root:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

RUN_ID=inplace_ttt_smoke \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=20 \
WARMUP_STEPS=2 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=0 \
INPLACE_TTT_ENABLE=1 \
INPLACE_TTT_TARGET=mlp.proj \
INPLACE_TTT_CHUNK_SIZE=1024 \
INPLACE_TTT_EVAL_SEQ_LEN=1024 \
INPLACE_TTT_LR=1e-3 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Notes

This is an experimental prototype, not a reproduction of the In-Place TTT paper and not a leaderboard submission.
