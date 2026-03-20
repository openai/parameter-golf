This non-record submission captures the first official-style FineWeb smoke run of **SMM-LM** inside the `parameter-golf` evaluation path.

## Summary

This run adapts a minimal SMM-style structured-memory language model into the official `train_gpt.py` workflow and evaluates it on the cached FineWeb validation split using the repository's `val_bpb` path.

This is **not** a 10-minute 8xH100 leaderboard attempt. It is a **single-GPU smoke submission** intended to show that:

- the SMM path runs end-to-end inside the official training/evaluation script,
- `val_bpb` is computed through the official FineWeb validation logic,
- model quantization and artifact-size accounting still work,
- the compressed submission remains under the 16,000,000-byte artifact limit,
- and the SMM direction remains competitive against a baseline under the same smoke regime.

## Setup

Hardware / execution regime:

- Single H100 80GB
- Runpod remote machine
- FineWeb cached subset with `--train-shards 1`

Main SMM configuration:

- `MODEL_KIND=smm`
- `NUM_LAYERS=6`
- `MODEL_DIM=256`
- `NUM_HEADS=4`
- `NUM_KV_HEADS=4`
- `MLP_MULT=2`
- `SMM_NUM_MEMORY_SLOTS=64`
- `SMM_MEMORY_VALUE_DIM=256`
- `SMM_TOP_K=8`
- `SMM_MEMORY_INIT=freq`
- `SMM_FREEZE_MEMORY=1`
- `SMM_INIT_TOKEN_BUDGET=500000`
- `ITERATIONS=300`
- `WARMUP_STEPS=5`
- `RUN_TTT_EVAL=0`

The current SMM migration does not yet implement the upstream LoRA test-time-training path, so this submission uses the common, directly comparable metric:

- `final_int8_zlib_roundtrip_exact val_bpb`

## Main Result

SMM smoke run:

- `model_params: 3,776,088`
- `final_int8_zlib_roundtrip_exact val_loss: 2.99375746`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.77307180`
- `Total submission size int8+zlib: 4,509,767 bytes`

This is well below the 16,000,000-byte cap.

## Baseline Reference

A baseline run under the same smoke regime was also executed for comparison:

- `MODEL_KIND=baseline`
- same `NUM_LAYERS=6`, `MODEL_DIM=256`, `NUM_HEADS=4`, `NUM_KV_HEADS=4`, `MLP_MULT=2`
- `ITERATIONS=300`
- `RUN_TTT_EVAL=0`

Baseline reference result:

- `model_params: 3,414,808`
- `final_int8_zlib_roundtrip_exact val_loss: 3.01110288`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.78334473`
- `Total submission size int8+zlib: 4,036,102 bytes`

In this smoke regime, the SMM path improves over the baseline by about:

- `1.78334473 - 1.77307180 = 0.01027293 bpb`

## Interpretation

This result should be interpreted conservatively.

- It is a **non-record** run.
- It uses only **1 training shard** of cached FineWeb.
- It is **single-GPU**, not the official 8xH100 / 10-minute regime.
- It demonstrates that a minimal structured-memory path can be inserted into the official `parameter-golf` workflow without breaking FineWeb evaluation, quantization, or artifact accounting.

The result is encouraging because the direction remains favorable even in the official evaluation path, but this run should be viewed as a migration milestone rather than a leaderboard claim.

## Included Files

- `train_gpt.py`: the adapted official-style training script with a minimal SMM branch
- `train.log`: the exact SMM smoke training log
- `baseline_reference.log`: the baseline reference log for the same smoke regime
- `submission.json`: metadata for this non-record result
