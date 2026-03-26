# 8L / 448 / WD0.04 Width Branch (WIP, non-record)

This is a non-record / WIP submission to `records/track_10min_16mb/`.

## Summary

This branch comes from local 1xGPU search focused on compact-model scaling under the 16MB artifact limit.

Main local finding so far:
- widening from `384 -> 448` at 8 layers beat my stronger `9L / 384` depth branch
- best exact roundtrip came from seed 42

Current best local result:
- `NUM_LAYERS=8`
- `MODEL_DIM=448`
- `NUM_HEADS=8`
- `NUM_KV_HEADS=2`
- `MLP_MULT=1`
- `TIE_EMBEDDINGS=0`
- `MATRIX_LR=0.06`
- `MUON_WEIGHT_DECAY=0.04`
- `WARMDOWN_ITERS=300`
- `EVAL_STRIDE=64`
- `EVAL_BATCH_SEQS=256`

Best exact roundtrip:
- `val_loss = 2.38410966`
- `val_bpb = 1.41200403`

Artifact size:
- total submission size int8+zlib: `7063821` bytes

## Why this branch is interesting

In my local search:
- `8L / 448` width beat `9L / 384` depth
- TTT helps modestly on the winner family, but the main gain is from width placement
- the artifact is still well under the 16,000,000-byte cap

## Included files

- `train_gpt.py`
- `submission.json`
- `train_seed42.log`
- `train_seed1337.log`
- `train_ttt_seed1337.log`

## Status

This is a non-record research PR and compute-grant reference. Official-track 8xH100 validation is still pending.
