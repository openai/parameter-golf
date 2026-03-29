# Experiment 01: Mixed Export

This mini-project keeps the training graph identical to the baseline and changes only the serialization policy.

## Change

- baseline export: mostly per-row int8 plus float passthrough
- experiment export: large MLP matrices matching `INT4_TARGET_NAME_PATTERNS` are symmetrically quantized to int4 and packed two values per byte
- the rest of the tensors keep the baseline-style int8 or float passthrough behavior

## Why This Exists

This is the cleanest way to test the artifact-first hypothesis:

- if the mixed export gives smaller artifacts at similar roundtrip `val_bpb`, it is promising
- if the roundtrip loss cliff is too large, we can kill the direction without changing the training model

## Main Knobs

- `INT4_TARGET_NAME_PATTERNS`
- `INT4_CLIP_PERCENTILE`
- `INT8_CLIP_PERCENTILE`
- `INT8_KEEP_FLOAT_MAX_NUMEL`
- `WARMDOWN_ITERS`
- `WARMUP_STEPS`

## Optuna-Friendly Surface

The best search surface here is small:

- `INT4_CLIP_PERCENTILE`
- `INT8_KEEP_FLOAT_MAX_NUMEL`
- `WARMDOWN_ITERS`
- `WARMUP_STEPS`

Avoid turning this into a huge discrete search over pattern strings at first.
