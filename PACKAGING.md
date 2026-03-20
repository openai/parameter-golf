# PACKAGING

This repo is still based on upstream `openai/parameter-golf`, with only two code files modified:
- `train_gpt.py`
- `train_gpt_mlx.py`

## What changed vs upstream

The changes are all in the post-training int8 export / roundtrip path:

1. `INT8_GROUP_SIZE` is now configurable from the environment.
   - If a matrix width is divisible by the group size, export uses grouped per-row scales.
   - If not divisible, it falls back to the previous per-row scale behavior.

2. `INT8_CLIP_PERCENTILE` is now configurable from the environment.

3. Optional outlier-column preservation was added.
   - `INT8_OUTLIER_COLS`
   - `INT8_OUTLIER_MIN_ROWS`
   - `INT8_OUTLIER_NAME_PATTERNS`
   
   Matching 2D tensors can keep their top-energy columns in fp16 while zeroing them out before int8 quantization.

4. Export stats now report `outlier_cols:` bytes so the extra payload is visible in logs.

5. Matching logic was implemented in both PyTorch and MLX so local MLX iteration and CUDA runs stay aligned.

## How to verify artifact-size and `val_bpb`

Run one of the commands from `RUNS.md` or the helper script.

In the log, capture these lines:

```text
Serialized model int8+zlib: ...
Total submission size int8+zlib: ...
final_int8_zlib_roundtrip val_loss:... val_bpb:...
```

What to check:
- `Total submission size int8+zlib` must be `< 16000000`
- `final_int8_zlib_roundtrip val_bpb` is the score to compare across experiments
- `outlier_cols:` should justify itself with a real `val_bpb` gain

## Useful one-liners

Artifact size and roundtrip score from a single log:

```bash
grep -E 'Serialized model int8\+zlib:|Total submission size int8\+zlib:|final_int8_zlib_roundtrip ' logs/remote_runs/<run>.log
```

Patch for transfer:

```bash
git diff > handoff.patch
```

Branch-ready check:

```bash
git status --short
git diff --stat
```

## Minimal handoff contents

To move this work elsewhere, copy:
- `train_gpt.py`
- `train_gpt_mlx.py`
- `RUNS.md`
- `PACKAGING.md`
- `scripts/run_remote_experiment.sh`
- optionally `handoff.patch`
