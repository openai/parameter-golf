# PR update plan for AntDX316 submission

This branch now includes:

- `train_gpt.py` with optional sliding-window validation
- `scripts/run_8xh100_submission.sh` with 8xH100-friendly defaults

## Recommended first rerun

```bash
bash scripts/run_8xh100_submission.sh
```

Defaults used by the script:

- `NUM_LAYERS=12 MODEL_DIM=416 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- `TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=524288`
- `MAX_WALLCLOCK_SECONDS=600`
- `EVAL_SEQ_LEN=4096`
- `SLIDING_WINDOW_STRIDE=64`

## What to look for in the log

```bash
grep -E 'eval_seq_len:|sliding_window_stride:|Total submission size int8\+zlib:|final_int8_zlib_roundtrip |final_int8_zlib_roundtrip_exact ' logs/<RUN_ID>.txt
```

## Updating the PR record folder after a successful rerun

1. Create a new dated folder under `records/track_10min_16mb/`
2. Copy in:
   - `train_gpt.py`
   - `logs/<RUN_ID>.txt` as `train.log`
   - a fresh `submission.json`
   - a `README.md` describing the exact run settings and metrics
3. Commit and push to `antdx316-parameter-golf-submission`
4. Add a PR comment noting the new `final_int8_zlib_roundtrip_exact val_bpb`

## Suggested PR comment

> Update: reran the submission with sliding-window validation (`EVAL_SEQ_LEN=4096`, `SLIDING_WINDOW_STRIDE=64`) on 8xH100. New exact roundtrip metric: `val_bpb=...`, `val_loss=...`, total submission size `...` bytes. Added fresh logs and updated submission metadata.
