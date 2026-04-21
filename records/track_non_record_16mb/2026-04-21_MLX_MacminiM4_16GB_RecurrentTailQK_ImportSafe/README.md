This is a non-record Apple Silicon / MLX submission built from a verified local run in `frido22/low_vram_institute`.

It targets the same Mac mini line as the earlier submission, but improves the reported score materially and hardens the records-folder script for CPU preflight checks.

## Result

- Hardware: `Mac mini M4 16GB`
- Track: non-record under `records/track_non_record_16mb`
- Verified best run ID: `2026_04_21_run_0041`
- Final exact post-quant score: `val_bpb = 1.51996743`
- Final exact post-quant loss: `val_loss = 3.42884704`
- Pre-quant eval at stop: `val_bpb = 1.5396`, `val_loss = 3.4732`
- Train stop: `773` steps at `537886 ms`
- End-to-end runtime: `619.191007 s`
- Int8+zlib model size: `15,672,235` bytes
- Packaged `train_gpt.py` size: `77,032` bytes
- Packaged total artifact size: `15,749,267` bytes

The included `train.log` is the exact verified `2026_04_21_run_0041` run log. The packaged records-folder script adds only records-folder path hardening and a file-name string fix relative to that verified source script, which increases counted code bytes by `91` while leaving the training/eval logic unchanged.

## Method Summary

This submission stays in the same compact `SP1024 9x512 KV4` family, but it pushes more of the surviving float budget into recurrent tail attention geometry instead of keeping a full last block float.

Key ingredients:

- `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- tied embeddings with learned `logit_bias` and `logit_gain`
- rank-64 previous-token bigram adapter in the output path
- two recurrent decoder-tail blocks with learned residual gates
- quant-aware endgame with periodic roundtrip blending near wallclock stop
- EMA over float-kept and projection-sensitive tail tensors during that endgame
- int8 per-row quantization with row offsets and transpose-aware handling for `mlp.fc.weight`
- float-kept tail budget reallocated toward recurrent-tail attention precision:
  - keep Q and K float in both recurrent tail blocks
  - keep final-block V float
  - keep the last two projection matrices float
  - keep small control tensors float

The local hypothesis was that the final int8+zlib score is more sensitive to recurrent tail attention geometry than to spending the same bytes on a broader fp16 tail block rescue. In practice, this beat the previous Mac mini submission by about `0.0472` BPB while staying under the decimal `16,000,000` byte cap.

## Submission Hardening

The records-folder `train_gpt.py` adds two packaging-only hardening changes relative to the verified source run script:

- optional `mlx` imports are guarded at module import time, so the file still imports cleanly when `mlx` is absent
- default dataset and tokenizer paths resolve from repository root, so the script can be run directly from the records folder once the repo data has been downloaded in the standard locations

Local checks performed before preparing this folder:

- fresh 10-minute rerun in the source repo using the import-safe script, which produced the reported `1.51996743`
- `py_compile` preflight on the packaged `train_gpt.py` under Python `3.10`
- top-level import smoke on Python `3.10` with `numpy` and `sentencepiece` installed but `mlx` absent, verifying that import succeeds and `_OPTIONAL_IMPORT_ERROR` is set instead of crashing at import time

## Included Files

- `train_gpt.py` — the self-contained submission script
- `train.log` — exact verified run log for `2026_04_21_run_0041`
- `submission.json` — metadata in the style used by existing records folders
- `requirements.txt` — minimal dependency list for this MLX path
