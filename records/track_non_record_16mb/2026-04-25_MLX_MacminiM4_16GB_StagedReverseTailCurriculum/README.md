This is a non-record Apple Silicon / MLX submission built from a verified local run in `frido22/low_vram_institute`.

It supersedes the earlier Mac mini PR package with the best local result from the overnight search.

## Result

- Hardware: `Mac mini M4 16GB`
- Track: non-record under `records/track_non_record_16mb`
- Verified best run ID: `2026_04_25_run_0004`
- Final exact post-quant score: `val_bpb = 1.51106031`
- Final exact post-quant loss: `val_loss = 3.40875376`
- Pre-quant eval at stop: `val_bpb = 1.5338`, `val_loss = 3.4601`
- Train stop: `773` steps at `542283 ms`
- End-to-end runtime: `624.492459 s`
- Int8+zlib model size: `15,674,932` bytes
- Packaged `train_gpt.py` size: `80,003` bytes
- Packaged total artifact size: `15,754,935` bytes

The included `train.log` is the exact verified `2026_04_25_run_0004` run log. The packaged records-folder script adds only repository-root path hardening relative to the verified source run script, increasing counted code bytes by `95` while leaving the training and evaluation logic unchanged.

## Method Summary

This stays in the compact `SP1024 9x512 KV4` MLX family, with the strongest local recurrent-tail/export branch found on the Mac mini.

Key ingredients:

- `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- tied embeddings with learned output calibration
- rank-64 previous-token bigram adapter in the output path
- two reverse-order recurrent decoder-tail blocks
- staged recurrence curriculum: the deepest recurrent block is emphasized early, with earlier recurrent-tail contribution ramped in later
- decoder-output carry anchoring and aligned decoder skips
- quant-aware endgame with periodic int8 roundtrip blending near wallclock stop
- EMA over float-kept and projection-sensitive tail tensors
- int8 per-row quantization with row offsets and transpose-aware handling for `mlp.fc.weight`
- float-kept tail budget focused on recurrent-tail attention and projection sensitivity instead of keeping a full final block float

The local hypothesis was that the 600-second Mac mini regime benefits from spending early updates on the strongest recurrent exit path, then restoring the full recurrent-tail depth before export/eval. In this run, that beat the previous packaged Mac mini submission by about `0.0089` BPB while staying under the decimal `16,000,000` byte cap.

## Submission Hardening

The records-folder `train_gpt.py` keeps the same import-safety hardening as the previous PR package:

- optional `mlx` imports are guarded at module import time, so the file imports cleanly when `mlx` is absent
- default dataset and tokenizer paths resolve from repository root, so the script can be run directly from the records folder once the repo data has been downloaded in the standard locations

Local checks performed before updating this folder:

- verified source run `2026_04_25_run_0004` with final exact score `1.51106031`
- `py_compile` preflight on the packaged `train_gpt.py`
- top-level import smoke with `numpy` and `sentencepiece` installed but `mlx` absent

## Included Files

- `train_gpt.py` - self-contained submission script
- `train.log` - exact verified run log for `2026_04_25_run_0004`
- `submission.json` - metadata in the style used by existing records folders
- `requirements.txt` - minimal dependency list for this MLX path
