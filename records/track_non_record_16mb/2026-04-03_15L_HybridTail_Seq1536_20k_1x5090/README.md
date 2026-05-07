# Non-Record Submission: 15L Hybrid Tail Ternary + Seq1536 (20k, 1x5090)

This is an unlimited-compute non-record submission for the 16MB track.

It captures the strongest finished run from the ternary/hybrid branch:
- `15` layers
- MLP-only ternary path
- targeted hybrid precision on the tail MLP block
- `TRAIN_SEQ_LEN=1536`
- trained for `20k` steps on a single `RTX 5090`

This run is not intended to satisfy the record-track `8xH100` 10-minute constraint. It is submitted as a non-record result because it represents a distinct architecture/compression direction and remained under the `16,000,000` byte cap with a strong final score.

## Result

- Final roundtrip: `val_loss=2.07398758`, `val_bpb=1.22833292`
- Pre-export quant proxy: `val_loss=2.07253893`, `val_bpb=1.22747495`
- Artifact size: `15,669,833` bytes
- Compressed model size: `15,574,237` bytes
- Code size: `95,596` bytes
- Step stop: `20,000`
- Wallclock: `11,684.835s`
- GPU: `1xRTX5090`

## Why This Submission Is Interesting

- It shows that a ternary branch can get very close to the stronger mixed-lowbit family without giving up the compression-first character of the design.
- The key idea is selective precision protection: the model keeps an MLP-only ternary path while promoting the most sensitive tail MLP block.
- Increasing training context to `1536` was important for this branch and transferred well from short proxy runs to a longer `20k` cloud run.

## Configuration Summary

- Layout: `NUM_LAYERS=15 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3.0`
- Sequence length: `TRAIN_SEQ_LEN=1536`
- Batch tokens: `TRAIN_BATCH_TOKENS=122880`
- Ternary path: `TERNARY_QAT=1 TERNARY_EXPORT=1 TERNARY_TARGETS=mlp`
- Tail promotion: `TERNARY_EXCLUDE_NAME_PATTERNS=blocks.14.mlp.`
- Bit overrides: `QAT_BITS_OVERRIDES=blocks.14.mlp.=5 EXPORT_BITS_OVERRIDES=blocks.14.mlp.=5`
- Export: `BIGRAM_EXPORT_BITS=5 MAG_PRUNE_PCT=0.032`

## Included Files

- `train_gpt.py` — code snapshot used for the submitted run
- `train.log` — training and final evaluation log
- `submission.json` — metadata for the non-record submission
- `requirements.txt` — dependency reference
