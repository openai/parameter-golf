# Non-Record Submission: 15L Ternary MLP-Only (20k)

This is an unlimited-compute non-record submission for the 16MB track.

It captures the cleanest mature result from the pure ternary MLP-only branch:
- `15` layers
- MLP-only ternary quantization
- no ternary attention
- `20k` training steps
- under-cap final export

This run is not intended to satisfy the record-track `8xH100` 10-minute requirement. It is submitted as a non-record result because it is a distinct pure-ternary design that finished cleanly under the artifact cap and provides a useful reference point for future ternary work.

## Result

- Final roundtrip: `val_loss=2.20139220`, `val_bpb=1.30378846`
- Pre-export quant proxy: `val_loss=2.20080222`, `val_bpb=1.30343904`
- Artifact size: `15,037,372` bytes
- Compressed model size: `14,957,135` bytes
- Code size: `80,237` bytes
- Step stop: `20,000`
- Wallclock: `23,469.558s`
- GPU: `1xRTX5060`

## Why This Submission Is Interesting

- It is the strongest finished pure ternary MLP-only run in this repo history.
- The run showed a tight quant-proxy-to-roundtrip match, which made the ternary branch trustworthy rather than just a size-only curiosity.
- It stays comfortably under the cap while using a much more aggressive quantization regime than the mixed-lowbit baseline family.

## Configuration Summary

- Layout: `NUM_LAYERS=15 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3.0`
- Sequence length: `TRAIN_SEQ_LEN=1024`
- Batch tokens: `TRAIN_BATCH_TOKENS=65536`
- Ternary path: `TERNARY_QAT=1 TERNARY_EXPORT=1 TERNARY_TARGETS=mlp`
- Export: `BIGRAM_EXPORT_BITS=5 MAG_PRUNE_PCT=0.032`

## Included Files

- `train_gpt.py` — code snapshot used for the submitted run
- `train.log` — training and final evaluation log
- `submission.json` — metadata for the non-record submission
- `requirements.txt` — dependency reference
