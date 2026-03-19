This non-record submission reports a 1-hour run on 1xH100 to demonstrate substantial quality gains from longer 1-GPU training while staying under the 16MB artifact cap.

Compute note:
- This run uses `1x H100` due to limited compute budget while waiting for grant-backed `8xH100` record attempts.
- Since this run exceeds 10 minutes, it is intended for the non-record unlimited-compute track.

Approach:
- Keep baseline architecture and data/tokenizer pipeline fixed (`9x512`, `sp1024`).
- Use `WARMDOWN_ITERS=100` to keep LR decay focused near the end of training.
- Extend wallclock to 1 hour to test scaling behavior on a single GPU.

Configuration:
- Hardware: `1x H100 80GB`
- Wallclock cap: `3600s`
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Scheduler change: `WARMDOWN_ITERS=100`
- Batch/sequence: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`

Command used:
```bash
RUN_ID=grant_push_1h_wd100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
WARMDOWN_ITERS=100 \
MAX_WALLCLOCK_SECONDS=3600 \
VAL_LOSS_EVERY=500 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Results:
- Stop point: `step 7435` at `3600.152s`
- Pre-quant stop eval: `val_loss=2.1071`, `val_bpb=1.2480`
- Post-quant roundtrip exact: `val_loss=2.11737037`, `val_bpb=1.25402600`
- Serialized model int8+zlib: `15,810,866 bytes`
- Code size: `47,686 bytes`
- Total submission size int8+zlib: `15,858,552 bytes`

Comparison to earlier same-session 1xH100 10-minute run:
- Previous best 10-minute run: `val_bpb=1.34862689`
- This 1-hour run: `val_bpb=1.25402600`
- Improvement: `0.09460089`

Included files:
- `train_gpt.py` (exact script snapshot used)
- `train.log` (full training/eval output)
- `submission.json` (metadata)
