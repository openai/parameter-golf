This non-record submission reports a budget-efficient 30-minute 1xH100 run using the baseline script with a scheduler tweak (`WARMDOWN_ITERS=100`).

## Why this run exists
- We are compute constrained on 8xH100 and using 1xH100 to generate reliable signal before larger grant-backed attempts.
- This run tests whether extending wallclock from 10m to 30m on the same 1x recipe produces a meaningful quality jump while staying under the 16MB artifact cap.

## Configuration
- Hardware: `1x H100 80GB`
- Wallclock cap: `1800s`
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Scheduler: `WARMDOWN_ITERS=100`
- Batch/sequence: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`

## Command
```bash
RUN_ID=run5_wd100_30m \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
WARMDOWN_ITERS=100 \
MAX_WALLCLOCK_SECONDS=1800 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=200 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Results
- Stop point: `step 3545` at `1800.187s`
- Pre-quant stop eval: `val_loss=2.1506`, `val_bpb=1.2737`
- Post-quant roundtrip exact: `val_loss=2.15624752`, `val_bpb=1.27705124`
- Serialized model int8+zlib: `15,792,656 bytes`
- Code size: `47,686 bytes`
- Total submission size int8+zlib: `15,840,342 bytes`

## Comparison to today's 10m control
- 10m control (same pod/session): `val_bpb=1.34638424`
- This 30m run: `val_bpb=1.27705124`
- Improvement: `0.06933300`

Included files:
- `train_gpt.py` (exact script snapshot used)
- `train.log` (full run output)
- `submission.json` (metadata)
