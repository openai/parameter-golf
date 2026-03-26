# Non-Record Submission: MOEA Outer-Loop Proxy F2 on 1xRTX 3090

This is a non-record 16MB submission documenting the first artifact handoff from an offline multi-objective evolutionary search workflow into a runnable `parameter-golf` submission folder.

The optimizer itself is not part of the artifact. It was used offline to search over architecture, training, and systems choices, then distill a local proxy candidate into a self-contained `train_gpt.py` snapshot.

This run is intentionally not a leaderboard attempt. It is a single-GPU local proxy run with compile disabled and a reduced validation window (`VAL_TOKEN_LIMIT=1048576`) so the full train -> export -> roundtrip eval path closes on local hardware.

## Why This Is Non-Record

- It was run on `1x RTX 3090`, not `8x H100`.
- It uses a reduced validation window for an F2-style proxy metric, not the full challenge validation sweep.
- It is meant to prove the search-to-artifact workflow and provide a credible first PR / compute-credit reference point, not claim SOTA.

## Method

- Offline MOEA outer loop searched over compact backbone and runtime-safe training settings.
- The distilled artifact here uses a small GQA transformer with tied embeddings and post-training int8+zlib export.
- No test-time training is used in this submission.
- `ENABLE_COMPILE=0` was added in the record-local script so the proxy path is runnable on this box.

## Configuration

- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=6 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=2`
- Embeddings: `TIE_EMBEDDINGS=1`
- Training: `ITERATIONS=12 WARMUP_STEPS=1 WARMDOWN_ITERS=4`
- Batch / context: `TRAIN_BATCH_TOKENS=8192 TRAIN_SEQ_LEN=256 VAL_BATCH_SIZE=16384`
- Proxy eval scope: `VAL_TOKEN_LIMIT=1048576`
- Systems: `ENABLE_COMPILE=0 CUDA_VISIBLE_DEVICES=0`

## Command

```bash
TMPDIR=/tmp \
CUDA_VISIBLE_DEVICES=0 \
RUN_ID=moea_outerloop_proxy_f2_1x3090 \
DATA_PATH=/mnt/c/users/wes/desktop/openai_parameter_golf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/mnt/c/users/wes/desktop/openai_parameter_golf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 NUM_LAYERS=6 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=2 \
TIE_EMBEDDINGS=1 ITERATIONS=12 WARMUP_STEPS=1 WARMDOWN_ITERS=4 \
TRAIN_BATCH_TOKENS=8192 TRAIN_SEQ_LEN=256 VAL_BATCH_SIZE=16384 VAL_TOKEN_LIMIT=1048576 \
VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=6 MAX_WALLCLOCK_SECONDS=0 ENABLE_COMPILE=0 \
/home/wesunix/miniconda3/envs/py311/bin/python train_gpt.py
```

## Key Metrics

- Pre-quant proxy eval: `val_loss:5.8015`, `val_bpb:3.4759`
- Post-quant roundtrip proxy eval: `val_loss:5.80268220`, `val_bpb:3.47665807`
- Train time: `7879ms`
- Final roundtrip eval time: `9729ms`
- Peak memory: `166 MiB allocated`, `190 MiB reserved`
- Serialized model int8+zlib: `3234871 bytes`
- Code size: `48475 bytes`
- Total submission size int8+zlib: `3283346 bytes`

## Included Files

- `train_gpt.py` - record-local code snapshot used for the run
- `train.log` - exact proxy training log for this submission
- `submission.json` - metadata for the non-record submission
