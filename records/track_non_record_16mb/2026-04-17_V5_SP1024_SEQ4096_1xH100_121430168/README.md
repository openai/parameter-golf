# Non-Record Submission: V5 SP1024 + Seq4096 (1xH100)

## Overview

This directory contains a **non-record** 16MB submission for OpenAI Parameter Golf.
The run uses the official FineWeb SentencePiece-1024 tokenization path with
`TRAIN_SEQ_LEN=4096` on a single H100 GPU.

- Run name / ID: `v5_sp1024_top10_a`
- Track: `non-record-16mb`
- Hardware: `1xH100`
- Stop step: `6000`
- Wallclock (recorded): `3218` seconds

## Why this is non-record

This run is explicitly submitted under **non-record-16mb** and does not claim
record-track status. It was not executed under the official 10-minute / 8xH100
record constraint.

## Exact metrics

Final post-quantized round-trip validation metrics (exact values):

- `val_loss`: **2.05029752**
- `val_bpb`: **1.21430168**

Submission size accounting:

- int8+zlib model size: **15793702** bytes
- code size: **47686** bytes
- total submission size: **15841388** bytes

Additional run values:

- `step_stop`: **6000**
- In-log pre-roundtrip evaluation at step 6000:
  - `val_loss`: `2.0411`
  - `val_bpb`: `1.2089`

## Training configuration

From the run log and metadata:

- Dataset path pattern: `./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin`
- Validation path pattern: `./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin`
- Tokenizer: `./data/tokenizers/fineweb_1024_bpe.model`
- `TRAIN_SEQ_LEN=4096`
- `TRAIN_BATCH_TOKENS=524288`
- `ITERATIONS=6000`
- `WARMUP_STEPS=30`
- Seed: `1337`

## Model architecture

The training code used for this run defines a GPT-style model with:

- Vocabulary size: `1024`
- Transformer layers: `9`
- Model dimension: `512`
- Attention heads: `8`
- KV heads (GQA): `4`
- MLP multiplier: `2`
- Tied embeddings: enabled
- Reported parameter count in run log: `17059912`

## Quantization / serialization

The final model artifact is stored as an int8 + zlib-compressed payload.

- Serialized FP model size (from log): `67224983` bytes
- Serialized int8+zlib model size: `15793702` bytes
- Total submission size int8+zlib: `15841388` bytes

## Included files

- `README.md` — submission documentation
- `submission.json` — metadata for leaderboard ingestion
- `results.tsv` — tabular metrics row
- `train_gpt.py` — run-local training script snapshot
- `train.log` — recovered run log (script + execution output)
- `final_model.int8.ptz` — final quantized artifact

## Limitations / next steps

- This is intentionally a non-record run and should not be compared as a
  record-track attempt.
- The recovered `train.log` includes both script text and execution output in
  one file.
- Potential follow-up work:
  - add a cleaner separated stdout log artifact format,
  - add repeated-seed non-record runs for variance estimates,
  - continue tuning for lower post-quant `val_bpb` while staying under the
    16MB cap.
