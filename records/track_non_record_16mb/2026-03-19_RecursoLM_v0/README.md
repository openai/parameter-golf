# RecursoLM v0 (Non-record 16MB track)

This run implements a recurrent, parameter-tied language model intended for the OpenAI Parameter Golf challenge.

## Summary
- 2 unique transformer blocks reused recurrently (`RECURRENCE_STEPS=16` by default)
- Model width `MODEL_DIM=384`
- GQA configured as multi-query attention (`NUM_HEADS=4`, `NUM_KV_HEADS=1`)
- ALiBi attention bias (no learned positional embedding table)
- SwiGLU MLP with fixed hidden width (`MLP_DIM=1024`)
- Learned recurrence gate: `h = gate * h + (1 - gate) * h_prev`
- Tied token embedding and output projection
- Official challenge int8 + zlib roundtrip export retained

## Why this approach
The design trades stored parameters for iterative depth by sharing block weights across recurrence steps. This fits the 16MB artifact budget while still increasing effective compute depth.

## Data source
Use challenge-provided cached FineWeb shards and tokenizer only:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

## Repro command (1 GPU smoke)
From repo root:

```bash
RUN_ID=recurso_v0_smoke \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TRAIN_SEQ_LEN=512 \
MODEL_DIM=384 \
NUM_LAYERS=2 \
RECURRENCE_STEPS=16 \
NUM_HEADS=4 \
NUM_KV_HEADS=1 \
MLP_DIM=1024 \
ITERATIONS=400 \
VAL_LOSS_EVERY=0 \
TRAIN_BATCH_TOKENS=262144 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 records/track_non_record_16mb/2026-03-19_RecursoLM_v0/train_gpt.py
```

## 8xH100 intended launch

```bash
RUN_ID=recurso_v0_8xh100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TRAIN_SEQ_LEN=512 \
MODEL_DIM=384 \
NUM_LAYERS=2 \
RECURRENCE_STEPS=16 \
NUM_HEADS=4 \
NUM_KV_HEADS=1 \
MLP_DIM=1024 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 records/track_non_record_16mb/2026-03-19_RecursoLM_v0/train_gpt.py
```

## Notes
- This is an initial implementation run and has not yet been fully benchmarked on 8xH100.
- Tokenizer modifications are intentionally avoided for this first version to keep BPB accounting straightforward and reviewer-friendly.
