# Aria Local Non-Record: 8x576 KV4 i1200 rope100k (full val, pre-roundtrip)

This non-record run captures a full-validation result with a tuned RoPE base (`ROPE_BASE=100000`) on top of the 8x576 KV4 setup.

## Summary

- Track: `non-record-unlimited-compute-16mb`
- Dataset: `fineweb10B_sp1024` (full fixed validation split)
- Model shape: `NUM_LAYERS=8`, `MODEL_DIM=576`, `NUM_HEADS=8`, `NUM_KV_HEADS=4`
- RoPE base: `ROPE_BASE=100000`
- Iterations: `1200`
- Full-val pre-roundtrip: `val_bpb=1.9018` (`val_loss=3.2110`)
- Total submission size int8+zlib: `14,184,908 bytes` (under cap)

## Command

```bash
RUN_ID=aria_confirm_fullval_8x576_kv4_i1200_rope100k \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=8 MODEL_DIM=576 NUM_HEADS=8 NUM_KV_HEADS=4 \
QK_GAIN_INIT=1.5 ROPE_BASE=100000 \
ITERATIONS=1200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Key Logged Lines

- `step:1200/1200 val_loss:3.2110 val_bpb:1.9018`
- `Total submission size int8+zlib: 14184908 bytes`

## Files

- `train.log`: complete run log
- `train_gpt.py`: training script snapshot used for this run
- `submission.json`: metadata

