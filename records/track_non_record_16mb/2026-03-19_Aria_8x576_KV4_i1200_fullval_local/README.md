# Aria Local Non-Record: 8x576 KV4 i1200 (full val)

This non-record submission captures a local full-validation run using the repository `train_gpt.py` with a wider/shallower model than baseline.

## Summary

- Track: `non-record-unlimited-compute-16mb`
- Dataset: `fineweb10B_sp1024` (full fixed validation split)
- Model shape: `NUM_LAYERS=8`, `MODEL_DIM=576`, `NUM_HEADS=8`, `NUM_KV_HEADS=4`
- Iterations: `1200`
- Final metric (int8+zlib roundtrip exact): `val_bpb=1.90535128`
- Submission size (int8+zlib + code): `14,220,659 bytes` (under 16,000,000 cap)

## Command

```bash
RUN_ID=aria_confirm_fullval_8x576_kv4_i1200 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=8 MODEL_DIM=576 NUM_HEADS=8 NUM_KV_HEADS=4 \
ITERATIONS=1200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Key Results

- Pre-quant full-val: `val_loss=3.2164`, `val_bpb=1.9049`
- Roundtrip full-val exact: `val_loss=3.21710583`, `val_bpb=1.90535128`
- Serialized model int8+zlib: `14,172,973 bytes`
- Code size: `47,686 bytes`
- Total submission size int8+zlib: `14,220,659 bytes`

## Files

- `train.log`: complete run log
- `train_gpt.py`: training script snapshot used for this run
- `submission.json`: metadata

