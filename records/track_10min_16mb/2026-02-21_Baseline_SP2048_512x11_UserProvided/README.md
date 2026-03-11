This record packages the user-provided baseline run output excerpt (the exact lines shared in chat), not a full remote training log file.

Baseline configuration (from the provided command):
- `TOKENIZER_KIND=sp`
- `VOCAB_SIZE=2048`
- `NUM_LAYERS=11`
- `MODEL_DIM=512`
- `NUM_HEADS=8`
- `ITERATIONS=10000`

Key metrics (user-provided excerpt):
- End-of-training fp eval: `val_loss:2.2925`, `val_bpb:1.1427`
- `train_time: 602514ms`
- Post-quant roundtrip eval: `val_loss:2.3184`, `val_bpb:1.1556`
- Submission size int8+zlib total: `29355689 bytes`

Included log:
- `train.log` (user-provided excerpt)
