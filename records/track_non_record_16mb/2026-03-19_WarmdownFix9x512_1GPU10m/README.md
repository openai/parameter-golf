This non-record submission targets a stronger 1xH100 baseline by changing learning-rate schedule behavior while preserving the baseline architecture.

Compute note:
- This run uses `1x H100` due to limited compute budget while waiting for grant-backed `8xH100` record attempts.

Idea:
- Keep architecture and data pipeline fixed to baseline (`9x512`, `sp1024`).
- Change scheduler dynamics with `WARMDOWN_ITERS=100` so LR does not decay almost the entire run on slower 1-GPU step times.

Configuration:
- Hardware: `1x H100 80GB`
- Wallclock cap: `600s`
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Scheduler change: `WARMDOWN_ITERS=100` (default baseline is larger)

Command used:
```bash
RUN_ID=exp2_record_chase_warmdown100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
WARMDOWN_ITERS=100 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Results:
- Stop point: `step 1089` at `600.500s`
- Pre-quant stop eval: `val_loss=2.2755`, `val_bpb=1.3477`
- Post-quant roundtrip exact: `val_loss=2.27710001`, `val_bpb=1.34862689`
- Serialized model int8+zlib: `14,651,172 bytes`
- Code size: `47,686 bytes`
- Total submission size int8+zlib: `14,698,858 bytes`

Comparison (same session, same hardware):
- Previous 1xH100 baseline result: `val_bpb=1.34993042`
- This run: `val_bpb=1.34862689`
- Improvement: `0.00130353`

Included files:
- `train_gpt.py` (exact script snapshot used)
- `train.log` (full training/eval output)
- `submission.json` (metadata)
