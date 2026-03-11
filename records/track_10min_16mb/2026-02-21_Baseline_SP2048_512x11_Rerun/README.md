This is a reproducible rerun of the SP-2048 `11x512` baseline command on `8x H100` using the saved `train_gpt.py` snapshot in this folder.

Command (same as user baseline, plus `RUN_ID` for log naming):
```bash
RUN_ID=baseline_sp2048_11x512_rerun_1771696500 \
TOKENIZER_KIND=sp \
TOKENIZER_PATH=data/matched_10B/tokenizers/fineweb_2048_bpe.model \
ENABLE_VAL_BPB=1 \
DATA_PATH=data/matched_10B/datasets/fineweb10B_sp2048 \
VOCAB_SIZE=2048 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
ITERATIONS=10000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key metrics (from `train.log`):
- End-of-training fp eval: `val_loss:2.3084`, `val_bpb:1.1506`
- `train_time:645269ms`
- Post-quant roundtrip eval: `val_loss:2.3712`, `val_bpb:1.1819`
- Submission size int8+zlib total: `27656378 bytes`

Note: this reproducible rerun is slower than the earlier user-reported baseline and exceeds the 10-minute `train_time` cutoff on this box (`645269ms > 600000ms`).

Files:
- `train_gpt.py` (exact training code snapshot used for rerun)
- `train.log` (full training log)
