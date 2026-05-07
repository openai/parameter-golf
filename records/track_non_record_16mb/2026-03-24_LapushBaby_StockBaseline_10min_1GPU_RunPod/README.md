# Non-record: Stock `train_gpt.py` — 1×GPU (RunPod), 600s wallclock

**Track:** `non-record-16mb`  
**Author:** LapushBaby (`@LapushBaby`)

## Summary

This submission documents a **reproducible** training run using the **unmodified** repository root `train_gpt.py` (copied into this folder per submission requirements), on a **single GPU** RunPod instance with a **600-second** training-time cap. It is **not** intended to compete with 8×H100 leaderboard entries; it preserves metrics and commands for transparency.

## Score (post int8 + zlib roundtrip)

- **`val_bpb`:** **1.36294332** (`final_int8_zlib_roundtrip_exact`)
- **`val_loss`:** 2.30127271
- **Total submission bytes (int8+zlib + code):** 12,264,252 (under 16,000,000)

## Hardware / constraints

- **GPUs:** 1× (RunPod; exact SKU not essential for non-record documentation)
- **Training time cap:** `MAX_WALLCLOCK_SECONDS=600`
- **Steps completed:** 967 / 20000 (stopped by wallclock, as expected)

## Data

- Variant: **sp1024**
- Training shards downloaded: **10** (`python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10`)
- Paths:
  - `DATA_PATH=./data/datasets/fineweb10B_sp1024/`
  - `TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model`

## Run command (reproduce)

From the repository root (after installing `requirements.txt` and downloading data):

```bash
RUN_ID=real_10min \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

The `train_gpt.py` in **this folder** is a snapshot of the root training script at the time of packaging; it should match upstream `train_gpt.py` unless noted otherwise.

## Files

- `train_gpt.py` — training script snapshot
- `train.log` — key metric lines from the run
- `submission.json` — metadata

## Notes

- Single-seed run; multi-seed sweeps were not performed (non-record).
- For leaderboard-class scores, see `/records/track_10min_16mb/` entries trained on **8×H100** with specialized `train_gpt.py` implementations.
