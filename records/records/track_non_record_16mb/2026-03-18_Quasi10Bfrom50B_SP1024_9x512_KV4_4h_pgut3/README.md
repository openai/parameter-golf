This record captures an unlimited-compute non-record submission built from the current root `train_gpt.py`.

This run is not intended to satisfy the 10-minute cutoff for the main leaderboard. It uses the same 9x512 SP-1024 tied-embedding baseline layout, but extends training to a 4-hour wallclock cap on `pgut3` while evaluating against the full 50k-document validation split every 20k steps.

Configuration:
- Track: `non-record`, unlimited compute, still under the `16,000,000` byte artifact cap
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Tied embedding LR: `TIED_EMBED_LR=0.05`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- Validation cadence: `VAL_LOSS_EVERY=20000` on the full `fineweb_val_*` split

Command (track-relevant params):
```bash
OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
RUN_ID=train_gpt_pgut3_quasi10b_sp1024_4h_20260318_075102 \
DATA_PATH=/tmp/fineweb_quasi10Bfrom50B_50keval_sp1024_v0/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/tmp/fineweb_quasi10Bfrom50B_50keval_sp1024_v0/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TIE_EMBEDDINGS=1 \
TIED_EMBED_LR=0.05 \
ITERATIONS=500000 \
WARMUP_STEPS=20 \
MAX_WALLCLOCK_SECONDS=14400 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=1024 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=20000 \
torchrun --standalone --nproc_per_node=8 /root/code/parameter-golf/train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `329430/500000` steps due to the 4-hour wallclock cap.
- Best pre-quant eval at stop: `val_loss:1.9837`, `val_bpb:1.1749`
- Post-quant roundtrip eval: `val_loss:2.0386`, `val_bpb:1.2074`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.20737944`
- Train time: `14400039ms` (`step_avg:43.71ms`)
- Peak memory: `10184 MiB allocated`, `10588 MiB reserved`
- Serialized model int8+zlib: `15762519 bytes`
- Code size: `47642 bytes`
- Total submission size int8+zlib: `15810161 bytes`

Training volume:
- Global batch: `524288` tokens/step
- Total train tokens seen: `172716195840`

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (leaderboard metadata)
