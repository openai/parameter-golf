This record captures a 10-minute `8xH100` Modal run submitted on the non-record track.

The run stayed under the `16,000,000` byte artifact cap, but its final int8+zlib roundtrip score of `1.22964598` did not set a new leaderboard record. This folder preserves the exact `train_gpt.py` snapshot and training log from that run.

Trainer setup for this snapshot:
- current repository `train_gpt.py` snapshot copied into the record folder
- published FineWeb `sp1024` export downloaded into a persistent Modal Volume and mounted into the training job
- hardware: `8xH100` on Modal using single-node `torchrun --standalone --nproc_per_node=8`
- wallclock cap: `600.0` seconds

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- Validation cadence: `VAL_LOSS_EVERY=200` on the published validation split
- Dataset shards staged in Modal Volume: `80` train shards

Command (track-relevant params):
```bash
DATA_PATH=/root/parameter-golf/data/datasets/fineweb10B_sp1024 \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600.0 \
NCCL_IB_DISABLE=1 \
OMP_NUM_THREADS=1 \
RUN_ID=20260319_004357_modal_8xh100_record_attempt_45d2820c \
TOKENIZER_PATH=/root/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_LOG_EVERY=50 \
TRAIN_SEQ_LEN=1024 \
VAL_BATCH_SIZE=524288 \
VAL_LOSS_EVERY=200 \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `13262/20000` steps.
- Pre-quant eval at stop: `val_loss:2.0622`, `val_bpb:1.2214`
- Post-quant roundtrip eval: `val_loss:2.0762`, `val_bpb:1.2296`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.22964598`
- Train time: `599.942s`
- Peak memory: `10184 MiB allocated`, `10302 MiB reserved`
- Serialized model int8+zlib: `15804251` bytes
- Code size: `49353` bytes
- Total submission size int8+zlib: `15853604` bytes

Included files:
- `train_gpt.py` (exact code snapshot used for the run)
- `train.log` (exact training log)
- `submission.json` (leaderboard metadata)
