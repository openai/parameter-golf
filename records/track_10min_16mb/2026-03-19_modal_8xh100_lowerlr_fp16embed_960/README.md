This record captures `Modal 8xH100 LowerLR FP16Embed 960`.

Trainer changes in this snapshot:
- current repository `train_gpt.py` snapshot copied into the record folder
- published FineWeb export downloaded into a persistent Modal Volume and mounted into the training job
- hardware: `8xH100` on Modal using single-node `torchrun --standalone --nproc_per_node=8`
- wallclock cap: `600.0` seconds

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Tied embedding LR: `TIED_EMBED_LR=0.03`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- Dataset shards staged in Modal Volume: `80` train shards

Command (track-relevant params):
```bash
DATA_PATH=/root/parameter-golf/data/datasets/fineweb10B_sp1024 \
INT8_AXIS_MODE=auto \
ITERATIONS=20000 \
MATRIX_LR=0.02 \
MAX_WALLCLOCK_SECONDS=600.0 \
MLP_HIDDEN=960 \
NCCL_IB_DISABLE=1 \
OMP_NUM_THREADS=1 \
RUN_ID=20260319_022139_modal_8xh100_lowerlr_fp16embed_960 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
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
- Timed training stopped at `13289/20000` steps.
- Pre-quant eval at stop: `val_loss:2.0654`, `val_bpb:1.2232`
- Post-quant roundtrip eval: `val_loss:2.0666`, `val_bpb:1.2240`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.22395035`
- Train time: `599.863s`
- Peak memory: `10113 MiB allocated`, `10468 MiB reserved`
- Serialized model int8+zlib: `15792082` bytes
- Code size: `52036` bytes
- Total submission size int8+zlib: `15844118` bytes

Included files:
- `train_gpt.py` (exact code snapshot used for the run)
- `train.log` (exact training log)
- `submission.json` (leaderboard metadata)
