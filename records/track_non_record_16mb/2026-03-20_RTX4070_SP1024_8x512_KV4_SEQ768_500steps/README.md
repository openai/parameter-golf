This record captures a local NVIDIA consumer-GPU non-record submission built from the current root `train_gpt.py`.

This run is **not** intended to satisfy the official 10-minute / 8xH100 leaderboard requirements. It documents a local 1x RTX 4070 run of an 8-layer 512-dim configuration discovered by a throughput-oriented search path and then re-evaluated against the full published validation split.

Configuration:
- Track: `non-record`, local consumer GPU, still under the `16,000,000` byte artifact cap
- Hardware: `1x NVIDIA GeForce RTX 4070 Laptop GPU (8 GiB)`
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=8 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Tied embedding LR: `TIED_EMBED_LR=0.07`
- Matrix LR: `MATRIX_LR=0.05`
- Batching: `TRAIN_BATCH_TOKENS=98304 TRAIN_SEQ_LEN=768`
- Validation: full published `fineweb_val_*` split
- Training data: first published `fineweb_train_000000.bin` shard only

Command (track-relevant params):
```bash
RUN_ID=local-de7915d3d64c \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024_local_proxy_62021632 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_HEADS=8 \
TIE_EMBEDDINGS=1 \
NUM_LAYERS=8 \
MODEL_DIM=512 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TIED_EMBED_LR=0.07 \
MATRIX_LR=0.05 \
TRAIN_SEQ_LEN=768 \
TRAIN_BATCH_TOKENS=98304 \
VAL_BATCH_SIZE=98304 \
ITERATIONS=500 \
WARMUP_STEPS=4 \
TRAIN_LOG_EVERY=10 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=900 \
OMP_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCHINDUCTOR_COMPILE_THREADS=1 \
NCCL_IB_DISABLE=1 \
CC=gcc \
CXX=g++ \
python train_gpt.py
```

Key metrics (from `train.log`):
- Final pre-quant eval at step 500: `val_loss:2.7165`, `val_bpb:1.6089`
- Post-quant roundtrip eval: `val_loss:2.7208`, `val_bpb:1.6114`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.61140740`
- Train time: `691552ms` (`step_avg:1383.10ms`)
- Eval time: `170439ms`
- Peak memory: `2005 MiB allocated`, `2036 MiB reserved`
- Serialized model int8+zlib: `9988629 bytes`
- Code size: `47642 bytes`
- Total submission size int8+zlib: `10036271 bytes`

Why this is interesting:
- It is currently the strongest local full-validation result in this experiment loop.
- It shows that a throughput-oriented, shorter-sequence training path can beat nearby local alternatives while remaining comfortably below the 16MB artifact cap.
- It provides a public non-record anchor for a candidate family selected through repeated local search and validation rather than one-off tuning.

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact local training log)
- `submission.json` (metadata)
