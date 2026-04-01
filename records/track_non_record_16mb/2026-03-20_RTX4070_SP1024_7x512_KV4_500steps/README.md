This record captures a local NVIDIA consumer-GPU non-record submission built from the current root `train_gpt.py`.

This run is **not** intended to satisfy the official 10-minute / 8xH100 leaderboard requirements. It documents a local 1x RTX 4070 run of a shallower 7-layer 512-dim KV4 configuration discovered through the local search loop and then evaluated against the full published validation split.

Configuration:
- Track: `non-record`, local consumer GPU, still under the `16,000,000` byte artifact cap
- Hardware: `1x NVIDIA GeForce RTX 4070 Laptop GPU (8 GiB)`
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=7 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Tied embedding LR: `TIED_EMBED_LR=0.05`
- Matrix LR: `MATRIX_LR=0.04`
- Batching: `TRAIN_BATCH_TOKENS=81920 TRAIN_SEQ_LEN=1024`
- Validation: full published `fineweb_val_*` split
- Training data: first published `fineweb_train_000000.bin` shard only

Command (track-relevant params):
```bash
RUN_ID=local-d1b545246e43 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024_local_proxy_62021632 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_HEADS=8 \
TIE_EMBEDDINGS=1 \
NUM_LAYERS=7 \
MODEL_DIM=512 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TIED_EMBED_LR=0.05 \
MATRIX_LR=0.04 \
TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=81920 \
VAL_BATCH_SIZE=81920 \
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
- Final pre-quant eval at step 500: `val_loss:2.7972`, `val_bpb:1.6567`
- Post-quant roundtrip eval: `val_loss:2.7981`, `val_bpb:1.6572`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.65718316`
- Train time: `255711ms` (`step_avg:511.42ms`)
- Eval time: `112875ms`
- Peak memory: `1507 MiB allocated`, `1542 MiB reserved`
- Serialized model int8+zlib: `10249187 bytes`
- Code size: `47642 bytes`
- Total submission size int8+zlib: `10296829 bytes`

Why this is interesting:
- It tests a shallower 7-layer configuration while keeping KV4, providing a useful control against the stronger 8-layer throughput-oriented run and the 7-layer KV2 run.
- It remains safely below the 16MB artifact cap while offering a public non-record result for a distinct architecture family.

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact local training log)
- `submission.json` (metadata)
