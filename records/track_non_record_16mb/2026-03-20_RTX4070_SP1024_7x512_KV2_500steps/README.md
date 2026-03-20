This record captures a local NVIDIA consumer-GPU non-record submission built from the current root `train_gpt.py`.

This run is **not** intended to satisfy the official 10-minute / 8xH100 leaderboard requirements. The goal is to provide a reproducible local-workstation experiment showing a compact 16MB-safe artifact on a single RTX 4070 while exploring a shallower + KV-thin configuration discovered through local search.

Configuration:
- Track: `non-record`, local consumer GPU, still under the `16,000,000` byte artifact cap
- Hardware: `1x NVIDIA GeForce RTX 4070 Laptop GPU (8 GiB)`
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=7 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=2 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Tied embedding LR: `TIED_EMBED_LR=0.05`
- Matrix LR: `MATRIX_LR=0.05`
- Batching: `TRAIN_BATCH_TOKENS=81920 TRAIN_SEQ_LEN=1024`
- Validation: full published `fineweb_val_*` split
- Training data: first published `fineweb_train_000000.bin` shard only

Command (track-relevant params):
```bash
RUN_ID=local-9fbd60b89a28 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024_local_proxy_62021632 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_HEADS=8 \
TIE_EMBEDDINGS=1 \
NUM_LAYERS=7 \
MODEL_DIM=512 \
NUM_KV_HEADS=2 \
MLP_MULT=2 \
TIED_EMBED_LR=0.05 \
MATRIX_LR=0.05 \
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
- Final pre-quant eval at step 500: `val_loss:2.8121`, `val_bpb:1.6655`
- Post-quant roundtrip eval: `val_loss:2.8129`, `val_bpb:1.6660`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.66595795`
- Train time: `243522ms` (`step_avg:487.04ms`)
- Eval time: `97655ms`
- Peak memory: `1437 MiB allocated`, `1468 MiB reserved`
- Serialized model int8+zlib: `10896858 bytes`
- Code size: `47642 bytes`
- Total submission size int8+zlib: `10944500 bytes`

Why this is interesting:
- It shows the current trainer can be made to run reproducibly on a modest local NVIDIA GPU with a compact artifact.
- It explores a shallower + KV-thin architecture (`7x512`, `KV2`) that outperformed several nearby local candidates in the same experiment loop.
- It acts as a public non-record anchor for a local-search-driven workflow before spending larger cloud budgets.

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact local training log)
- `submission.json` (metadata)
