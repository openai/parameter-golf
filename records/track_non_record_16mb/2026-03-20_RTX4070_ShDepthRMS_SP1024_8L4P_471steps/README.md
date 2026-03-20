This record captures a local NVIDIA consumer-GPU non-record submission using a **shared-depth RMS interface** variant of `train_gpt.py`.

This run is **not** intended to satisfy the official 10-minute / 8xH100 leaderboard requirements. It is a local workstation experiment testing whether a shared-depth model with extra pre-projection RMSNorm and tiny phase-conditioned scales can survive final compression better than a naive shared-depth baseline.

Configuration:
- Track: `non-record`, local consumer GPU, still under the `16,000,000` byte artifact cap
- Hardware: `1x NVIDIA GeForce RTX 4070 Laptop GPU (8 GiB)`
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=8 NUM_PHYSICAL_LAYERS=4 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Interface additions:
  - `EXTRA_PROJ_RMSNORM=1`
  - `PHASE_CONDITIONED_SCALES=1`
  - `PHASE_BUCKETS=4`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Tied embedding LR: `TIED_EMBED_LR=0.05`
- Matrix LR: `MATRIX_LR=0.04`
- Batching: `TRAIN_BATCH_TOKENS=73728 TRAIN_SEQ_LEN=1024`
- Validation: full published `fineweb_val_*` split
- Training data: first published `fineweb_train_000000.bin` shard only

Command (track-relevant params):
```bash
RUN_ID=shareddepth-v0-full \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024_local_proxy_62021632 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_HEADS=8 \
TIE_EMBEDDINGS=1 \
NUM_LAYERS=8 \
NUM_PHYSICAL_LAYERS=4 \
MODEL_DIM=512 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TIED_EMBED_LR=0.05 \
MATRIX_LR=0.04 \
TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=73728 \
VAL_BATCH_SIZE=73728 \
ITERATIONS=500 \
WARMUP_STEPS=4 \
TRAIN_LOG_EVERY=10 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=900 \
PHASE_BUCKETS=4 \
EXTRA_PROJ_RMSNORM=1 \
PHASE_CONDITIONED_SCALES=1 \
OMP_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCHINDUCTOR_COMPILE_THREADS=1 \
NCCL_IB_DISABLE=1 \
CC=gcc \
CXX=g++ \
python train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `471/500` steps due to the 900-second wallclock cap.
- Final pre-quant eval at stop: `val_loss:2.7975`, `val_bpb:1.6568`
- Post-quant roundtrip eval: `val_loss:2.7989`, `val_bpb:1.6577`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.65765109`
- Train time: `908129ms` (`step_avg:1928.09ms` at stop)
- Eval time: `119999ms`
- Peak memory: `1445 MiB allocated`, `1496 MiB reserved`
- Serialized model int8+zlib: `5861528 bytes`
- Code size: `50495 bytes`
- Total submission size int8+zlib: `5912023 bytes`

Why this is interesting:
- The model stores only 4 physical blocks for 8 logical passes, cutting parameter count to `7.89M` and the final artifact to ~`5.9MB`.
- It tests the specific frontier idea that shared depth can be stabilized by a better compression interface: extra pre-projection RMSNorm plus tiny phase-conditioned scales.
- It remains competitive enough locally to justify future follow-up rather than being an obviously broken moonshot.

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact local training log)
- `submission.json` (metadata)
