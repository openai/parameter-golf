This record captures the improved top-level `train_gpt.py` run on `pgut3` using the mirrored `SP-1024` dataset from Azure.

Important metric note:
- This run's score is the trainer's default final `int8+zlib` roundtrip metric.
- The exact printed leaderboard value is `final_int8_zlib_roundtrip_exact val_bpb:1.21687994`.
- This run cleanly beats the saved `SP-1024` baseline and lands just behind the current `SP-2048` leader.

Run environment:
- Pool: `pgut3`
- Pod: `pgut3-0`
- Cluster: `scandium`
- Hardware: `8x H100`
- Dataset: mirrored `SP-1024` shards from `az://oaidatasets2/speedrunkits/parametergolf_fineweb`, staged to `/tmp/parametergolf_fineweb` with `azcopy`

Trainer changes in this snapshot:
- wallclock-aware warmdown with `WARMDOWN_ITERS=1200`
- per-channel attention and MLP branch scales
- vector skip weights and vector residual mixing
- per-head `q_gain` with `QK_GAIN_INIT=1.5`
- `fp32` preservation for tiny control tensors during export
- `INT8_CLIP_PERCENTILE=99.99984`

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Tied embedding LR: `TIED_EMBED_LR=0.05`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- Runtime cap: `MAX_WALLCLOCK_SECONDS=598`

Command (track-relevant params):
```bash
OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
RUN_ID=train_gpt_pgut3_sp1024_full_20260317_1653 \
DATA_PATH=/tmp/parametergolf_fineweb/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/tmp/parametergolf_fineweb/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TIE_EMBEDDINGS=1 \
TIED_EMBED_LR=0.05 \
ITERATIONS=20000 \
WARMUP_STEPS=20 \
MAX_WALLCLOCK_SECONDS=598 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=1024 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=0 \
VAL_TOKENS=10485760 \
VAL_BATCH_SIZE=524288 \
torchrun --standalone --nproc_per_node=8 /root/code/parameter-golf/train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `13365/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.0490`, `val_bpb:1.2102`
- Post-quant roundtrip eval: `val_loss:2.0604`, `val_bpb:1.2169`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.21687994`
- Train time: `598047ms` (`step_avg:44.75ms`)
- Peak memory: `10472 MiB allocated`, `10728 MiB reserved`
- Serialized model int8+zlib: `15814928 bytes`
- Code size: `48140 bytes`
- Total submission size int8+zlib: `15863068 bytes`

Training volume:
- Global batch: `524288` tokens/step
- Total train tokens seen: `7007109120`

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (leaderboard metadata)
