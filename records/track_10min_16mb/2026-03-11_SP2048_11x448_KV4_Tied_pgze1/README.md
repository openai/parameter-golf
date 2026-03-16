This record captures the requested `SP-2048` config rerun on a freshly created `pgze1` pool so the training run stayed isolated from other work on shared boxes.

Important metric note:
- This run's score is the trainer's default final `int8+zlib` roundtrip metric.
- The exact printed leaderboard value is `final_int8_zlib_roundtrip_exact val_bpb:1.21660255`.

Run environment:
- Pool: `pgze1`
- Pod: `pgze1-0`
- Cluster: `zebra`
- Quota/Priority: `flex` / `low`
- Hardware: `8x H100`

Requested configuration:
- Dataset: mirrored `SP-2048` shards from `az://oaidatasets2/speedrunkits/parametergolf_fineweb`, staged to `/tmp/parametergolf_fineweb`
- Layout: `VOCAB_SIZE=2048 NUM_LAYERS=11 MODEL_DIM=448 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Tied embedding LR/init: `TIED_EMBED_LR=0.05 TIED_EMBED_INIT_STD=0.005`
- Runtime: `WARMUP_STEPS=10 MAX_WALLCLOCK_SECONDS=600 VAL_TOKENS=524288 VAL_BATCH_SIZE=65536 VAL_LOSS_EVERY=20000`

Explicit trainer defaults used for reproducibility:
- `TRAIN_BATCH_TOKENS=524288`
- `TRAIN_SEQ_LEN=1024`
- `ITERATIONS=20000`
- `TRAIN_LOG_EVERY=200`

Command (track-relevant params):
```bash
OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
RUN_ID=train_gpt_pgze1_sp2048_11x448_kv4_20260311_2243 \
DATA_PATH=/tmp/parametergolf_fineweb/datasets/fineweb10B_sp2048 \
TOKENIZER_PATH=/tmp/parametergolf_fineweb/tokenizers/fineweb_2048_bpe.model \
VOCAB_SIZE=2048 \
NUM_LAYERS=11 \
MODEL_DIM=448 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TIE_EMBEDDINGS=1 \
TIED_EMBED_LR=0.05 \
TIED_EMBED_INIT_STD=0.005 \
ITERATIONS=20000 \
WARMUP_STEPS=10 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=1024 \
TRAIN_LOG_EVERY=200 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_TOKENS=524288 \
VAL_BATCH_SIZE=65536 \
VAL_LOSS_EVERY=20000 \
torchrun --standalone --nproc_per_node=8 /root/code/parameter-golf/train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `11195/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.4614`, `val_bpb:1.2094`
- Post-quant roundtrip eval: `val_loss:2.4761`, `val_bpb:1.2166`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.21660255`
- Train time: `600069ms` (`step_avg:53.60ms`)
- Peak memory: `9491 MiB allocated`, `9720 MiB reserved`
- Serialized model int8+zlib: `15190311 bytes`
- Code size: `46189 bytes`
- Total submission size int8+zlib: `15236500 bytes`

Runtime note:
- This run stayed on a clean single-tenant 8xH100 pool for the full duration.
- Throughput stayed essentially flat from step `200` through step `11000`, finishing at `53.60ms/step`.

Training volume:
- Global batch: `524288` tokens/step
- Total train tokens seen: `5869404160`

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (metadata for this saved run)
