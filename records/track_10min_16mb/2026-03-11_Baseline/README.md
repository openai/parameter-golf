This record captures the current `track_10min_16mb` submission from `codex-speedprobe-pg` using the mirrored `SP-1024` dataset.

Important metric note:
- This record's score is the trainer's default final `int8+zlib` roundtrip metric.
- The exact printed leaderboard value is `final_int8_zlib_roundtrip_exact val_bpb:1.24998665`.

Winning configuration:
- Pod: `codex-speedprobe-pg-0`
- Dataset: mirrored `SP-1024` shards from `az://oaidatasets2/speedrunkits/parametergolf_fineweb`
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Tied embedding LR: `TIED_EMBED_LR=0.05`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- Runtime cap: `MAX_WALLCLOCK_SECONDS=598`

Command (track-relevant params):
```bash
OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
DATA_PATH=/tmp/parametergolf_fineweb/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/tmp/parametergolf_fineweb/tokenizers/fineweb_1024_bpe.model \
RUN_ID=train_gpt_codex_speedprobe_pg_sp1024_20260311_1718 \
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
- Timed training stopped at `14141/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.1017`, `val_bpb:1.2413`
- Post-quant roundtrip eval: `val_loss:2.1164`, `val_bpb:1.2500`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.24998665`
- Train time: `598028ms` (`step_avg:42.29ms`)
- Peak memory: `8926 MiB allocated`, `8998 MiB reserved`
- Serialized model int8+zlib: `15765064 bytes`
- Code size: `46138 bytes`
- Total submission size int8+zlib: `15811202 bytes`

Training volume:
- Global batch: `524288` tokens/step
- Total train tokens seen: `7413956608`

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (leaderboard metadata)
