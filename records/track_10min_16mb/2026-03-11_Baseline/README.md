This record captures the best confirmed `train_gpt.py` run from the 8xH100 10-minute sweep on the matched `SP-1024` dataset.

Important metric note:
- This record's score is the trainer's default final `int8+zlib` roundtrip metric.
- The exact printed leaderboard value is `final_int8_zlib_roundtrip_exact val_bpb:1.24610217`.

Winning configuration:
- Dataset: matched `SP-1024` shards from `az://oaidatasets2/speedrunkits/matched_10B_docs2m_seed1337`
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Tied embedding LR: `TIED_EMBED_LR=0.05`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- Runtime cap: `MAX_WALLCLOCK_SECONDS=598`

Command (track-relevant params):
```bash
OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
DATA_PATH=/tmp/matched_10B_docs2m_seed1337/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/tmp/matched_10B_docs2m_seed1337/tokenizers/fineweb_1024_bpe.model \
RUN_ID=h100_sp1024_l9_d512_h8_kv4_m2_tie_lr005_clean \
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
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `14431/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.0985`, `val_bpb:1.2394`
- Post-quant roundtrip eval: `val_loss:2.1098`, `val_bpb:1.2461`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.24610217`
- Train time: `598039ms` (`step_avg:41.44ms`)
- Peak memory: `8926 MiB allocated`, `9136 MiB reserved`
- Serialized model int8+zlib: `15774250 bytes`
- Code size: `45097 bytes`
- Total submission size int8+zlib: `15819347 bytes`

Training volume:
- Global batch: `524288` tokens/step
- Total train tokens seen: `7566000128`

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (leaderboard metadata)