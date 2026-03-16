This run captures the requested `SP-2048` config on `pgut1` using the mirrored FineWeb shards and the stock `train_gpt.py` trainer.

Important run note:
- The pool was idle when the run started, but another unrelated `train_gpt.py` job appeared on the same 8xH100 pod during training.
- Step time held near `57ms` through step `7000`, then degraded sharply and finished at `70.52ms/step`.
- This folder is therefore saved as a non-record reference run, not as a clean standalone benchmark submission.

Important metric note:
- This run's score is the trainer's default final `int8+zlib` roundtrip metric.
- The exact printed leaderboard value is `final_int8_zlib_roundtrip_exact val_bpb:1.22331228`.

Requested configuration:
- Pod: `pgut1-0`
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
RUN_ID=train_gpt_pgut1_sp2048_11x448_kv4_20260311_2217 \
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
- Timed training stopped at `8510/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.4784`, `val_bpb:1.2177`
- Post-quant roundtrip eval: `val_loss:2.4897`, `val_bpb:1.2233`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.22331228`
- Train time: `600113ms` (`step_avg:70.52ms`)
- Peak memory: `9491 MiB allocated`, `9720 MiB reserved`
- Serialized model int8+zlib: `15183431 bytes`
- Code size: `46189 bytes`
- Total submission size int8+zlib: `15229620 bytes`

Contention note:
- Another unrelated 8-GPU `train_gpt.py` job started on `pgut1` during this run.
- The timing inflection is visible directly in `train.log`: `step_avg` rises from `57.16ms` at step `7000` to `70.52ms` at the final timed validation.
- Because of that interference, this result should be treated as a saved reference run rather than a leaderboard claim.

Training volume:
- Global batch: `524288` tokens/step
- Total train tokens seen: `4461690880`

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (metadata for this saved run)
