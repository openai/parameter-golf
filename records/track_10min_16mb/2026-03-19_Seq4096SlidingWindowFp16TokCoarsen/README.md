This record captures the `seq_len=4096` sliding-window submission built on the same training recipe as the non-sliding long-context run, with scoring switched to stride-64 sliding evaluation after the standard exact roundtrip check.

Trainer and export changes in this snapshot:
- `TRAIN_SEQ_LEN=4096` with `TRAIN_BATCH_TOKENS=393216`
- `TIED_EMBED_LR=0.03`, `MATRIX_LR=0.02`, `SCALAR_LR=0.02`
- `MUON_MOMENTUM=0.99`, `MUON_MOMENTUM_WARMUP_START=0.92`, `MUON_MOMENTUM_WARMUP_STEPS=1500`
- `WARMDOWN_ITERS=3000`
- `INT8_KEEP_FLOAT_NAME_PATTERNS=tok_emb.weight`
- `INT8_COARSEN_OVERRIDES=blocks.5.:2`
- `EVAL_STRIDE=64`, `SW_EVAL_BATCH=32`

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Training context: `TRAIN_SEQ_LEN=4096`
- Batching: `TRAIN_BATCH_TOKENS=393216`
- Export budget control: keep `tok_emb.weight` in fp16 and coarsen only `blocks.5.`
- Eval strategy: slide a `4096`-token window by `64` tokens and score only the final `64` tokens per window

Command (track-relevant params):
```bash
RUN_ID=qualify_pr52_fp16tok_b5s2_sw64_seed1337 \
DATA_PATH=/root/code/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
WARMDOWN_ITERS=3000 \
TRAIN_LOG_EVERY=100 \
VAL_LOSS_EVERY=0 \
TRAIN_BATCH_TOKENS=393216 \
TRAIN_SEQ_LEN=4096 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TIE_EMBEDDINGS=1 \
TIED_EMBED_LR=0.03 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
TOK_EMB_QAT_MODE=off \
INT8_KEEP_FLOAT_NAME_PATTERNS=tok_emb.weight \
INT8_COARSEN_OVERRIDES=blocks.5.:2 \
EVAL_STRIDE=64 \
SW_EVAL_BATCH=32 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key metrics from the canonical run (`train.log`):
- Timed training stopped at `11734/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.0058`, `val_bpb:1.1880`
- Post-quant roundtrip eval: `val_loss:2.0071`, `val_bpb:1.1887`
- Sliding-window exact eval: `val_loss:1.98690144`, `val_bpb:1.17675682`
- Train time: `600047ms` (`step_avg:51.14ms`)
- Standard exact eval time: `2090ms`
- Sliding eval time: `273609ms`
- Peak memory: `7701 MiB allocated`, `8208 MiB reserved`
- Serialized model int8+zlib: `15882659 bytes`
- Code size: `60601 bytes`
- Total submission size int8+zlib: `15943260 bytes`

Reproducibility reruns:
- `seed=1338`: `val_bpb:1.17675183`, `bytes_total:15949486`, `step:11728`, `train_time:600033ms`
- `seed=1339`: `val_bpb:1.17910456`, `bytes_total:15950789`, `step:11727`, `train_time:600054ms`

Included files:
- `train_gpt.py` (exact sliding-window trainer snapshot used for the run)
- `train.log` (exact remote training log from the canonical run)
- `submission.json` (leaderboard metadata)
