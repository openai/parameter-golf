This record captures a near-SOTA non-record submission built from the `MetaStack v3` trainer snapshot.

The main result is a strong under-16MB `8xH100` run that reached `1.1762` BPB on sliding-window evaluation with `seq_len=1024` and `stride=64`. It is being submitted to the non-record track rather than the main leaderboard because the best confirmation runs stopped slightly over the strict `600s` training limit (`600073ms` and `600090ms`), so this package is honest about the timing caveat instead of claiming a strict record.

Configuration:
- Track: `non-record`, still under the `16,000,000` byte artifact cap
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3`
- Tied embeddings with fp16 passthrough during export
- Extra features: `BIGRAM_ENABLED=1`, `SMEARGATE_ENABLED=1`, orthogonal init, SWA, mixed int5/int6 quantization, 2% magnitude pruning
- Batching: `TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=1024`
- Sliding eval: `EVAL_SEQ_LEN=1024 EVAL_STRIDE=64 EVAL_BATCH_SEQS=256`

Primary command (seed 42 confirmation run):
```bash
RUN_ID=v3_competitive_h100_seed42 \
SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TRAIN_SEQ_LEN=1024 \
EVAL_SLIDING=1 \
EVAL_SEQ_LEN=1024 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=256 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_BATCH_TOKENS=786432 \
NUM_LAYERS=10 \
MUON_WEIGHT_DECAY=0.04 \
SCALAR_WEIGHT_DECAY=0.01 \
TOKEN_WEIGHT_DECAY=0.01 \
GRAD_CLIP_NORM=0.3 \
SWA_ENABLED=1 \
SWA_EVERY=50 \
SWA_START_FRAC=0.5 \
BIGRAM_ENABLED=1 \
SMEARGATE_ENABLED=1 \
MIXED_QUANT_MLP_BITS=5 \
PRUNE_FRACTION=0.02 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=200 \
torchrun --standalone --nproc_per_node=8 \
  records/track_non_record_16mb/2026-03-20_MetaStack_v3_near_sota_h100/train_gpt.py
```

Best confirmed result (`train_seed42.log`):
- Timed training stopped at `7812/20000` due to the wallclock cap check
- Pre-quant eval at stop: `val_loss:1.9917`, `val_bpb:1.1796`
- Post-quant int8 roundtrip: `val_bpb:1.1819`
- Post-quant int6 roundtrip: `val_bpb:1.2110`
- Sliding-window int6 eval: `val_loss:1.9859`, `val_bpb:1.1762`
- Train time: `600073ms`
- Step average: `76.81ms`
- Total submission size int6+zstd22: `12109181 bytes`

Confirmation run (`train_seed1337.log`):
- Pre-quant eval at stop: `val_bpb:1.1806`
- Sliding-window int6 eval: `val_bpb:1.1792`
- Train time: `600090ms`
- Total submission size int6+zstd22: `12099323 bytes`

Negative result (`stride32_eval.log`):
- Same training recipe, but `EVAL_STRIDE=32`
- Sliding-window int6 eval: `val_bpb:1.1788`
- This was worse than the `stride=64` best run, so `stride=64` remains the preferred evaluation setting

Included files:
- `train_gpt.py` (exact trainer snapshot used for the H100 runs)
- `train_seed42.log` (best confirmed run)
- `train_seed1337.log` (seed-1337 confirmation)
- `stride32_eval.log` (negative ablation)
- `submission.json` (metadata for the non-record entry)
