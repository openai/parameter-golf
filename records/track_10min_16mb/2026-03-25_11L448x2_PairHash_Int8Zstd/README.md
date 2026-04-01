This record captures the current winning safe submission candidate: `11L / 448 / 2x / PairHash / int8+zstd`.

Trainer changes in this snapshot:
- modified `train_gpt.py` snapshot copied into the record folder; this is the run-specific script used on the pod and includes the PairHash + `int8+zstd` export logic for this record
- `PairHash` enabled with `8192` buckets and `96` pair dimensions
- 10-minute wallclock cap on `8xH100`
- periodic validation every `2000` steps on the full `fineweb_val_*` split
- export path fixed to `int8+zstd`

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=448 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- PairHash embeddings: `PAIRHASH_ENABLED=1 PAIRHASH_BUCKETS=8192 PAIRHASH_DIM=96`
- Batching: `TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048`
- Averaging / quantization extras: `EMA_ENABLED=0 USE_SWA=0 QAT_ENABLED=0`

Command (track-relevant params):
```bash
RUN_ID=final_8x_11l448x2_s1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 MODEL_DIM=448 MLP_MULT=2 \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=786432 \
WARMDOWN_ITERS=3500 \
PAIRHASH_ENABLED=1 PAIRHASH_BUCKETS=8192 PAIRHASH_DIM=96 \
EMA_ENABLED=0 USE_SWA=0 QAT_ENABLED=0 \
EVAL_STRIDE=0 EVAL_DOC_ISOLATED=0 \
EXPORT_MODE=int8 USE_ZSTD=1 \
SEED=1337 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=2000 \
TRAIN_LOG_EVERY=100 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key metrics (from the captured winning run):
- Timed training stopped at `6436/20000` steps due to the wallclock cap.
- Last in-training eval at stop: `val_loss:2.3028`, `val_bpb:1.3638`
- Post-export roundtrip eval: `val_loss:2.31055144`, `val_bpb:1.36843871`
- Exact printed metric: `final_export_roundtrip_exact val_bpb:1.36843871`
- Train time: `600110ms` (`step_avg:93.24ms`)
- Peak memory: `16639 MiB allocated`, `16992 MiB reserved`
- Serialized model export int8+zstd: `15085258 bytes`
- Code size: `64461 bytes`
- Total submission size export: `15149719 bytes`

Interpretation:
- this run is safely under the `16,000,000` byte submission cap
- it substantially improves on the older valid baseline path
- it is the current submission mainline; `11L / 384 / 3x` remains only as a lighter backup

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact remote training log recovered from the pod)
- `submission.json` (leaderboard metadata draft)
