This non-record snapshot captures the strongest `1xH100` proxy run we obtained for a cognitively motivated CTM recipe.

It is not intended as a main-leaderboard claim. The run used the published `fineweb10B_sp1024` export and the standard exact int8+zlib evaluation path, but only `1xH100`, `10` training shards, and a `20 minute` wallclock cap. The purpose of this folder is to package the current best proxy recipe as a reproducible standalone record for review and follow-up.

Approach:
- Start from the public `9x512` GPT baseline with tied embeddings and `4` KV heads.
- Add a small causal CTM workspace bridge with `4` slots x `64` dims.
- Route workspace writes with `novelty + salience` (`CTM_NOVELTY_GAIN=1.0`, `CTM_SALIENCE_GAIN=0.5`).
- Add prediction-error-gated skip connections with `SKIP_GATE_MODE=error`.
- Add export-matched tail QAT (`QAT_MODE=export_int8`, `QAT_START_FRAC=0.70`) so the model trains against the same large-matrix int8 clipping/dequantization used by the final artifact path.

Configuration:
- Track: `non-record`, proxy run, not an `8xH100 / 10 minute` leaderboard submission
- Data: published `fineweb10B_sp1024` shards and published SP-1024 tokenizer
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- CTM: `CTM_SLOTS=4 CTM_DIM=64 CTM_NOVELTY_GAIN=1.0 CTM_SALIENCE_GAIN=0.5 CTM_SHARE_READ=0`
- Extra routing: `SKIP_GATE_MODE=error`
- Quantization-aware training: `QAT_MODE=export_int8 QAT_START_FRAC=0.70`
- Batching: `TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=1024`

Command (track-relevant params):
```bash
RUN_ID=ctm_causal_salience_error_skip_tailqat_sp1024_1xh100 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
CTM_SLOTS=4 \
CTM_DIM=64 \
CTM_NOVELTY_GAIN=1.0 \
CTM_SALIENCE_GAIN=0.5 \
CTM_SHARE_READ=0 \
SKIP_GATE_MODE=error \
QAT_MODE=export_int8 \
QAT_START_FRAC=0.70 \
TRAIN_BATCH_TOKENS=131072 \
TRAIN_SEQ_LEN=1024 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=1 /workspace/parameter-golf/train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `6759/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.1809`, `val_bpb:1.2917`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.29167890`
- Train time: `1200062ms` (`step_avg:177.55ms`)
- Peak memory: `3508 MiB allocated`, `4614 MiB reserved`
- Serialized model int8+zlib: `15941743 bytes`
- Code size at the time of the logged run: `67788 bytes`
- Total submission size int8+zlib at the time of the logged run: `16009531 bytes`

Important note:
- This logged proxy run missed the decimal `16,000,000` byte artifact cap by `9531` bytes, so it is not a valid record submission as-is.
- After this run, the root `train_gpt.py` was code-trimmed without changing the active CTM + error-skip + tail-QAT path, reducing the current script size to `49644` bytes and bringing the projected total below the cap. That exact rerun is not included in this folder, so this record should be read as an honest in-progress non-record snapshot rather than a leaderboard claim.

Included files:
- `train_gpt.py` (standalone code snapshot used for the logged proxy run)
- `train.log` (exact `1xH100` proxy training log)
- `submission.json` (metadata for this non-record snapshot)
