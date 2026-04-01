This record captures a non-record 16MB submission of a stripped recurrent candidate trained on the published `fineweb10B_sp1024` data.

The model is built around a fixed four-step recurrent schedule with an explicit proposal/correction split: two passes through a proposal core followed by two passes through a corrective core, i.e. `REC_CORE_SCHEDULE=0,0,1,1`. In practice I think of this as `A-A-C-C`: first push the state forward twice with the same proposal block, then refine it twice with the same corrective block.

The important mechanism is the corrective anchor gate. The corrective core does not just continue the recurrence blindly; it can pull back toward the original token-conditioned state through grouped gates (`REC_COEFF_GROUPS=2`). That `0011 + g2` pattern was the most reliable branch in my local search, so this submission packages that path directly instead of the larger experimental trainer.

This particular run uses `MODEL_DIM=768`, `NUM_HEADS=12`, and `NUM_KV_HEADS=6`, which keeps a `64`-wide head dimension. I kept that geometry because it stays efficient on H100, while `R768` still fits comfortably under the 16MB artifact cap. A wider `R896` version scored a bit better locally but missed the byte limit, so `R768` is the cap-valid submission.

The record-local `train_gpt.py` is an exact copy of the repo-root `train_gpt_candidate.py` used for the run. The point of that file is not to present every branch I explored; it is to make the counted code match the actual candidate as closely as possible.

Configuration:
- Track: `non-record-unlimited-compute-16mb`
- Layout: `MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=6 NUM_SHARED_CORES=2`
- Recurrent structure: `REC_SOLVER=block_anchor REC_CORE_SCHEDULE=0,0,1,1 REC_COEFF_GROUPS=2`
- Corrective path: `CORRECTIVE_CORE_INDEX=1 CORRECTIVE_CONTROL_MODE=shared`
- Batching: `TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=2048`
- Run regime: `1xH100`, `MAX_WALLCLOCK_SECONDS=600`
- Data/tokenizer: `fineweb10B_sp1024` and `fineweb_1024_bpe.model`

Command (track-relevant params):
```bash
RUN_ID=gpt_candidate_r768_0011_g2_1xh100_600s \
MODEL_FAMILY=recurrent \
MODEL_DIM=768 \
NUM_HEADS=12 \
NUM_KV_HEADS=6 \
NUM_SHARED_CORES=2 \
REC_STEPS_START=4 \
REC_STEPS_MAX=4 \
REC_SOLVER=block_anchor \
REC_CORE_SCHEDULE=0,0,1,1 \
REC_COEFF_GROUPS=2 \
CORRECTIVE_CORE_INDEX=1 \
CORRECTIVE_TAIL_TYPE=none \
CORRECTIVE_CONTROL_MODE=shared \
CORRECTIVE_C2_MODE=none \
START_CORE_INDEX=0 \
ALTERNATE_START_CORE=0 \
USE_VIRTUAL_LAYERS=0 \
USE_PRECOMPOSE=1 \
SLOT_ENABLED=0 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
SEED=3407 \
TRAIN_BATCH_TOKENS=131072 \
TRAIN_SEQ_LEN=2048 \
GRAD_ACCUM_STEPS=1 \
MAX_WALLCLOCK_SECONDS=600 \
WARMUP_STEPS=5 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
VAL_TOKENS_LIMIT=65537 \
TRAIN_LOG_EVERY=100 \
python train_gpt_candidate.py
```

Key metrics (from `train.log`):
- Timed training stopped at `7432/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.2774`, `val_bpb:1.3465`
- Post-quant roundtrip eval: `val_loss:2.2929`, `val_bpb:1.3557`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.35568780`
- Train time: `600151ms` (`step_avg:80.75ms`)
- Peak memory: `17417 MiB allocated`, `17572 MiB reserved`
- Model params: `13767194`
- Serialized model int8+zlib: `12569851 bytes`
- Code size: `48881 bytes`
- Total submission size int8+zlib: `12618732 bytes`

Training volume:
- Global batch: `131072` tokens/step
- Total train tokens seen: `974127104`

Included files:
- `train_gpt.py` (exact code snapshot used for the run)
- `train.log` (exact training log)
- `submission.json` (submission metadata)
