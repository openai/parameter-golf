This record captures an unlimited-compute non-record submission for the hybrid sparse diffusion line.

This run is not intended to satisfy the 10-minute cutoff for the main leaderboard. It keeps the current `v7` hybrid sparse diffusion architecture, runs on `8xH100`, and reports the raw full-validation result from the saved cloud run. The run reached `100000` steps with the curve still improving; it could have been pushed further by increasing `ITERATIONS`, but the available cloud budget was exhausted.

Configuration:
- Track: `non-record`, unlimited compute, still under the `16,000,000` byte artifact cap
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Optimizer LRs: `TIED_EMBED_LR=0.05 MATRIX_LR=0.04 SCALAR_LR=0.04`
- Diffusion settings: `DIFFUSION_NUM_STEPS=8 DIFFUSION_BLOCK_MIN=24 DIFFUSION_BLOCK_MAX=128`
- Diffusion masking: `DIFFUSION_MIN_MASK_FRAC=0.10 DIFFUSION_MAX_MASK_FRAC=0.60`
- Diffusion block starts: `DIFFUSION_BLOCK_START_MIN_FRAC=0.25 DIFFUSION_BLOCK_START_MAX_FRAC=0.90`
- Diffusion control: `DIFFUSION_TIME_SCALE=0.05 DIFFUSION_REFINE_LAST_N=5 DIFFUSION_BATCH_SHARED_BLOCK=1`
- Batching: `TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=1024 VAL_BATCH_SIZE=262144`
- Validation cadence: `VAL_LOSS_EVERY=0` with the full `fineweb_val_*` split at the end
- Runtime controls: `ITERATIONS=100000 WARMDOWN_ITERS=12000 WARMUP_STEPS=20 MAX_WALLCLOCK_SECONDS=7195 USE_COMPILE=0`

Command (track-relevant params):
```bash
OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
RUN_ID=challenge_pure_text_diffusion_v7_hybrid_sparse_2h_8xh100 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
NUM_LAYERS=9 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TIE_EMBEDDINGS=1 \
ITERATIONS=100000 \
WARMDOWN_ITERS=12000 \
WARMUP_STEPS=20 \
MAX_WALLCLOCK_SECONDS=7195 \
TRAIN_BATCH_TOKENS=131072 \
TRAIN_SEQ_LEN=1024 \
VAL_BATCH_SIZE=262144 \
VAL_TOKEN_LIMIT=0 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=500 \
MATRIX_LR=0.04 \
SCALAR_LR=0.04 \
TIED_EMBED_LR=0.05 \
DIFFUSION_NUM_STEPS=8 \
DIFFUSION_BLOCK_MIN=24 \
DIFFUSION_BLOCK_MAX=128 \
DIFFUSION_MIN_MASK_FRAC=0.10 \
DIFFUSION_MAX_MASK_FRAC=0.60 \
DIFFUSION_BLOCK_START_MIN_FRAC=0.25 \
DIFFUSION_BLOCK_START_MAX_FRAC=0.90 \
DIFFUSION_TIME_SCALE=0.05 \
DIFFUSION_REFINE_LAST_N=5 \
DIFFUSION_BATCH_SHARED_BLOCK=1 \
USE_COMPILE=0 \
torchrun --standalone --nproc_per_node=8 \
/root/code/parameter-golf/records/track_non_record_16mb/2026-03-31_HybridSparseDiffusion_2h_8xH100/train_gpt.py
```

Key metrics (from `train.log`):
- Raw full-val at stop: `val_loss:2.7001`, `val_bpb:1.5992`
- Training stopped at `100000/100000` steps
- Train time: `4628742ms` (`step_avg:46.29ms`)
- Peak memory: `2942 MiB allocated`, `3462 MiB reserved`
- Serialized model int8+zlib: `13279699 bytes`
- Packaged code size: `61092 bytes`
- Total submission size int8+zlib: `13340791 bytes`

Roundtrip note:
- The cloud run completed the raw full-val pass and artifact serialization, but the full exact roundtrip re-eval over all `62,021,632` validation tokens was not finished because the budget ran out.
- A local proxy roundtrip on the saved `final_model.int8.ptz` over `1,048,576` validation tokens scored `val_loss:2.70768559`, `val_bpb:1.62241719`.
- On the local RTX 4090 this proxy took `1214.67s`, so a full exact local roundtrip would be on the order of `20` hours and was not practical.

Training volume:
- Global batch: `131072` tokens/step
- Total train tokens seen: `13107200000`

Log note:
- `train.log` is an untouched raw copy of the cloud log. It contains an earlier compile-enabled launch that failed immediately, followed by the successful `USE_COMPILE=0` run reported above.

Included files:
- `train_gpt.py` (packaged code snapshot with the successful cloud defaults baked in)
- `train.log` (exact raw cloud log, renamed only)
- `submission.json` (metadata for this non-record package)
