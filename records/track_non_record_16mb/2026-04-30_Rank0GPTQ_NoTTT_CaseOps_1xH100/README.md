# Rank-0 GPTQ Serialization + No-TTT CaseOps Ablation

This is a non-record submission. It is not intended to challenge the 10-minute leaderboard. The goal is to capture a small systems ablation on the PR #1855 CaseOps/LQER/SparseAttnGate stack:

- disable validation-time TTT (`TTT_ENABLED=0`)
- keep full validation and full artifact accounting
- serialize/GPTQ only on rank 0 when running distributed (`POSTTRAIN_SINGLE_RANK=1`)

The rank-0 GPTQ change is motivated by Modal debugging: the PR #1855 script does the same CPU-side GPTQ serialization work on every torchrun rank. This variant keeps the scored model semantics unchanged for the no-TTT path, but avoids duplicate GPTQ work when `POSTTRAIN_SINGLE_RANK=1`.

## Result

Single Modal run, seed 42:

- Hardware: Modal GCP `us-east4`, one torchrun worker process (`WORLD_SIZE=1`)
- Train cap: `MAX_WALLCLOCK_SECONDS=600`
- Training stopped at step `652`
- Pre-quant post-EMA: `val_bpb=1.28347116`
- Post-roundtrip quantized: `val_bpb=1.28799323`
- Total submission size: `15,929,139` bytes
- Model artifact: `15,888,944` bytes
- Code compressed size: `40,195` bytes
- Full validation: `VAL_DOC_FRACTION=1.0`
- PPM disabled: `PPM_ENABLED=0`, `PPM_NATIVE_ENABLED=0`
- Exit code: `0`

The score is intentionally much worse than PR #1855 because this run disables phased TTT and only uses one worker process for training. It is included as a clean negative/control result and a reproducible systems hook for rank-0 GPTQ serialization.

## Reproduction

Install the normal Parameter Golf environment plus PR #1855 dependencies:

```bash
pip install -r requirements.txt
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
apt-get update && apt-get install -y lrzip
```

Prepare the CaseOps data:

```bash
MATCHED_FINEWEB_REPO_ID=romeerp/parameter-golf-caseops-v1 \
MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=datasets \
python3 data/cached_challenge_fineweb.py \
  --variant sp8192_lossless_caps_caseops_v1_reserved \
  --train-shards 80
```

Run:

```bash
RUN_ID=rank0_gptq_nottt_caseops_1xh100_seed42 \
SEED=42 \
CASEOPS_ENABLED=1 \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
TTT_ENABLED=0 \
POSTTRAIN_SINGLE_RANK=1 \
EMBED_BITS=7 \
MATRIX_LR=0.026 \
MIN_LR=0.1 \
MLP_CLIP_SIGMAS=11.5 \
ATTN_CLIP_SIGMAS=13.0 \
EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 \
WARMUP_STEPS=20 \
MUON_BACKEND_STEPS=5 \
GLOBAL_TTT_MOMENTUM=0.9 \
WARMDOWN_FRAC=0.85 \
BETA2=0.99 \
TTT_BETA2=0.99 \
TTT_WEIGHT_DECAY=0.5 \
TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_SCALE=0.5 \
GPTQ_RESERVE_SECONDS=0.5 \
GPTQ_CALIBRATION_BATCHES=16 \
VAL_LOSS_EVERY=0 \
GATED_ATTN_QUANT_GATE=1 \
SPARSE_ATTN_GATE_ENABLED=1 \
GATE_WINDOW=12 \
SMEAR_GATE_ENABLED=1 \
LQER_ENABLED=1 \
LQER_ASYM_ENABLED=1 \
LQER_RANK=4 \
LQER_FACTOR_BITS=4 \
LQER_ASYM_GROUP=64 \
LQER_TOP_K=3 \
FUSED_CE_ENABLED=1 \
COMPRESSOR=pergroup \
NCCL_NET=Socket \
PPM_ENABLED=0 \
PPM_NATIVE_ENABLED=0 \
VAL_DOC_FRACTION=1.0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Included Files

- `train_gpt.py`: runnable code snapshot
- `train_seed42_1xh100_nottt.log`: full Modal run log
- `lossless_caps.py`, `prepare_caseops_data.py`, `tokenizers/...model`: CaseOps support files copied from the PR #1855 stack
- `requirements.txt`: Python and system dependency notes
