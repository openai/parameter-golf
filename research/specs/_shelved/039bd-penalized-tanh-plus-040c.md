# Spec 039bD — penalized-tanh plus 040C composite

**Slug:** `penalized-tanh-plus-040c`
**Created:** 2026-04-25
**Status:** READY
**Branch:** `exp/039bd-penalized-tanh-plus-040c`
**Commit:** `c78534a`
**Links to:** `research/ideas/039bd-penalized-tanh-plus-040c.md`, `research/specs/039b-loop-band-activation-screen.md`, `research/specs/040-loop-layer-mlp-reallocation-screen.md`

## Hypothesis

The loop band `3,4,5` wants both:

- a richer MLP allocation (`040C`)
- a more stable recurrent-band activation (`039bA`)

The composite should outperform `039bA` alone if width reallocation is truly
additive on top of the activation win.

## Baselines

Primary comparison arms:

1. baseline
2. `039bA`
3. composite `039bD`

Pinned upstream references:

- `039b` runnable branch: `exp/039b-loop-band-activation-screen`
- `040` runnable branch: `exp/040-loop-layer-mlp-reallocation-screen`

## Config diff

Keep the whole `038/039` family fixed and apply:

- width split from `040C`
  - `MLP_SCHEDULE_ENABLED=1`
  - `MLP_EARLY_MULT=4.0`
  - `MLP_MIDDLE_MULT=5.0`
  - `MLP_LATE_MULT=3.4`
  - `MLP_MIDDLE_LAYERS=3,4,5`
- activation split from `039bA`
  - `MLP_OUTER_ACTIVATION=leaky_relu_square`
  - `MLP_MIDDLE_ACTIVATION=penalized_tanh`
  - `MLP_MIDDLE_NEGATIVE_SLOPE=0.5`
- training-only stop
  - `TRAINING_ONLY_SCREEN=1`

Pinned runnable code source:

- branch: `exp/039bd-penalized-tanh-plus-040c`
- commit: `c78534a`
- script:
  [train_gpt.py](/home/claude-user/ai-workspace/projects/parameter-golf/worktrees/039bd-composite-screen/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py)

## Regime

Training-only screen:

- `4×H100`
- `SEED=42`
- `MAX_WALLCLOCK_SECONDS=600`
- `TTT_ENABLED=0`
- `TRAINING_ONLY_SCREEN=1`

Compare:

- stop-time val_bpb
- pre-quant post-EMA val_bpb
- steps reached
- train loss trajectory
- throughput

Out of scope:

- GPTQ / quantized eval
- TTT

## Run protocol

Run only these three arms:

1. `baseline`
2. `039bA`
3. `039bD`

Resolved base env block:

```bash
DATA_DIR=/workspace/parameter-golf/data
DATASETS_DIR=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
TOKENIZER_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
TRAIN_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin
VAL_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin
VAL_BYTES_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_*.bin
VOCAB_SIZE=8192
NUM_LAYERS=11
XSA_LAST_N=11
MODEL_DIM=512
NUM_KV_HEADS=4
NUM_HEADS=8
MLP_MULT=4.0
NEGATIVE_SLOPE=0.5
MLP_MIDDLE_LAYERS=3,4,5
TIE_EMBEDDINGS=1
LOGIT_SOFTCAP=30
ROPE_BASE=10000
ROPE_DIMS=16
ROPE_TRAIN_SEQ_LEN=2048
ROPE_YARN=0
LN_SCALE=1
QK_GAIN_INIT=5.0
NUM_LOOPS=2
LOOP_START=3
LOOP_END=5
ENABLE_LOOPING_AT=0.35
PARALLEL_START_LAYER=8
PARALLEL_FINAL_LANE=mean
MIN_LR=0.1
EMBED_LR=0.6
TIED_EMBED_LR=0.03
TIED_EMBED_INIT_STD=0.005
MATRIX_LR=0.026
SCALAR_LR=0.02
MUON_MOMENTUM=0.97
MUON_BACKEND_STEPS=5
MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500
MUON_ROW_NORMALIZE=1
BETA1=0.9
BETA2=0.95
ADAM_EPS=1e-8
GRAD_CLIP_NORM=0.3
ADAM_WD=0.02
MUON_WD=0.095
EMBED_WD=0.085
EMA_DECAY=0.9965
TRAIN_BATCH_TOKENS=786432
TRAIN_SEQ_LEN=2048
TRAIN_LOG_EVERY=100
ITERATIONS=20000
WARMDOWN_FRAC=0.75
WARMUP_STEPS=20
VAL_BATCH_TOKENS=524288
EVAL_SEQ_LEN=2048
EVAL_STRIDE=64
VAL_LOSS_EVERY=0
CASEOPS_ENABLED=1
COMPRESSOR=brotli
MATRIX_BITS=6
MATRIX_CLIP_SIGMAS=12.85
ATTN_CLIP_SIGMAS=13.0
MLP_CLIP_SIGMAS=12.0
EMBED_BITS=7
EMBED_CLIP_SIGMAS=15.0
GPTQ_CALIBRATION_BATCHES=16
GPTQ_RESERVE_SECONDS=0.5
SKIP_GATES_ENABLED=1
SPARSE_ATTN_GATE_ENABLED=1
SPARSE_ATTN_GATE_INIT_STD=0.0
SPARSE_ATTN_GATE_SCALE=1.0
GATED_ATTN_ENABLED=0
GATED_ATTN_INIT_STD=0.005
GATED_ATTN_QUANT_GATE=1
ATTN_OUT_GATE_ENABLED=0
ATTN_OUT_GATE_SRC=proj
GATE_WINDOW=12
RECUR_ALPHA_ENABLED=1
RECUR_DIAG_P2P_COS=0
SMEAR_GATE_ENABLED=1
LQER_ENABLED=1
LQER_RANK=4
LQER_TOP_K=3
LQER_FACTOR_BITS=4
LQER_ASYM_ENABLED=1
LQER_ASYM_GROUP=64
SPINQUANT_ENABLED=0
SPINQUANT_SEED=42
SPINQUANT_SITES=attn_in,attn_proj_in,mlp_in,mlp_proj_in
SEED=42
MAX_WALLCLOCK_SECONDS=600
TTT_ENABLED=0
TRAINING_ONLY_SCREEN=1
```

Canonical launch block:

```bash
for arm in baseline 039bA 039bD; do
  case "$arm" in
    baseline)
      MLP_SCHEDULE_ENABLED=0
      MLP_EARLY_MULT=4.0
      MLP_MIDDLE_MULT=4.0
      MLP_LATE_MULT=4.0
      MLP_OUTER_ACTIVATION=leaky_relu_square
      MLP_MIDDLE_ACTIVATION=leaky_relu_square
      MLP_MIDDLE_NEGATIVE_SLOPE=0.5
      ;;
    039bA)
      MLP_SCHEDULE_ENABLED=0
      MLP_EARLY_MULT=4.0
      MLP_MIDDLE_MULT=4.0
      MLP_LATE_MULT=4.0
      MLP_OUTER_ACTIVATION=leaky_relu_square
      MLP_MIDDLE_ACTIVATION=penalized_tanh
      MLP_MIDDLE_NEGATIVE_SLOPE=0.5
      ;;
    039bD)
      MLP_SCHEDULE_ENABLED=1
      MLP_EARLY_MULT=4.0
      MLP_MIDDLE_MULT=5.0
      MLP_LATE_MULT=3.4
      MLP_OUTER_ACTIVATION=leaky_relu_square
      MLP_MIDDLE_ACTIVATION=penalized_tanh
      MLP_MIDDLE_NEGATIVE_SLOPE=0.5
      ;;
  esac
  env \
    DATA_DIR=/workspace/parameter-golf/data \
    DATASETS_DIR=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
    TOKENIZER_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
    TRAIN_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin \
    VAL_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin \
    VAL_BYTES_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_*.bin \
    VOCAB_SIZE=8192 NUM_LAYERS=11 XSA_LAST_N=11 MODEL_DIM=512 NUM_KV_HEADS=4 NUM_HEADS=8 MLP_MULT=4.0 NEGATIVE_SLOPE=0.5 \
    MLP_MIDDLE_LAYERS=3,4,5 MLP_SCHEDULE_ENABLED="$MLP_SCHEDULE_ENABLED" MLP_EARLY_MULT="$MLP_EARLY_MULT" MLP_MIDDLE_MULT="$MLP_MIDDLE_MULT" MLP_LATE_MULT="$MLP_LATE_MULT" \
    MLP_OUTER_ACTIVATION="$MLP_OUTER_ACTIVATION" MLP_MIDDLE_ACTIVATION="$MLP_MIDDLE_ACTIVATION" MLP_MIDDLE_NEGATIVE_SLOPE="$MLP_MIDDLE_NEGATIVE_SLOPE" \
    TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30 ROPE_BASE=10000 ROPE_DIMS=16 ROPE_TRAIN_SEQ_LEN=2048 ROPE_YARN=0 LN_SCALE=1 QK_GAIN_INIT=5.0 \
    NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35 PARALLEL_START_LAYER=8 PARALLEL_FINAL_LANE=mean \
    MIN_LR=0.1 EMBED_LR=0.6 TIED_EMBED_LR=0.03 TIED_EMBED_INIT_STD=0.005 MATRIX_LR=0.026 SCALAR_LR=0.02 \
    MUON_MOMENTUM=0.97 MUON_BACKEND_STEPS=5 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_ROW_NORMALIZE=1 \
    BETA1=0.9 BETA2=0.95 ADAM_EPS=1e-8 GRAD_CLIP_NORM=0.3 ADAM_WD=0.02 MUON_WD=0.095 EMBED_WD=0.085 EMA_DECAY=0.9965 \
    TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 TRAIN_LOG_EVERY=100 ITERATIONS=20000 WARMDOWN_FRAC=0.75 WARMUP_STEPS=20 \
    VAL_BATCH_TOKENS=524288 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 VAL_LOSS_EVERY=0 \
    CASEOPS_ENABLED=1 COMPRESSOR=brotli MATRIX_BITS=6 MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=12.0 EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 GPTQ_CALIBRATION_BATCHES=16 GPTQ_RESERVE_SECONDS=0.5 \
    SKIP_GATES_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_INIT_STD=0.0 SPARSE_ATTN_GATE_SCALE=1.0 \
    GATED_ATTN_ENABLED=0 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 ATTN_OUT_GATE_ENABLED=0 ATTN_OUT_GATE_SRC=proj GATE_WINDOW=12 \
    RECUR_ALPHA_ENABLED=1 RECUR_DIAG_P2P_COS=0 SMEAR_GATE_ENABLED=1 \
    LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 \
    SPINQUANT_ENABLED=0 SPINQUANT_SEED=42 SPINQUANT_SITES=attn_in,attn_proj_in,mlp_in,mlp_proj_in \
    SEED=42 MAX_WALLCLOCK_SECONDS=600 TTT_ENABLED=0 TRAINING_ONLY_SCREEN=1 \
    RUN_ID="039bd-${arm}" \
    torchrun --standalone --nproc_per_node=4 train_gpt.py
done
```

## Acceptance

Interesting outcome:

- `039bD` is clearly better than `039bA`

Failure outcome:

- `039bD` ties or loses to `039bA`, implying the width split is not adding on
  top of the activation win
