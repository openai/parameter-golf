# Spec 039b — loop-band activation screen

**Slug:** `loop-band-activation-screen`
**Created:** 2026-04-24
**Updated:** 2026-04-25
**Status:** READY
**Branch:** `exp/039b-loop-band-activation-screen`
**Commit:** `5bbf12f`
**Links to:** `research/ideas/039b-loop-band-activation-screen.md`, `research/specs/039-neg-slope-screen-on-1797-base.md`

## Backward bug fix

Original commit `8f10d16` had a bug in the fused `LeakyReLU(s)²` Triton
backward (`2·s·x` instead of `2·s²·x` for negative-side inputs). All prior
039b results are invalid.

Critically, the contamination was asymmetric across arms:

- **baseline / 039bC**: all layers used the fused path → buggy backward everywhere
- **039bA / 039bB**: outer layers used the fused path (buggy), but middle layers
  used `penalized_tanh` / `tanh` via eager execution → correct backward in the
  middle band

This means 039bA may have "won" partly because it received correct gradients in
layers 3,4,5 while the baseline did not — an unfair advantage unrelated to the
activation choice. Re-running on `5bbf12f` makes all arms use the correct
`leaky_relu_square` backward on the outer layers, giving a fair comparison.

## Hypothesis

The recurrent middle physical layers `3,4,5` want a different MLP activation
than the outer trunk. Keeping outer layers on `LeakyReLU(0.5)^2` but changing
only the loop band may improve short-run learning signal.

## Baseline

Use the fixed `039b` code line.

Pinned current runnable base:

- branch: `exp/039b-loop-band-activation-screen`
- commit: `5bbf12f`

Uniform activation baseline:

- all `11` physical layers use `LeakyReLU(0.5)^2`

## Config diff

Keep the whole `039` stack fixed and change only the loop-band MLP activation.

Pinned implementation API for this spec:

- `MLP_OUTER_ACTIVATION=leaky_relu_square`
- `MLP_MIDDLE_ACTIVATION=<leaky_relu_square|penalized_tanh|tanh>`
- `MLP_MIDDLE_NEGATIVE_SLOPE=<float>`
- `MLP_MIDDLE_LAYERS=3,4,5`
- `TRAINING_ONLY_SCREEN=1`

Interpretation:

- outer layers are all physical layers not in `MLP_MIDDLE_LAYERS`
- middle layers are exactly `MLP_MIDDLE_LAYERS`
- `NEGATIVE_SLOPE=0.5` remains the outer-layer default
- `MLP_MIDDLE_NEGATIVE_SLOPE` matters only when
  `MLP_MIDDLE_ACTIVATION=leaky_relu_square`

Pinned runnable code source:

- branch: `exp/039b-loop-band-activation-screen`
- commit: `5bbf12f`
- script:
  [train_gpt.py](/home/claude-user/ai-workspace/projects/parameter-golf/worktrees/039b-loop-band-activation/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py)

Looped middle physical layers are:

- `3,4,5`

Three arms:

### baseline — uniform current activation

- outer -> `LeakyReLU(0.5)^2`
- middle -> `LeakyReLU(0.5)^2`

### 039bA — middle penalized tanh

- outer -> `LeakyReLU(0.5)^2`
- middle -> `penalized_tanh`

### 039bB — middle tanh

- outer -> `LeakyReLU(0.5)^2`
- middle -> `tanh`

> 039bC (middle `LeakyReLU(0.3)²`) shelved — slope variants within the same
> family are no longer interesting now that s=0.5 is confirmed as the outer
> default. See `_shelved/` if needed.

Implementation note:

- outer layers keep the current fused path
- non-default middle activations may use eager execution in the MLP path
- quantization / serialization / deserialize support is explicitly out of scope
  for this first screen

## Regime

This is an explicitly training-only screen.

Pinned short-run intent:

- `4×H100`
- `SEED=42`
- `MAX_WALLCLOCK_SECONDS=1200`
- `ENABLE_LOOPING_AT=0.35`
- `TTT_ENABLED=0`
- `TRAINING_ONLY_SCREEN=1`

Compare:

- steps reached
- train loss trajectory
- validation loss / BPB from the pre-quant diagnostic
- train time

Out of scope for this spec:

- GPTQ / quantized artifact generation
- deserialize / rebank compatibility questions
- TTT

## Seed policy

Use one seed only:

- `42`

## Hardware ladder

1. `4×H100` only
2. no `8×H100` in this spec
3. no quantized eval in this spec

## Run protocol

Run three training-only jobs:

1. uniform baseline
2. `039bA` middle `penalized_tanh`
3. `039bB` middle `tanh`

Same seed, same wallclock, same env otherwise.

Execution rule:

- stop after the pre-quant diagnostic
- do not attempt to serialize or evaluate the quantized model in this spec

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
MLP_OUTER_ACTIVATION=leaky_relu_square
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
MAX_WALLCLOCK_SECONDS=1200
TTT_ENABLED=0
TRAINING_ONLY_SCREEN=1
```

Canonical launch block:

```bash
declare -A MIDDLE_ACTIVATION=(
  [baseline]=leaky_relu_square
  [039bA]=penalized_tanh
  [039bB]=tanh
)

for arm in baseline 039bA 039bB; do
  env \
    DATA_DIR=/workspace/parameter-golf/data \
    DATASETS_DIR=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
    TOKENIZER_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
    TRAIN_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin \
    VAL_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin \
    VAL_BYTES_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_*.bin \
    VOCAB_SIZE=8192 NUM_LAYERS=11 XSA_LAST_N=11 MODEL_DIM=512 NUM_KV_HEADS=4 NUM_HEADS=8 MLP_MULT=4.0 NEGATIVE_SLOPE=0.5 \
    MLP_OUTER_ACTIVATION=leaky_relu_square MLP_MIDDLE_LAYERS=3,4,5 \
    MLP_MIDDLE_ACTIVATION="${MIDDLE_ACTIVATION[$arm]}" \
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
    SEED=42 MAX_WALLCLOCK_SECONDS=1200 TTT_ENABLED=0 TRAINING_ONLY_SCREEN=1 \
    RUN_ID="039b-${arm}" \
    torchrun --standalone --nproc_per_node=4 train_gpt.py
done
```

## Acceptance

Interesting outcome:

- any middle-band alternative clearly beats the uniform baseline on the short
  training screen

Most interesting outcome:

- `039bA` wins, supporting the story that recurrent-band MLPs want a more
  bounded activation than the outer trunk

Kill criteria:

- all three middle-band alternatives are flat or worse than baseline
- gains are dominated by throughput/compiler artifacts rather than learning
  signal

## Open questions

- does a loop-band activation win survive quantization later, or is it only a
  training-only effect?
- is penalized-tanh better than just lowering the leak within the same family?
- should a winning loop-band activation later be combined with `040`, or tested
  separately first?
