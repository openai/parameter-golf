# QAT Int6 + MLP3 + Sliding Window + Overtone Init

**Track:** 10min_16mb
**Status:** code snapshot only; no verified H100 training log checked in yet

This folder captures the first submission-style snapshot of the int6/QAT branch. It is organized like a record folder so the implementation can be reviewed and reproduced, but it is not yet a complete leaderboard submission because the corresponding `train.log` and final track-compliant metrics are not included.

## Core changes in this snapshot

### Int6 quantization-aware training
The model trains with fake per-row quantization in the forward pass so weights learn to tolerate export at int6 precision. This is controlled by `USE_QAT=1 QAT_BITS=6`.

### Int6 packed export
Large matrices are quantized to int6 and packed `4 values -> 3 bytes`, reducing payload size versus int8 and freeing budget for more parameters under the 16MB cap. This is controlled by `INT6_EXPORT=1`.

### Explicit MLP sizing
`MLP_HIDDEN=992` replaces the coarse `MLP_MULT` knob with an exact hidden width so the parameter budget can be targeted much more tightly.

### Sliding-window validation
`SW_STRIDE=256` and `EVAL_SEQ_LEN=1408` score tokens with more left context than the training sequence length, using NTK-RoPE extrapolation to extend context at eval time.

### Initialization and schedule changes
The snapshot also includes overtone embedding initialization, a depth-dependent `resid_mix` initialization, and full-budget warmdown to reduce quantization damage near the end of training.

## Target configuration

```bash
NUM_LAYERS=11
MODEL_DIM=448
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_HIDDEN=992
USE_QAT=1
QAT_BITS=6
INT6_EXPORT=1
SW_STRIDE=256
EVAL_SEQ_LEN=1408
WARMDOWN_ITERS=20000
MATRIX_LR=0.06
TIED_EMBED_LR=0.07
SCALAR_LR=0.06
ADAM_WEIGHT_DECAY=0.01
MUON_BACKEND_STEPS=5
```

## Included files

- `train_gpt.py`: self-contained code snapshot for this branch state
- `README.md`: implementation notes and intended run configuration
- `submission.json`: metadata placeholder for later completion with verified run outputs

## Missing for a merge-ready record PR

- `train.log` from the actual run
- final `val_loss`, `val_bpb`, artifact bytes, and code bytes
- confirmation that the run meets the `10min_16mb` track constraints
