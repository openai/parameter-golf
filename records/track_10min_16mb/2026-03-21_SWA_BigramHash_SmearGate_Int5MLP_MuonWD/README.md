# SWA + BigramHash + SmearGate + Int5MLP + MuonWD + zstd-22

**Track:** 10min_16mb
**Status:** code snapshot only; no verified H100 training log checked in yet

This folder captures the follow-on submission-style snapshot that builds on the March 19 int6/QAT branch. As with the earlier folder, this is organized for code review and later reproduction, but it is not yet a complete record submission because the run log and final measured metrics are absent.

## Core changes in this snapshot

### SWA during warmdown
During the low-learning-rate phase, the script periodically collects checkpoints and averages them before export. The intent is to smooth the final basin and reduce post-quantization variance.

### BigramHash embedding
A hashed bigram table is added on top of the unigram token embedding to inject cheap short-range contextual features before the transformer stack.

### SmearGate
A learned per-dimension gate blends each token embedding with the previous token embedding before attention, adding another low-cost local-context path.

### Mixed int5/int6 quantization
MLP weights are quantized more aggressively than attention weights, recovering artifact budget that can be spent on model capacity elsewhere.

### Muon weight decay and zstd compression
The snapshot adds direct weight decay inside Muon and uses `zstd` level 22 when available, with graceful fallback to `zlib`.

## Inherited stack

- int6 QAT
- packed int6 export path
- explicit `MLP_HIDDEN=992`
- sliding-window evaluation with longer eval context
- overtone embedding init
- depth-scheduled `resid_mix`
- aggressive full-budget warmdown

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
SWA_ENABLED=1
SWA_START_FRAC=0.4
SWA_EVERY=50
BIGRAM_VOCAB_SIZE=10240
BIGRAM_DIM=64
USE_SMEAR_GATE=1
MUON_WEIGHT_DECAY=0.04
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
