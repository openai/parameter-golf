This records the submissions for `ContextFuse-2048-BigramSmear`.

`val_bpb` is the primary target for this entry. This package builds on our earlier PR `#143`, `ContextFuse-2048`, and keeps the same baseline-derived train@2048 path while adding compression-aware input features and a stronger mixed-int6 export.

This is not presented as a new SOTA claim.

## Summary

- Prior submission: PR `#143`, `ContextFuse-2048`, `val_bpb=1.17792945`
- Corrected canonical run: `val_bpb=1.15369565`
- Improvement over PR `#143`: `-0.02423380` BPB
- Canonical training log: `train.log`
- Canonical corrected fixed-eval log: `train_fixed_eval_seed1337.log`

## Canonical Result

The canonical metric for this submission is the corrected reevaluation of the `SEED=1337` model:

- corrected `val_loss=1.94796677`
- corrected `val_bpb=1.15369565`
- original training time for this model: `589978ms`
- corrected fixed-eval time: `150215ms`
- compressed model bytes: `15267279`
- standalone artifact bytes in this folder: `15331125`

The corrected metric comes from re-running the saved raw checkpoint through the fixed sliding-window scorer in `train_fixed_eval_seed1337.log`.

## Scoring Correction

The original submission branch used a sliding-window evaluator that could emit multiple truncated tail windows and rescore the same final stride tokens more than once when `EVAL_STRIDE < TRAIN_SEQ_LEN`.

This folder now fixes that bug in `train_gpt.py` by:

1. generating only full windows plus at most one final tail-alignment window
2. scoring only the globally new target span for each window instead of blindly scoring the last `stride` targets of every non-first window

Because of that fix:

- `train.log`, `train_seed42.log`, and `train_seed7.log` are retained as original automatically produced training logs from the pre-fix runs
- their final `val_loss` / `val_bpb` lines should be treated as pre-fix numbers, not canonical corrected metrics
- this package does **not** make a three-seed statistical claim after the scorer fix

The submission metadata uses only the corrected canonical seed `1337` value.

## What Changed From PR #143

PR `#143` (`ContextFuse-2048`) already had:

- `TRAIN_SEQ_LEN=2048`
- sliding-window final eval with `EVAL_STRIDE=64`
- fp16 embedding preservation

This follow-up adds the strongest compression-aware components we found that still transferred honestly:

1. `BigramHashEmbedding` on the input path
2. `SmearGate` to blend each token representation with the previous token
3. mixed `int6` export for large `mlp` and `attn` matrices
4. `MLP_HIDDEN=1536` so the int6 budget buys back a larger MLP
5. `MUON_WEIGHT_DECAY=0.02`
6. `SWA_ENABLED=1` over the late low-LR phase
7. corrected control-tensor handling so only `bigram.scale` is exempted from normal quantization
8. a narrower fp16 keep rule: `FP16_KEEP_NAME_PATTERNS=tok_emb,blocks.8.attn.c_k`

## Method Credit

This submission intentionally credits the prior work it builds on:

- PR `#143`
  - our original `ContextFuse-2048` submission
  - base scaffold for train@2048, sliding eval, and fp16 embedding preservation
- PR `#135`
  - `BigramHash + SmearGate + mixed int6 + 3x MLP`
- PR `#162`
  - `Muon WD + SWA + refined fp16 keep pattern`
- public record folders already in this repo
  - `2026-03-18_LongContextSeq2048`
  - `2026-03-19_SlidingWindowEval`

## Configuration

- Layout:
  - `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4`
  - `MLP_HIDDEN=1536`
- Input enhancements:
  - `USE_SMEAR_GATE=1`
  - `BIGRAM_VOCAB_SIZE=4096`
  - `BIGRAM_DIM=128`
- Export strategy:
  - `MIXED_INT6_EXPORT=1`
  - `INT6_CATEGORIES=mlp,attn`
  - `FP16_EMBED_PASSTHROUGH=1`
  - `FP16_LATE_K_LAYERS=0`
  - `FP16_KEEP_NAME_PATTERNS=tok_emb,blocks.8.attn.c_k`
  - `CONTROL_TENSOR_NAME_PATTERNS=attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,bigram.scale`
- Training:
  - `TRAIN_BATCH_TOKENS=786432`
  - `TRAIN_SEQ_LEN=2048`
  - `MAX_WALLCLOCK_SECONDS=590`
  - `TIED_EMBED_LR=0.03 MATRIX_LR=0.02 SCALAR_LR=0.02`
  - `MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500`
  - `MUON_BACKEND_STEPS=5`
  - `ADAM_WEIGHT_DECAY=0.01`
  - `MUON_WEIGHT_DECAY=0.02`
  - `GRAD_CLIP_NORM=0.3`
  - `WARMDOWN_ITERS=3000`
  - `SWA_ENABLED=1 SWA_START_FRAC=0.5 SWA_EVERY=200`
- Evaluation:
  - `EVAL_STRIDE=64`
  - `EVAL_BATCH_SEQS=32`

## Artifact Accounting

The intended submission artifact is the standalone `train_gpt.py` in this record folder, not the Modal wrapper used during experimentation.

Standalone artifact accounting for this folder:

- `train_gpt.py`: `63846` bytes
- canonical compressed model (`seed 1337`): `15267279` bytes
- canonical standalone total: `15331125` bytes

That leaves `668875` bytes of headroom under the `16000000` byte cap.

## Reproduction Command

```bash
cd records/track_10min_16mb/2026-03-20_ContextFuse-2048-BigramSmear
RUN_ID=contextfuse2048_bigramsmear \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=590 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=100 \
TRAIN_BATCH_TOKENS=786432 \
VAL_BATCH_SIZE=524288 \
TRAIN_SEQ_LEN=2048 \
NUM_LAYERS=9 \
MLP_HIDDEN=1536 \
USE_SMEAR_GATE=1 \
BIGRAM_VOCAB_SIZE=4096 \
BIGRAM_DIM=128 \
MIXED_INT6_EXPORT=1 \
INT6_CATEGORIES=mlp,attn \
CONTROL_TENSOR_NAME_PATTERNS=attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,bigram.scale \
FP16_EMBED_PASSTHROUGH=1 \
FP16_LATE_K_LAYERS=0 \
FP16_KEEP_NAME_PATTERNS=tok_emb,blocks.8.attn.c_k \
ORTHO_INIT=1 \
SWA_ENABLED=1 \
SWA_START_FRAC=0.5 \
SWA_EVERY=200 \
TIED_EMBED_LR=0.03 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_BACKEND_STEPS=5 \
ADAM_WEIGHT_DECAY=0.01 \
MUON_WEIGHT_DECAY=0.02 \
GRAD_CLIP_NORM=0.3 \
WARMDOWN_ITERS=3000 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=32 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included Files

- `train_gpt.py` — standalone record-folder training/eval/export script with the corrected sliding-window scorer
- `train.log` — original `SEED=1337` training log from the pre-fix run
- `train_fixed_eval_seed1337.log` — corrected exact reevaluation log for the canonical saved model
- `train_seed42.log` — original `SEED=42` pre-fix training log, included for transparency only
- `train_seed7.log` — original `SEED=7` pre-fix training log, included for transparency only
- `submission.json` — metadata for the entry
- `README.md` — this file
