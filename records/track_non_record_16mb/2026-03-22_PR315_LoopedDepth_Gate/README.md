## Gate Branch: PR315-Derived Looped / Tied Depth

This folder is the structural-depth gate branch for the current strategy program. It is forked from the exact [PR #325](https://github.com/openai/parameter-golf/pull/325) looped-transformer implementation, which itself was built on top of the `PR315` frontier feature set.

This is not the primary submission candidate. Its purpose is to answer one question cleanly:

Can a looped / partially tied depth design stay competitive enough on throughput to be worth further tuning on the current frontier base?

## Base Design

Inherited from `PR325`:
- frontier-style `PR315` feature stack
- `EMA=0.997`
- `Partial RoPE` on `16/64` head dims
- `LN Scale`
- `XSA` on the last 4 effective attention blocks
- bigram hash `2048x128`
- seq2048, global batch `786432`

Looped-depth structure:
- `NUM_LAYERS=6`
- `MODEL_DIM=640`
- `NUM_HEADS=10`
- `NUM_KV_HEADS=5`
- `LOOP_CORE_LAYERS=2`
- `LOOP_REPEATS=5`
- `LOOP_ATTN_EVERY=2`
- `LOOP_ADAPTER_DIM=64`
- `LOOP_REPEAT_EMBED=1`
- effective executed depth: `14`

## What This Gate Is For

Use this branch to test:
- training throughput under the official 10-minute budget
- whether shared-core depth is still viable on a frontier-style base
- loop geometry sweeps
- shared-vs-untied allocation choices
- adapter and repeat-embedding ablations

Do not treat the inherited `PR325` numbers as current evidence for this repo state. Fresh validation is still required.

## Suggested First Gate Runs

Keep the original structural settings first:

```bash
NUM_LAYERS=6 \
MODEL_DIM=640 \
NUM_HEADS=10 \
NUM_KV_HEADS=5 \
LOOP_CORE_LAYERS=2 \
LOOP_REPEATS=5 \
LOOP_ATTN_EVERY=2 \
LOOP_ADAPTER_DIM=64 \
LOOP_REPEAT_EMBED=1 \
EMA_ENABLED=1 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Those settings are now also baked into the script defaults, along with the PR315-style optimizer and feature defaults:
- `EMA_ENABLED=1`, `EMA_DECAY=0.997`, `SWA_ENABLED=0`
- `XSA_LAST_N=4`
- `ROPE_DIMS=16`, `LN_SCALE=1`
- `BIGRAM_VOCAB_SIZE=2048`
- `MATRIX_LR=0.025`, `SCALAR_LR=0.025`, `TIED_EMBED_LR=0.035`
- `MUON_MOMENTUM=0.99`, `MUON_MOMENTUM_WARMUP_START=0.92`, `MUON_MOMENTUM_WARMUP_STEPS=1500`
- `MUON_WD=0.04`, `ADAM_WD=0.04`
- `ITERATIONS=9000`, `WARMDOWN_ITERS=3000`

Then sweep only the loop geometry:
- `LOOP_REPEATS`
- `LOOP_ATTN_EVERY`
- `LOOP_CORE_LAYERS`
- `LOOP_ADAPTER_DIM`

## Current Validation Status

This branch is implemented and ready for gate testing, but it is not freshly validated yet.

Inherited historical reference from `PR325`:
- sliding `val_bpb`: `1.14620421`
- total bytes: `15,589,099`
- step average: `123.70ms`

Those numbers are useful only as a reference point for the original branch, not as current measured results for this workspace.
