# Cutting BPB Below 1.0810

This document describes experiments and **environment variables** added to [`train_gpt_from_blob.py`](../train_gpt_from_blob.py) (decoded competition submission).

## New / notable environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLP_SWIGLU` | `0` | Set `1` to use SwiGLU MLP (`silu(fc)*fc_gate`) instead of `LeakyReLU^2`. Reduce `MLP_MULT` if artifact size is tight. |
| `TTT_SELECTIVE_LAYERS` | *(empty)* | Comma-separated block indices for TTT (e.g. `3,4,5`). Empty = adapt all parameters (default). |
| `GPTQ_RECURRENT_INT8` | `0` | Set `1` to quantize weights in `blocks[LOOP_START..LOOP_END]` at **int8**; others stay `MATRIX_BITS` (default int6). |
| `BIGRAM_VOCAB_SIZE` | `0` | Set e.g. `1024` to enable bigram hash embedding (additive to token embedding). `0` disables. |
| `BIGRAM_DIM` | `128` | Bigram embedding dimension (must match pipeline expectations if enabled). |

Existing knobs (see `Hyperparameters` in source): `QK_GAIN_INIT`, `TTT_*`, `MUON_WD`, `MATRIX_LR`, `EMA_DECAY`, `WARMDOWN_FRAC`, `GRAD_CLIP_NORM`, `EVAL_STRIDE`, `ROPE_DIMS`, `NUM_LOOPS`, `LOOP_START`, `LOOP_END`, `ENABLE_LOOPING_AT`, etc.

## Phase 1: sweeps (no code edits required beyond this file)

Use the scripts in [`scripts/`](../scripts/):

| Script | Purpose |
|--------|---------|
| `bpb_sweep_qk.sh` | `QK_GAIN_INIT` grid |
| `bpb_sweep_ttt.sh` | TTT epochs, chunk size, LR |
| `bpb_sweep_train.sh` | WD, MLR, EMA, warmdown, grad clip |
| `bpb_sweep_stride.sh` | `EVAL_STRIDE` (e.g. 32 vs 64) |
| `bpb_num_loops_experiment.sh` | `NUM_LOOPS=3` and wider loop range |
| `bpb_phase2_selective_ttt.sh` | `TTT_SELECTIVE_LAYERS=3,4,5` |
| `bpb_phase2_recurrent_int8.sh` | `GPTQ_RECURRENT_INT8=1` |
| `bpb_arch_swiglu.sh` | `MLP_SWIGLU=1` (+ optional `MLP_MULT`) |
| `bpb_arch_rope32.sh` | `ROPE_DIMS=32` |
| `bpb_arch_bigram.sh` | `BIGRAM_VOCAB_SIZE` / `BIGRAM_DIM` |
| `bpb_confirmation_seeds.sh` | seeds 42, 314, 999 |

Override `TRAIN_SCRIPT` or `NPROC_PER_NODE` if needed.

## Phase 2: targeted features (implemented in code)

- **Selective TTT**: `TTT_SELECTIVE_LAYERS=3,4,5`
- **Recurrent int8 GPTQ**: `GPTQ_RECURRENT_INT8=1` (uses `LOOP_START`/`LOOP_END` for layer range)
- **Deeper recurrence**: raise `NUM_LOOPS` (e.g. `3`) and/or widen `LOOP_START`/`LOOP_END`; see `bpb_num_loops_experiment.sh`

## Phase 3: architecture toggles

- **SwiGLU**: `MLP_SWIGLU=1` (optionally lower `MLP_MULT`)
- **RoPE**: `ROPE_DIMS=32` (was 16 in baseline)
- **Bigram**: `BIGRAM_VOCAB_SIZE=1024` (and tune `BIGRAM_DIM`)

Bigram tensors are saved as **float16 passthrough** in GPTQ to avoid extra Hessian plumbing.

## Confirmation runs

`bpb_confirmation_seeds.sh` runs seeds **42**, **314**, **999** with your chosen env (edit the script or export vars before sourcing).

## Prerequisites

- Data: `DATA_DIR`, `fineweb10B_sp{VOCAB_SIZE}`, tokenizer
- 8 GPUs: `WORLD_SIZE` must divide 8 (`grad_accum_steps = 8 // world_size`)
- FlashAttention 3: `from flash_attn_interface import flash_attn_func`
