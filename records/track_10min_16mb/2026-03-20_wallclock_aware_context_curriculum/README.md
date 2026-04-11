# Wallclock-Aware Context Curriculum

## Summary

This is a staged `10min_16mb` experiment focused on the bottleneck from the previous failed record attempt: weak early wallclock progress.

Main bet:

- train at `SHORT_SEQ_LEN=1024` for the early portion of the real remaining wallclock budget
- switch to `FINAL_SEQ_LEN=2048` later
- keep total batch tokens fixed
- buy more useful optimizer steps early, then recover longer-context training later

The rest of the stack stays deliberately conservative:

- mixed `int5` MLP / `int6` attention export
- `zstd-22` compression
- `Muon`
- `SWA`
- `SmearGate`
- `BigramHash`
- `OrthoInit`
- grad clip
- sliding-window evaluation

## Why This Exists

The previous export-rotation run was too slow and too weak early:

- `step 500`: `val_bpb 1.4119`
- `step 1000`: `val_bpb 1.3268`
- `step 1500`: `val_bpb 1.2907`

That suggested the first-order problem was underfitting by wallclock, not export cleverness. This trainer is meant to test a cleaner hypothesis: cheaper early sequence lengths may buy more BPB per millisecond than holding `2048` from step 1.

## Status

This folder is intentionally a work in progress.

- syntax check passes
- no official 8xH100 result is claimed yet
- we are waiting on compute credits before running the main experiment and ablations

## Main Toggles

- `SEQ_WARMUP_ENABLED=0` disables the curriculum and gives the clean baseline
- `SHORT_SEQ_LEN=1024` sets the early context length
- `FINAL_SEQ_LEN=2048` sets the later context length
- `SEQ_WARMUP_FRAC=0.30` controls how much of the post-warmup wallclock budget stays short
- `EARLY_ABORT_ENABLED=1` enables optional off-pace guardrails

## Planned Runs

1. `SEQ_WARMUP_ENABLED=0` baseline
2. `SEQ_WARMUP_ENABLED=1 SHORT_SEQ_LEN=1024 SEQ_WARMUP_FRAC=0.30`
3. Ablate `SEQ_WARMUP_FRAC`
4. If promising by `step 500 / 1000 / 1500`, run additional seeds

## Commands

Main 8-GPU run:

```bash
NCCL_IB_DISABLE=1 RUN_ID=seq_curriculum_clean MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Cheap smoke test:

```bash
RUN_ID=smoke_seq_curriculum MAX_WALLCLOCK_SECONDS=120 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```
