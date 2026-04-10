# Sequence Curriculum: Short-to-Long Training Schedule

**Author:** genji0306 (Opensens Research)
**Date:** 2026-03-27
**Track:** Non-record / Exploration
**Status:** Pending H100 validation

## Hypothesis

All current leaderboard submissions use a fixed sequence length throughout training. Starting with shorter sequences (e.g., 512) and progressively increasing to the target length (e.g., 2048) yields faster per-step throughput in the early phase, resulting in significantly more total gradient updates within the 10-minute training budget. Early layers primarily learn local patterns that do not require long context.

## Approach

Implement a multi-phase sequence length schedule:
- Phase 1: Train at reduced sequence length for faster step times
- Phase 2: Transition to intermediate length
- Phase 3: Train at full target length for the final portion of steps

The approach targets the training efficiency bottleneck directly — more steps in the same wall-clock budget.

## Verification Plan

- Measure total steps achieved vs. fixed-length baseline under the same 10-min cap
- Compare val_bpb at matched wall-clock time
- Validate that curriculum-trained models match or exceed fixed-length quality

## Compute Needed

Estimated 3-4 runs on 1xH100 (~40 min total).
