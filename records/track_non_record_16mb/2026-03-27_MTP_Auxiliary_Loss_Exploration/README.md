# Multi-Token Prediction Auxiliary Loss Exploration

**Author:** genji0306 (Opensens Research)
**Date:** 2026-03-27
**Track:** Non-record / Exploration
**Status:** Pending H100 validation

## Hypothesis

The existing `train_gpt.py` codebase includes a fully implemented Multi-Token Prediction (MTP) head that predicts tokens at position `t+k` from the final hidden state. This MTP infrastructure is present but disabled (`MTP_NUM_HEADS=0`) across all current leaderboard submissions. Enabling MTP as an auxiliary training loss may improve representation quality at zero artifact size cost, since the MTP head is excluded from the exported model.

## Approach

Enable the existing MTP code path with a small auxiliary loss weight. The MTP head is discarded at export, so there is no impact on the 16MB artifact constraint. This exploration requires no code modifications — only environment variable configuration.

## Verification Plan

- Compare val_bpb with and without MTP on the current top-performing stack
- Sweep loss weight and number of prediction heads
- Confirm artifact size is unchanged after export

## Compute Needed

Estimated 2-3 runs on 1xH100 (~30 min total).
