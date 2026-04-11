# Shared-Core U-Net Transformer — Leo Feasby

## Summary

A **single transformer block reused 9× in a U-Net structure** with learned skip connections. One shared core, 9-layer effective depth, 18.9M parameters.

## Results

| Track | val_bpb | Roundtrip bpb | Runtime | Notes |
|-------|---------|--------------|---------|-------|
| 10-min valid submission | 1.3053 | 1.2475 | 546s | Within 598s budget |
| Unlimited compute (2.3h) | **1.1454** | **1.1723** | ~2.3h | Still descending at cutoff |

## Key findings

- **seq_len=2048** gives ~0.07 bpb improvement over seq_len=1024 at minimal cost on H100s
- **Warmdown length is the dominant lever** — the loss follows warmdown faithfully, and the 10-min budget limits warmdown far more than compute
- **True floor is below 1.13 bpb** — loss was still dropping at ~0.003/500 steps when training stopped
- **WARMDOWN_START_STEP**: step-based warmdown trigger, decoupled from wallclock, allowing precise schedule control

## Submissions

- [`track_non_record_16mb/2026-03-22_SharedCore_Seq2048_Warmdown/`](track_non_record_16mb/2026-03-22_SharedCore_Seq2048_Warmdown/) — 2.3h unlimited compute run, **1.1454 val bpb**
