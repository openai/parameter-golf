# PR 1147 — Partition Function Inflation — Analytical (Non-Record)

**Author:** Robby Sneiderman
**Branch date:** 2026-03-30
**Claimed BPB:** not stated (analytical submission — neural-only baseline cited as 1.130 BPB)
**Artifact size:** not stated
**Seeds:** not stated
**Hardware:** 8×H100 (diagnostic-only)

## Files retrieved
- `records__track_non_record_16mb__2026-03-30_PartitionFunctionInflation_Analytical__README.md`
- `records__track_non_record_16mb__2026-03-30_PartitionFunctionInflation_Analytical__train_gpt.py`

## Environment variables (diagnostic, from README)
- `MEASURE_Z=1` — Log per-token partition function Z during eval
- `NORM_LAMBDA=<float>` — Interpolation weight for normalization correction
- `NORMALIZE_STEPWISE=1` — Apply per-order stepwise normalization diagnostic

## Claimed changes (from README, verbatim)
"This is a non-record analytical submission documenting a normalization pathology in Dirichlet-smoothed n-gram caches used for the Parameter Golf challenge.

## Key Finding
The partition function Z* exceeds 1000 at 1M buckets (alpha=2), meaning Dirichlet cache probabilities are inflated by ~10x before mixing with the neural model. A lambda sweep over the normalization correction shows BPB is approximately linear in the normalization weight, and all bucket-size configurations perform worse than the neural-only baseline (1.130 BPB) after proper normalization. The apparent gains from cache augmentation are an artifact of unnormalized probability mass.

Full paper, experiment logs, and analysis code: https://github.com/Robby955/partition-function-inflation"
