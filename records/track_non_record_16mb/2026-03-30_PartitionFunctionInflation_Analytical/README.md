# Partition Function Inflation -- Analytical (Non-Record)

This is a **non-record analytical submission** documenting a normalization
pathology in Dirichlet-smoothed n-gram caches used for the Parameter Golf
challenge.

## Key Finding

The partition function Z* exceeds 1000 at 1M buckets (alpha=2), meaning
Dirichlet cache probabilities are inflated by ~10x before mixing with the
neural model. A lambda sweep over the normalization correction shows BPB
is approximately linear in the normalization weight, and **all bucket-size
configurations perform worse than the neural-only baseline (1.130 BPB)
after proper normalization**. The apparent gains from cache augmentation
are an artifact of unnormalized probability mass.

## Paper and Data

Full paper, experiment logs, and analysis code:
<https://github.com/Robby955/partition-function-inflation>

## Diagnostic Environment Variables

| Variable | Purpose |
|---|---|
| `MEASURE_Z=1` | Log per-token partition function Z during eval |
| `NORM_LAMBDA=<float>` | Interpolation weight for normalization correction |
| `NORMALIZE_STEPWISE=1` | Apply per-order stepwise normalization diagnostic |

## Reproducibility

Run the submission script with `EVAL_ONLY=1` and the diagnostic env vars
above on 8xH100. See the linked repository for full instructions.
