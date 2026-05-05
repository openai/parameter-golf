# Experiment 056: relu² MLP_HIDDEN=1400 to fit budget

## Status: COMPLETED

## Results
| Metric | Value |
|--------|-------|
| Steps | 11,160 @ 53.8ms/step |
| Artifact | **13,432,900 bytes ✅** (2.5MB headroom — too much trimmed) |
| Standard eval | 1.1951 BPB |
| Sliding eval | *waiting* |

## Key Finding
MLP_HIDDEN=1400 (vs 1536) costs 0.021 BPB — too aggressive. Need ~1500 instead.
The artifact has 2.5MB headroom — wasted space.

## wandb
- Run name: 056_fit_budget_mlp1400
