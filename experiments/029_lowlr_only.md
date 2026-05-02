# Experiment 029: Lower LR Only (ablation)

## Status: RUNNING
## Config: Baseline arch + MATRIX_LR=0.02 + SCALAR_LR=0.02 + TIED_EMBED_LR=0.03 + softcap=15 + eps=1e-10
## Hypothesis: Test if lower LR alone helps at 2K steps (PR #39 showed it helps at 14K steps)
## Note: LR benefit may only show at longer training
