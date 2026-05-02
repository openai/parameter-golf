# Experiment 016: Animal 8xH100 — Thunder Config on NVLink

## Status: QUEUED (after 015)

## Config
- **Instance**: animal.netravi.net 8xH100 SXM (NV18 full NVLink mesh)
- **Script**: 8xH100_train_gpt.py (original Thunder script)
- **Key settings**: USE_SWIGLU=1, SWIGLU_HIDDEN=672, MATRIX_LR=0.04, SCALAR_LR=0.04, TIED_EMBED_LR=0.05, LAWA_ENABLED=1, LAWA_INTERVAL=100, WARMDOWN_ITERS=3600, EVAL_STRIDE=256, COMPILE_MODE=default
- **Data**: Train data (fineweb_train_*.bin)

## Hypothesis
The Thunder run (exp 010) got 1.2364 BPB with only 8,420 steps (71ms/step PCIe bottleneck). On NVLink at ~48ms/step we should get ~12,500 steps — 50% more training. This should significantly improve the BPB.

## Difference from 014/015
- SWIGLU_HIDDEN=672 (vs 668) — slightly larger model
- LAWA_INTERVAL=100 (more snapshots)
- WARMDOWN_ITERS=3600
- EVAL_STRIDE=256 (vs 64)
- Same LRs as original Thunder run

## Results
*Pending*

## Run Command
```bash
USE_SWIGLU=1 WANDB_PROJECT=parameter-golf torchrun --nproc_per_node=8 train_gpt_swiglu.py
```

## Notes
- This is the exact wandb "8xH100_swiglu_lowlr_lawa" config but on proper NVLink hardware
- May need COMPILE_MODE=default (reduce-overhead crashes on torch 2.10)
- Artifact may be slightly over 16MB with h=672 — if so, fall back to h=668
