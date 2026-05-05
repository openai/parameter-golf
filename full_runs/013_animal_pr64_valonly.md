# Experiment 013: Animal 8xH100 — PR64 Val-Only (killed early)

## Status: KILLED at step ~3600

## Config
- **Instance**: animal.netravi.net 8xH100 SXM (NV18 full NVLink mesh)
- **Script**: train_gpt_pr64_valonly.py (PR#64's CombinedOptimal)
- **Key settings**: MLP_MULT=3, seq_len=4096, int6 STE QAT, matrix_lr=0.02, muon_momentum=0.99, warmdown=3000, eval_stride=64
- **PyTorch**: 2.10.0+cu128

## Partial Results
| Step | val_bpb | train_time | step_avg |
|------|---------|------------|----------|
| 1000 | 1.3502  | 57s        | 57.3ms   |
| 3000 | 1.2590  | 174s       | 57.9ms   |

Killed at step ~3600 by user request to run our SwiGLU script instead.

## Notes
- 57-58ms/step on NV18 NVLink (matches PR64's reported 10,438 steps in 10 min)
- First crash: Triton cache tried to write to full root disk. Fixed with TRITON_CACHE_DIR=/data/
- PR64 reported 0.9695 BPB final — we didn't get to see final results
