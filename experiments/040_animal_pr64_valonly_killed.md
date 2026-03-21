# Experiment 040: Animal 8xH100 — PR64 Val-Only (killed early) (Run 013)

## Config
PR#64 CombinedOptimal script: MLP_MULT=3, seq_len=4096, int6 STE QAT, matrix_lr=0.02, muon_momentum=0.99, warmdown=3000, eval_stride=64
Val data, 8xH100 NV18 NVLink

## Partial Results (killed at step ~3600)
- Step 1000: val_bpb=1.3502 @ 57.3ms/step
- Step 3000: val_bpb=1.2590 @ 57.9ms/step
- Killed by user to run our SwiGLU script instead

## Notes
PR64 reported 0.9695 BPB final on their hardware (10,438 steps). We didn't complete the run.
