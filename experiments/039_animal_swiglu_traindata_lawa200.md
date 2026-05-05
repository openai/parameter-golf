# Experiment 039: Animal 8xH100 — SwiGLU Train Data + LAWA interval=200 (Run 015)

## Config
USE_SWIGLU=1, SWIGLU_HIDDEN=668, LAWA_ENABLED=1, LAWA_INTERVAL=200, EVAL_STRIDE=64, COMPILE_MODE=default
Train data, 8xH100 NV18 NVLink

## Results
- Steps: 12,398 @ 48.4ms/step
- LAWA: 19 snapshots averaged
- Artifact: 15,953,789 bytes ✅
- Standard eval: **1.2285 BPB** (LAWA hurt by 0.0017 vs 038)
- Sliding eval s64: **1.1950 BPB** (LAWA hurt by 0.0015 vs 038)

## Conclusion
LAWA still hurts even with fewer snapshots. Don't use for submission.

## wandb
- ID: bcq2nrlz
- URL: https://wandb.ai/ishanramrakhiani-bindwell/parameter-golf/runs/bcq2nrlz
