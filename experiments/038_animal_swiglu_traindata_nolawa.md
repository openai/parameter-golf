# Experiment 038: Animal 8xH100 — SwiGLU Train Data, No LAWA (Run 014)

## Config
USE_SWIGLU=1, SWIGLU_HIDDEN=668, LAWA_ENABLED=0, EVAL_STRIDE=64, COMPILE_MODE=default
Train data, 8xH100 NV18 NVLink

## Results
- Steps: 11,907 @ 50.0ms/step
- Artifact: 15,954,593 bytes ✅
- Standard eval: **1.2268 BPB** (beats baseline 1.2244)
- Sliding eval s64: **1.1935 BPB**
- NTK RoPE: 1.2505 (worse)

## wandb
- ID: pf453kpx
- URL: https://wandb.ai/ishanramrakhiani-bindwell/parameter-golf/runs/pf453kpx
