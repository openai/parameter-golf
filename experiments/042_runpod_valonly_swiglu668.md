# Experiment 042: Runpod 8xH100 — Val-Only SwiGLU h=668, No LAWA (Run 012)

## Config
USE_SWIGLU=1, SWIGLU_HIDDEN=668, LAWA_ENABLED=0, EVAL_STRIDE=64, COMPILE_MODE=default
Val data, 8xH100 SXM Runpod

## Results
- Steps: 12,554 @ 47.80ms/step
- Artifact: 15,988,602 bytes ✅
- Standard eval: **1.0972 BPB**
- Sliding eval: *(connection dropped before result)*

## wandb
- ID: 5nw2jnmi
- URL: https://wandb.ai/ishanramrakhiani-bindwell/parameter-golf/runs/5nw2jnmi
