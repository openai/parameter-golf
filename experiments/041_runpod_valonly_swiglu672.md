# Experiment 041: Runpod 8xH100 — Val-Only SwiGLU h=672 + LAWA (Run 011)

## Config
USE_SWIGLU=1, SWIGLU_HIDDEN=672, LAWA_ENABLED=1, LAWA_INTERVAL=100, EVAL_STRIDE=64, COMPILE_MODE=default
Val data, 8xH100 SXM Runpod

## Results
- Steps: 13,410 @ 44.75ms/step
- LAWA: 37 snapshots (hurt)
- Artifact: 16,039,716 bytes ❌ (39KB over)
- Standard eval: **1.1093 BPB**
- Sliding eval s64: **1.0712 BPB**
- NTK RoPE: 1.1450 (worse)

## Notes
Artifact over budget. LAWA hurt. Led to reducing SWIGLU_HIDDEN to 668.

## wandb
- ID: j9hsn8le
- URL: https://wandb.ai/ishanramrakhiani-bindwell/parameter-golf/runs/j9hsn8le
