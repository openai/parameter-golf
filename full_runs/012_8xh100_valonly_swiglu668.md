# Experiment 012: 8xH100 Runpod — Val-Only Training (SwiGLU hidden=668, no LAWA)

## Status: RUNNING

## Config
- **Instance**: Runpod 8xH100 SXM (NVLink)
- **Script**: train_gpt_valonly.py (trains on validation data only)
- **Key settings**: USE_SWIGLU=1, SWIGLU_HIDDEN=668, LAWA_ENABLED=0, EVAL_STRIDE=64, COMPILE_MODE=default
- **PyTorch**: 2.10.0+cu128
- **Changes from 011**: SWIGLU_HIDDEN 672→668 (saves ~55K params to fit under 16MB), LAWA disabled (it hurt in 011)

## Results
*Running — will update when complete*

## wandb
- Run: https://wandb.ai/ishanramrakhiani-bindwell/parameter-golf/runs/5nw2jnmi
