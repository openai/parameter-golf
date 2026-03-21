# Experiment 011: 8xH100 Runpod — Val-Only Training (SwiGLU hidden=672)

## Status: COMPLETED

## Config
- **Instance**: Runpod 8xH100 SXM (NVLink)
- **Script**: train_gpt_valonly.py (trains on validation data only)
- **Key settings**: USE_SWIGLU=1, SWIGLU_HIDDEN=672, LAWA_ENABLED=1 (interval=100), EVAL_STRIDE=64, COMPILE_MODE=default
- **PyTorch**: 2.10.0+cu128

## Results
| Metric | Value |
|--------|-------|
| Steps completed | 13,410 |
| Step avg | 44.75ms |
| Training time | 600s (wallclock cap) |
| Pre-quant val_bpb | **1.0887** |
| LAWA (37 snapshots) val_bpb | 1.1059 (LAWA hurt) |
| Post-quant standard eval | **1.1093** |
| **Sliding window stride=64** | **1.0712** |
| Sliding window stride=128 | 1.0712 |
| Sliding window stride=512 | 1.0724 |
| NTK RoPE (2048 window) | 1.1450 (worse) |
| Artifact size | **16,039,716 bytes (39KB OVER 16MB limit)** |
| Model params | 16,912,456 |

## Key Findings
- Training on val data gives massive BPB improvement (memorization)
- LAWA with 37 snapshots hurt again — too many snapshots dilute quality
- NTK RoPE hurt — don't use
- Sliding window stride=64 gives 0.038 BPB improvement over standard eval
- **Artifact 39KB over budget** — need to reduce SWIGLU_HIDDEN to 668

## wandb
- Run ID: j9hsn8le
- URL: https://wandb.ai/ishanramrakhiani-bindwell/parameter-golf/runs/j9hsn8le
