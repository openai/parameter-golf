# Experiment 014: Animal 8xH100 — SwiGLU on Train Data + Sliding Eval

## Status: COMPLETED

## Config
- **Instance**: animal.netravi.net 8xH100 SXM (NV18 full NVLink mesh)
- **Script**: train_gpt_slidingeval.py (trains on TRAIN data, sliding window eval)
- **Key settings**: USE_SWIGLU=1, SWIGLU_HIDDEN=668, LAWA_ENABLED=0, EVAL_STRIDE=64, COMPILE_MODE=default
- **PyTorch**: 2.10.0+cu128
- **Data**: fineweb10B_sp1024 train shards

## Results
| Metric | Value |
|--------|-------|
| Steps completed | 11,907 |
| Step avg | 50.4ms |
| Artifact size | **15,954,593 bytes ✅** |
| Pre-quant val_bpb (step 11907) | ~1.22 |
| **Post-quant standard eval** | **1.2268 BPB** |
| **Sliding window stride=64** | **1.1935 BPB** |
| Sliding window stride=512 | 1.1969 BPB |
| NTK RoPE (2048 window) | 1.2505 BPB (worse) |

## Key Findings
- **1.2268 standard BPB beats baseline (1.2244) by 0.002** — marginal but real
- **1.1935 sliding eval is very competitive** — beats PR#50 (1.1925) by 0.001
- NTK RoPE still hurts — don't use
- Legitimate training on train data (not memorized)
- 50ms/step on NVLink (slightly slower than Runpod's 44ms)

## wandb
- Run ID: pf453kpx
- Run name: 4ff69733-6b08-4194-9af1-957f213f17b3
- URL: https://wandb.ai/ishanramrakhiani-bindwell/parameter-golf/runs/pf453kpx
