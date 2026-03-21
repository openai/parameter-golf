# Experiment 015: Animal 8xH100 — SwiGLU Train Data + LAWA + Sliding Eval

## Status: COMPLETED

## Config
- **Instance**: animal.netravi.net 8xH100 SXM (NV18 full NVLink mesh)
- **Script**: train_gpt_slidingeval.py
- **Key settings**: USE_SWIGLU=1, SWIGLU_HIDDEN=668, LAWA_ENABLED=1, LAWA_INTERVAL=200, EVAL_STRIDE=64, COMPILE_MODE=default
- **Data**: Train data (fineweb_train_*.bin)

## Results
| Metric | Value |
|--------|-------|
| Steps completed | 12,398 |
| Step avg | 48.4ms |
| LAWA snapshots | 19 (including final) |
| Artifact size | **15,953,789 bytes ✅** |
| **Post-quant standard eval** | **1.2285 BPB** |
| **Sliding window stride=64** | **1.1950 BPB** |
| Sliding window stride=128 | 1.1952 BPB |
| Sliding window stride=512 | 1.1985 BPB |

## Comparison with 014 (no LAWA)
| Metric | 014 (no LAWA) | 015 (LAWA) | Diff |
|--------|--------------|------------|------|
| Standard eval | 1.2268 | 1.2285 | +0.0017 (LAWA hurt) |
| Sliding eval | 1.1935 | 1.1950 | +0.0015 (LAWA hurt) |

## Key Findings
- LAWA with 19 snapshots (interval=200) still slightly hurts — consistent with all prior LAWA experiments
- LAWA appears to be a net negative for this architecture/config regardless of snapshot count
- **Conclusion: Don't use LAWA for submission**

## wandb
- Run ID: bcq2nrlz
- Run name: 6f26fe1a-d603-4293-9e37-ea506f3d13e6
- URL: https://wandb.ai/ishanramrakhiani-bindwell/parameter-golf/runs/bcq2nrlz
