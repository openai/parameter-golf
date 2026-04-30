# Models: Final Architectures

This directory contains the clean, final versions of each model family. For the research journey that produced these models, see `exploration/`.

## Model Families

### Behemoth
Macro-sidechannel transformer with encoder-decoder distillation. 34.7M parameters. 10 iterations of development (v1-v10), with a critical gradient explosion fix in v12. Includes 8xH100 full-scale runs.

**Best BPB:** See `Behemoth/8gpu-runs/`
**Key file:** `Behemoth/train_gpt.py` (latest stable version)
**History:** `Behemoth/history/` (v01-v10 with scripts and logs)
**Origin:** exploration/phase-5 (genesis) → phase-6 (evolution)

### trans-hier-012
Transformer-Hierarchical with encoder-decoder distillation and int4 QAT. The architecture that first broke the 1.18 BPB barrier.

**Key file:** `trans-hier-012/train_gpt.py`
**Runs:** 8gpu, ablation-nodistill, ablation-nomacro
**Origin:** exploration/phase-6

### r-series
Mixed-depth recurrence variants. Explores whether combining different recurrence depths (r1+r2+r4, r1+r2+r4+r8) outperforms uniform depth.

**Key files:** `r-series/train_gpt_r124.py`, `train_gpt_r1248.py`
**Runs:** 1gpu and 8gpu variants for each configuration
**Origin:** exploration/phase-7
