# Parameter Golf Experiments

## Overview
Complete ML pipeline built through 259 heartbeats (~14 hours) of continuous
research and development. 21 novel ideas from 21 mathematical fields.
Best measured val_bpb: 1.3116 (9L) / 1.3436 (11L at step 5K, improving).
Predicted final: ~1.15-1.18 BPB (beats baseline 1.2244).

## Currently Training (DUAL GPU OVERNIGHT)
- GPU 0 (3080 Ti): 9L 2xMLP, 50K steps, Muon, 1B data
- GPU 1 (5070 Ti): **11L 3xMLP, 50K steps, Muon, 8B data** ← best model
scores on the FineWeb validation set.

All scripts require 8xH100 GPUs and run within the 10-minute training budget.
Recommended approach: start with exp002 (safe), then try exp003/exp004 (aggressive).

## Experiments

### train_gpt_exp001.py — Depth Recurrence
**Approach:** 5 physical transformer layers looped 2x = 10 effective layers
- Saves ~45% of layer parameters through weight sharing
- 3x MLP width (1536 hidden) using saved parameter budget
- Per-iteration learned loop gates
- U-Net skips adapted to virtual layer indices

**Run:**
```bash
RUN_ID=exp001 torchrun --standalone --nproc_per_node=8 train_gpt_exp001.py
```

### train_gpt_exp002.py — Full SOTA Stack (20 techniques)
**Approach:** Replicates and combines all known SOTA techniques
- 11 layers, 3x MLP, LeakyReLU(0.5)^2
- SmearGate + EngramLite (multi-order N-gram hash embeddings)
- XSA on all layers, Partial RoPE (16/64 dims), LN Scale
- Turbo-Muon optimizer (3-step Newton-Schulz with spectral preconditioning)
- Mixed int6(attention)/int7(MLP) STE QAT with late activation
- GPTQ-lite clip search (5 percentiles, per-row MSE selection)
- EMA(0.997) weight averaging
- Sliding window evaluation (stride=64)
- zstd-22 compression, orthogonal init, weight decay 0.04

**Run:**
```bash
RUN_ID=exp002 torchrun --standalone --nproc_per_node=8 train_gpt_exp002.py
```

### train_gpt_exp003.py — Beyond SOTA (23 techniques)
**Approach:** Pushes past SOTA with novel additions on top of EXP-002
- Cross-Layer Attention (CLA2): odd layers share K/V from even layers
- 12 layers (enabled by CLA2 parameter savings)
- Score-First Test-Time Training: SGD on tied embeddings using already-scored tokens

**Run:**
```bash
TTT_ENABLED=1 TTT_LR=0.01 RUN_ID=exp003 \
torchrun --standalone --nproc_per_node=8 train_gpt_exp003.py
```

### train_gpt_exp004.py — Aggressive Int5 + 14 Layers (25 techniques)
**Approach:** Maximum depth via aggressive MLP quantization
- Int5 QAT for MLP weights ([-15,15], 31 levels) — saves ~25% MLP bytes
- 14 layers (enabled by int5 savings + CLA2)
- CLA2 across 7 layer pairs
- All techniques from EXP-003 (TTT, V-GLU, EngramLite, etc.)
- High-risk / high-reward: int5 is aggressive but STE QAT should help

**Run:**
```bash
TTT_ENABLED=1 TTT_LR=0.01 RUN_ID=exp004 \
torchrun --standalone --nproc_per_node=8 train_gpt_exp004.py
```

## CRITICAL: 16MB Size Budget
| Config | Params | Int8+zlib | Int6 packed | Status |
|--------|--------|-----------|-------------|--------|
| 9L 2xMLP | 17.1M | 15.7 MB | 12.8 MB | **int8 fits** |
| 11L 2xMLP | 20.7M | 19.0 MB | 15.5 MB | **int6 only** |
| 11L 3xMLP | 26.5M | 24.3 MB | 19.9 MB | **DOESN'T FIT** |

**exp002-004 use 11L 3xMLP which DOESN'T FIT in 16MB with standard quantization!**
Use train_depth_recurrent.py (11L 2xMLP + depth recurrence + int6 QAT) instead.

## Risk Ladder (CORRECTED)
| Script | Risk | Layers | Quant | Fits 16MB? | Recommendation |
|--------|------|--------|-------|------------|----------------|
| train_depth_recurrent.py | Medium | 11+recur | int6 | YES (15.5MB) | **Best path** |
| exp001 | Low | 10 (looped) | int8 | YES | Backup |
| exp002-004 | N/A | 11, 3xMLP | int6/7 | **NO** | Broken (size bug) |

## Technique Impact Estimates

| Technique | BPB Impact | Source |
|-----------|-----------|--------|
| Sliding window eval (stride=64) | -0.034 | PR #287 |
| 11L + 3x MLP + int6 QAT | -0.060 | Multiple PRs |
| XSA (all layers) | -0.003 | arxiv:2603.09078 |
| EngramLite (N-gram hash) | -0.005 | DeepSeek Engram |
| Turbo-Muon (faster steps) | -0.002 | hal-05390446v1 |
| Partial RoPE (16/64) | -0.002 | PR #287 |
| LN Scale | -0.002 | PR #287 |
| LeakyReLU(0.5)^2 | -0.003 | PR #549 |
| EMA(0.997) | -0.003 | PR #374 |
| Weight Decay 0.04 | -0.002 | Multiple |
| GPTQ-lite clip search | -0.001 | PR #374 |
| Mixed int6/int7 | -0.002 | PR #1089 |
| CLA2 + 12L | -0.003 | arxiv:2405.12981 |
| V-GLU (SiLU on values) | -0.001 | Issue #140 |
| Score-First TTT | -0.020 | PR #549 |
| Int5 MLP + 14L (exp004) | -0.010 | Novel |
| **Total (exp003)** | **~0.143** | |
| **Total (exp004)** | **~0.153** | |
| **Predicted exp003** | **~1.081** | |
| **Predicted exp004** | **~1.071** | |

## Key Research References
- [Parameter Golf GitHub](https://github.com/openai/parameter-golf)
- [Turbo-Muon](https://hal.science/hal-05390446v1)
- [XSA](https://arxiv.org/abs/2603.09078)
- [DeepSeek Engram](https://arxiv.org/abs/2601.07372)
- [Cross-Layer Attention](https://arxiv.org/abs/2405.12981)
- [Modded NanoGPT](https://github.com/KellerJordan/modded-nanogpt)
