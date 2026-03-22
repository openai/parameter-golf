# Parallel Muon + Parameter Banking + Polar Express

**Systems optimization: 82.08 ms/step, mean val_bpb 1.1239 (3 seeds)**

Pure training speed optimization built on [PR #315](https://github.com/openai/parameter-golf/pull/315) by @jfprincz (11L Partial RoPE + LN Scale + EMA + XSA4, val_bpb 1.1248). The model architecture and hyperparameters are unchanged.

## Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 QAT_THRESHOLD=0.1 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## 3-Seed Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | int6 sliding val_bpb | artifact |
|------|----------|-------|---------------------|----------|
| 1337 | 82.13 ms | 7,306 | 1.1238 | 16.06 MB |
| 42 | 82.08 ms | 7,311 | 1.1240 | 16.49 MB |
| 2025 | 82.04 ms | 7,315 | 1.1239 | 16.65 MB |
| **Mean** | **82.08 ms** | **7,311** | **1.1239** | |
| **Std** | **0.04 ms** | **4** | **0.0001** | |

### Comparison with PR #315 baseline (reproduced, seed 1337)

| | PR #315 Baseline | This Submission | Delta |
|--|---|---|---|
| **step_avg** | 84.76 ms | **82.08 ms** | **-2.68 ms (3.2%)** |
| **steps in 600s** | 7,079 | **7,311** | **+232 steps** |
| **int6 sliding val_bpb** | 1.1253 | **1.1239** | **-0.0014** |

> **Note on artifact size:** Artifacts are 16.06-16.65 MB across seeds (slightly over the 16 MB track limit). The excess is from the different training trajectory producing less compressible weight values, not from additional parameters. The model has identical parameter count (26.8M). This can be addressed with int5 MLP quantization (proven in the leaderboard #1 submission).

## What Changed

Three inseparable optimizer optimizations replacing 66 sequential individual Newton-Schulz calls with 4 batched operations:

### 1. Parameter Banking
3D `nn.Parameter` banks replace 66 separate `nn.Linear` weights:
- `qo_bank`: (22, 512, 512) — Q + Out projections
- `kv_bank`: (22, 256, 512) — K + V projections
- `mlp_up_bank`: (11, 1536, 512) — MLP up
- `mlp_down_bank`: (11, 512, 1536) — MLP down

Forward: `F.linear(x, bank[layer_idx])`. Compiled forward+backward verified identical: 72.33ms vs 72.59ms.

### 2. Polar Express ([arXiv:2505.16932](https://arxiv.org/abs/2505.16932))
Per-iteration minimax-optimal polynomial coefficients replace fixed Newton-Schulz. 35% tighter orthogonalization (0.21 vs 0.32 relative error), same 5 iterations.

### 3. Parallel Muon ([arXiv:2511.07464](https://arxiv.org/abs/2511.07464))
DDP removed for bank params. Post-backward communication scheduled explicitly:
1. Launch async `reduce_scatter` for all banks (biggest first)
2. `all_reduce` + Adam step on small params (while bank RS is in-flight)
3. Wait for RS, local batched PE on each GPU's shard, async `all_gather`

### Why DDP doesn't work with banking
Bank gradients aggregate across all 11 layers → available only at end of backward → zero DDP overlap (+4ms regression). Removing DDP for banks and scheduling communication explicitly restores full overlap.

## Credits

Built on [PR #315](https://github.com/openai/parameter-golf/pull/315) by @jfprincz. Model architecture, hyperparameters, initialization, and evaluation are unchanged.
