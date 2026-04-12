# 11L 4xMLP RecycledCore ParallelResid Int5

**val_bpb: 1.1464** (single seed 1337, int5 sliding-window) | **15.93 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM)

| Seed | step_avg | steps | pre-quant bpb | int5 roundtrip bpb | **int5 sliding bpb** | Artifact |
|------|----------|-------|---------------|--------------------|--------------------|----------|
| 1337 | 110.14ms | 5,449 | 1.1329 | 1.1699 | **1.1464** | 15,925,822 |

Single-seed run. Seeds 42 and 2025 not yet completed.

## Key Innovation

PR #1493 alignment changes applied to existing recycled-core architecture:

1. **4x MLP expansion** (512->2048->512) with 11 layers, up from 3x/12L — wider MLPs trade depth for capacity per layer
2. **3-layer recycled-core** (layers 3,4,5 replayed once) giving 14 virtual layers from 11 physical, with delayed activation at 35% of wallclock to avoid destabilizing early training
3. **Parallel residuals** (GPT-J style) on last 5 layers — attention and MLP read from same normalized input, outputs summed
4. **QK-Gain 5.25** — higher initial gain for query-key normalization (up from 1.5)
5. **GPTQ SDClip** quantization (k*std per-row clipping) replacing percentile search
6. **72% fractional warmdown** and **WD=0.095** (up from 0.04)

## Training Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 physical, 14 virtual (recycled-core: layers 3,4,5 x2) |
| Dim / Heads | 512d, 8 heads, 4 KV heads (GQA), head_dim=64 |
| MLP | 4x expansion (512->2048->512), LeakyReLU(0.5)^2 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| Parallel Residuals | GPT-J style, last 5 layers |
| QK Norm | RMSNorm + learnable Q gain (init=5.25) |
| LN Scale | 1/sqrt(layer+1) |
| U-Net skips | Learned skip weights |
| Gated attention | Per-head sigmoid gate |
| Value residual | First-layer value blending |
| BigramHash | 2048 buckets, dim=128, scale=0.05 |
| SmearGate | Per-dim sigmoid gate |
| Value Embeddings | dim=128, layers 9,10 |
| Weight avg | EMA(0.9965) + SWA(every 50, 13 snapshots) |
| Quantization | GPTQ SDClip int5 mlp+attn (k=8.0) + LZMA preset=9 |
| Optimizer | Muon (matrices, lr=0.025, WD=0.095) + Adam (embeddings/scalars) |
| Warmdown | 72% of wallclock (fractional) |
| Compile | max-autotune, cudagraph_trees=False |
| Embeddings | Tied |
| Logit softcap | 30.0 |
| Context | train=2048, eval sliding-window stride=64 |

## Run Command

```bash
# From repo root (all params are defaults at this commit):
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Environment variable overrides available: `NUM_LAYERS`, `MLP_MULT`, `QK_GAIN_INIT`, `WARMDOWN_FRAC`, `MUON_WD`, `ADAM_WD`, `EMA_DECAY`, `PARALLEL_RESIDUAL_LAST_N`, `RECYCLE_LAYERS`, `RECYCLE_WARMUP_FRAC`, etc.

## Ablation

Compared to the previous best result on this codebase (11L int6, `457c0cf`, sliding BPB 1.1223):

| Change | Metric | Value | Notes |
|--------|--------|-------|-------|
| Previous best (11L int6 `457c0cf`) | sliding BPB | 1.1223 | Int6 quantization |
| This run (11L 4xMLP int5) | pre-quant BPB | 1.1329 | Before quantization |
| This run (11L 4xMLP int5) | int5 roundtrip BPB | 1.1699 | +0.0370 quant degradation |
| This run (11L 4xMLP int5) | **int5 sliding BPB** | **1.1464** | +0.0241 vs previous best |

The regression vs previous best is primarily due to **int5 quantization** (+0.024 BPB degradation vs int6). The pre-quant BPB (1.1329) suggests the architectural changes (4x MLP, parallel residuals, QK-Gain 5.25, delayed recurrence) are roughly neutral to slightly positive, but int5 quantization erases the gains.

Key gap vs leaderboard best (PR #1493, 1.0810 BPB): ~0.065 BPB, dominated by SP8192 tokenizer (~-0.02 to -0.04), int5 vs int6 quant (~+0.024), and simpler recurrence (14 vs 17 virtual layers).

## Credits

- **PR #1493 architecture** (SP8192 + 3L recurrence + parallel residuals + QK-Gain 5.25): target for alignment
- **Recycled-core depth**: hourglass-inspired weight sharing
- **LeakyReLU(0.5)^2**: [PR #493](https://github.com/openai/parameter-golf/pull/493) by @parinzee
- **Parameter Banking + Parallel Muon**: [PR #399](https://github.com/openai/parameter-golf/pull/399) by @abaybektursun
- **Base model stack (XSA, EMA, BigramHash, etc.)**: [PR #414](https://github.com/openai/parameter-golf/pull/414) by @signalrush
- **Legal TTT recipe**: [PR #461](https://github.com/openai/parameter-golf/pull/461) by @Christopher-Lee-McClendon (not enabled in this run)
