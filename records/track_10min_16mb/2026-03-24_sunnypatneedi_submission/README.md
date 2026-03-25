# LeakyReLU(0.5)^2 + AdamW TTT (30ep cosine + per-layer LR) + XSA + Int6

**val_bpb: FILL_BPB** (3-seed mean) | **FILL_MB MB** artifact | 8xH100 SXM, 600s train + ~585s eval

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | Steps | Pre-TTT BPB | Post-TTT BPB (s64) | Artifact |
|------|-------|-------------|---------------------|----------|
| 42   | FILL  | FILL        | FILL                | FILL     |
| 1337 | FILL  | FILL        | FILL                | FILL     |
| 2024 | FILL  | FILL        | FILL                | FILL     |

**Mean: FILL | Std: FILL**

## Key Innovation: AdamW TTT with cosine + per-layer LR on SOTA base

The merged SOTA (PR #549, 1.1194) uses a weak 3-epoch SGD TTT that gives only -0.0025 bpb. We replace it with PR #481's proven AdamW recipe:

1. **AdamW optimizer** (weight_decay=0) instead of SGD with momentum
2. **30 epochs** with **cosine LR decay** instead of 3 epochs flat
3. **Per-layer LR groups**: MLP output projections get 3x base LR (more quant-damaged), MLP input projections get 0.5x, everything else 1x
4. **All blocks unfrozen** (freeze_blocks=0)

PR #481 demonstrated this recipe gives -0.066 bpb on their base (1.1577 -> 1.0970). On the stronger PR #549 base (~1.12 pre-TTT), we expect -0.010 to -0.025 bpb.

## Architecture (from PR #549 SOTA)

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV GQA) |
| MLP | 3x expansion, **LeakyReLU(0.5)^2** |
| BigramHash | 2048 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/sqrt(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + SWA(every 50) |
| Quantization | GPTQ-lite int6 + zstd-22 |

## TTT Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (weight_decay=0) |
| Base LR | 0.0005 |
| Per-layer LR | mlp.proj: 3x, mlp.fc: 0.5x, other: 1x |
| Epochs | 30 |
| Schedule | Cosine decay |
| Freeze blocks | 0 (all unfrozen) |
| Batch seqs | 64 per GPU (512 total) |
| Max steps/epoch | 300 |

## Timing Budget

| Phase | Time |
|-------|------|
| Training | 600s (10 min) |
| Int6 roundtrip eval (diagnostic) | ~20s |
| AdamW TTT (30 epochs) | ~465s |
| Sliding window eval (stride=64) | ~120s |
| **Total eval** | **~605s (within 10 min)** |

## Run Command

```bash
cd /workspace/parameter-golf
SEED=42 XSA_LAST_N=4 TTT_ENABLED=1 TTT_LR=0.0005 TTT_EPOCHS=30 \
TTT_COSINE=1 TTT_PERLAYER=1 TTT_FREEZE_BLOCKS=0 TTT_BATCH_SEQS=64 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-24_sunnypatneedi_submission/train_gpt.py
```

## Provenance

Built on PR #549 (abaybektursun, merged SOTA 1.1194), with TTT recipe from PR #481 (mrdavtan, 1.0970):
- PR #549 / PR #414 (signalrush) - base architecture, int6 GPTQ-lite, EMA/SWA, LeakyReLU
- PR #481 (mrdavtan) - AdamW TTT with cosine decay and per-layer LR
- PR #198 / PR #503 (jfprincz) - XSA (exclusive self-attention)
- PR #287 (jfprincz) - Partial RoPE + LN Scale

## Test Plan

- [ ] 3 seeds run on 8xH100 SXM
- [ ] All 3 seeds train in <=600s
- [ ] All 3 seeds total eval (TTT + sliding) in <=600s
- [ ] All 3 seeds artifact <=16,000,000 bytes
- [ ] Post-TTT sliding BPB beats 1.1194 by >=0.005 nats
- [ ] Statistical significance p<0.01 across 3 seeds
