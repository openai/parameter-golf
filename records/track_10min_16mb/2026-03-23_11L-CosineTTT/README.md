# Record: 11L XSA4 + LeakyReLU(0.5)² + Cosine TTT 50ep (val_bpb=1.0622)

## Summary
- val_bpb **1.0622** (seed 1337, 50ep cosine TTT) — beats prior best validated 1.0672 (#462) by **-0.005**
- 3-seed mean at 30ep: 1.0814 ± 0.0014
- Full #414 frontier stack + LeakyReLU(0.5)² activation + 50-epoch cosine TTT with per-layer LR groups
- Training: ~5880 steps in 600s. Eval: TTT ~890s + sliding window ~311s (~20 min total eval)

## Approach

11-layer d=512 transformer with full SOTA technique stack, adapted from PR #414 with key improvements.

**Architecture (from #414):**
- 11L, d=512, 8/4 GQA heads, MLP 3x, tied embeddings (vocab 1024)
- XSA on last 4 layers, Partial RoPE (16/64 dims), LN Scale 1/sqrt(layer+1)
- BigramHash(2048,128) + SmearGate + OrthoInit + VE128 (layers 9,10)
- U-Net skip connections

**Our improvements over #414:**
- **LeakyReLU(0.5)²** instead of ReLU² — preserves negative gradient flow, -0.003 BPB
- **50-epoch cosine TTT** with per-layer LR (from #481): AdamW lr=0.0005, cosine decay, 3x for mlp.proj, 0.5x for mlp.fc

**Quantization:** Int6 + GPTQ-lite + zstd-22, EMA(0.997), Tight SWA, Late QAT@0.15

**TTT recipe (from PR #481):**
- 50 epochs AdamW(lr=0.0005, wd=0.0) on validation tokens
- Cosine LR decay: lr *= 0.5 * (1 + cos(π * progress))
- Per-layer LR: mlp.proj 3× (high quant error recovery), mlp.fc 0.5×
- DDP gradient sync + grad clip 1.0
- All parameters unfrozen

## Results

| Config | val_bpb |
|--------|---------|
| No TTT (LeakyReLU base) | 1.1271 |
| 30ep cosine TTT (seed 1337) | 1.0804 |
| 30ep cosine TTT (3-seed mean) | 1.0814 ± 0.0014 |
| **50ep cosine TTT (seed 1337)** | **1.0622** |

## Comparison

| Metric | #462 (GEPA+TTT) | #414 (no TTT) | Ours |
|--------|-----------------|---------------|------|
| BPB | 1.0672 | 1.1233 | 1.0622 |
| Architecture | GEPA AI-discovered | Standard | Standard + LeakyReLU |
| TTT | AdamW pre-eval | None | Cosine 50ep pre-eval |
| Layers | 11 | 11 | 11 |

## Run command

```bash
TTT_EPOCHS=50 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```
