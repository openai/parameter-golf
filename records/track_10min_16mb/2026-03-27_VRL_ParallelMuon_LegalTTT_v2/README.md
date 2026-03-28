# 11L VRL + BigramHash3072 + Parallel Muon + Legal SGD TTT v2

**val_bpb = 1.1264** (mean of 3 seeds, post int6+zstd quantization + legal TTT) | **15.83 MB** | 8×H100 SXM

## 3-Seed Results

| Seed | Steps | Pre-quant bpb | Post-quant bpb | Post-TTT bpb | Artifact bytes | Valid |
|------|-------|---------------|----------------|--------------|----------------|-------|
| 1337 | 6170 | 1.1443 | 1.1524 | 1.1268 | 15,828,109 | Yes |
| 42 | 6177 | 1.1428 | 1.1506 | 1.1253 | 15,828,109 | Yes |
| 45 | 6182 | 1.1435 | 1.1528 | 1.1270 | 15,813,731 | Yes |

Mean post-TTT bpb: **1.1264**

## Changes from PR #549 (SOTA 1.1194)

### Added
1. **VRL (Value Residual Learning)** — Layer 0's V output blended into all subsequent layers via learned per-layer sigmoid gates (arxiv:2410.17897).
2. **BigramHash 3072** — Doubled from 1536 (+free -0.0009 bpb per PR #549 ablation).
3. **Tight SWA over EMA** — When SWA snapshots exist, use snapshot average.
4. **zstd-22 compression** — Replacing lzma preset 6. Better compression ratio.
5. **FA3/FA2/SDPA fallback chain** — Graceful degradation across PyTorch versions.
6. **Sliding window eval bug fix** — Full-length windows only, fixed scoring offset.

### Changed
- `TTT_FREEZE_BLOCKS` default 2 → 0 (all blocks unfrozen, per SOTA ablation)
- `TTT_ENABLED` default 0 → 1 (always run TTT for competitive submissions)

### Removed
- **Full GPTQ** dropped (Hessian calibration added 30-60s overhead without meaningful compression improvement; GPTQ-lite 5-percentile clip search retained)

## Architecture

```
Model:      11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
MLP:        3x expansion (1536), LeakyReLU(0.5)²
Embedding:  Tied, 1024 vocab, BigramHash(3072, dim=128)
Attention:  XSA last 4 layers, Partial RoPE (16/64), FA3/FA2/SDPA
Position:   NTK-aware RoPE, train_seq=2048
Norm:       RMSNorm, LN Scale 1/sqrt(L+1)
Skip:       U-Net encoder/decoder, learned skip weights
VRL:        Sigmoid-gated V0 blend on layers 1-10
VE:         Value Embedding (dim=128) on layers 9,10

Optimizer:  Parallel Muon (batched NS5, parameter banking)
            + AdamW for embeddings/scalars
Schedule:   warmdown=3500, grad_clip=0.3, momentum warmup 0.92→0.99/1500
Averaging:  EMA(0.997) → Tight SWA → LAWA fallback
QAT:        Late QAT @ lr_scale < 0.15

Quant:      Int6 GPTQ-lite (5 percentile clip) + zstd-22
TTT:        Legal score-first SGD, lr=0.002, mom=0.9, 3 epochs, chunk=32K
```

## Reproduction

```bash
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-27_VRL_ParallelMuon_LegalTTT_v2/train_gpt.py
```

Multi-seed:
```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=42   torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=45   torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Env var overrides:
- `NO_COMPILE=1` for debugging (~300ms/step instead of ~97ms/step)
- `FULL_GPTQ=1` to re-enable Hessian GPTQ
- `TTT_ENABLED=0` to skip test-time training

## Credits

- **Base architecture**: PR #549 by @abaybektursun (1.1194 SOTA)
- **Parallel Muon**: PR #399 by @abaybektursun
- **Legal TTT**: PR #461 by @Christopher-Lee-McClendon
- **LeakyReLU²**: PR #493 by @parinzee, PR #518 by @sofiabod
- **VRL**: arxiv:2410.17897 by Qin et al.
