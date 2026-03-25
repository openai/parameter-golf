# Record: 11L Parallel Muon + LeakyReLU² MLP3x + Legal Score-First TTT

**3-seed mean val_bpb: 1.1253** (std=0.0002) | **~15 MB** | 8xH100 SXM

## 3-Seed Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | EMA bpb | Quantized bpb | **TTT bpb** |
|------|----------|-------|---------|---------------|-------------|
| 1337 | 91.5ms | 6,556 | 1.1194 | 1.1291 | **1.1255** |
| 42 | 89.2ms | 6,726 | 1.1195 | 1.1278 | **1.1253** |
| 2024 | 89.3ms | 6,722 | 1.1193 | 1.1280 | **1.1251** |
| **Mean** | **90.0ms** | **6,668** | **1.1194** | **1.1283** | **1.1253** |

## Architecture (29.8M parameters)

- 11 transformer layers, dim=512, 8 heads / 4 KV heads (GQA)
- **Parallel Muon** with parameter banking (4 contiguous 3D banks) + batched Newton-Schulz
- MLP 3x expansion (hidden=1536) with **LeakyReLU(0.5)²** activation
- **SmearGate** + **BigramHash(1536, dim=128)**
- **Value Residual (ResFormer)** — cache V from layer 0, blend via learned lambda
- **Gated Attention** — per-head sigmoid gate (nn.Linear, bias init 4.0)
- **XSA on last 4 layers** — exclusive self-attention
- **Partial RoPE** — 16/64 head dimensions
- Tied FP16 embeddings, U-Net skip connections, orthogonal initialization
- Flash Attention 3 for causal attention

## Training

- **Parallel Muon optimizer**: 3-phase async reduce-scatter → Adam → NS5+all-gather
  - lr=0.025, momentum 0.92→0.99/1500 steps, WD=0.04
  - No DDP — manual gradient sync for non-bank params
- Adam for embeddings (lr=0.035) and scalars (lr=0.025)
- Batch 786,432 tokens, seq_len 2048
- EMA (decay=0.997) + SWA (every 50 steps when scale < 0.2)
- Warmdown 3500 iterations (wallclock-based)
- Late QAT via STE (final 15% of wallclock), symmetric [-31, 31] range
- Gradient clipping 0.3
- torch.compile(fullgraph=True) — no DDP wrapper for maximum compilation

## Quantization

- Int6 uniform per-row with GPTQ-lite (5-percentile clip search per row)
- FP16 passthrough for tied embeddings
- zstd-22 compression
- Unbank → quantize → rebank for compatibility with parameter banking

## Legal Score-First TTT (PR #461 / #549 recipe)

Every token scored BEFORE any weight update:

```
for each 32K-token chunk:
    Phase 1 — SCORE: sliding window eval (inference_mode, stride=64)
    Phase 2 — TRAIN: SGD(lr=0.002, momentum=0.9), 3 epochs, all blocks unfrozen, cosine LR
```

TTT improves quantized BPB by ~0.003 (1.1283 → 1.1253).

## Credits

- Parallel Muon / Parameter Banking: PR #399 by @abaybektursun
- LeakyReLU²: PR #493 by @parinzee, PR #518 by @sofiabod
- TTT recipe: PR #461 by @Christopher-Lee-McClendon (adapted: freeze=0)
- Base model stack: PR #414 by @signalrush
