# Non-Record: 11L Parallel Muon + LN Scale + LeakyReLU² MLP3x + Legal TTT

**3-seed mean val_bpb: 1.1215** (std=0.0002) | **~15.85 MB** | 8xH100 SXM

## 3-Seed Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | EMA bpb | Quantized bpb | **TTT bpb** |
|------|----------|-------|---------|---------------|-------------|
| 1337 | 88.8ms | 6,759 | 1.1161 | 1.1238 | **1.1217** |
| 42 | 88.8ms | 6,757 | 1.1158 | 1.1234 | **1.1213** |
| 2024 | 88.9ms | 6,752 | 1.1160 | 1.1234 | **1.1215** |
| **Mean** | **88.8ms** | **6,756** | **1.1160** | **1.1235** | **1.1215** |

## Architecture (26.8M parameters)

- 11 transformer layers, dim=512, 8 heads / 4 KV heads (GQA)
- **Parallel Muon** with parameter banking (4 contiguous 3D banks) + batched Newton-Schulz
- MLP 3x expansion (hidden=1536) with **LeakyReLU(0.5)²** activation
- **LN Scale** — depth-dependent normalization: 1/sqrt(layer_idx+1)
- **SmearGate** + **BigramHash(1536, dim=128)**
- **Value Residual (ResFormer)** — cache V from layer 0, blend via learned lambda
- **Gated Attention** — per-head sigmoid gate (nn.Linear, bias init 4.0)
- **XSA on last 4 layers** — exclusive self-attention
- **Partial RoPE** — 16/64 head dimensions
- Tied FP16 embeddings, U-Net skip connections, orthogonal initialization
- Flash Attention 3 for causal attention

## Training

- **Parallel Muon optimizer**: 3-phase async reduce-scatter -> Adam -> NS5+all-gather
  - lr=0.025, momentum 0.92->0.99/1500 steps, WD=0.04
  - No DDP -- manual gradient sync for non-bank params
- Adam for embeddings (lr=0.035) and scalars (lr=0.025)
- Batch 786,432 tokens, seq_len 2048
- EMA (decay=0.997) + SWA (every 50 steps when scale < 0.2)
- Warmdown 3500 iterations (wallclock-based)
- Late QAT via STE (final 15% of wallclock)
- Gradient clipping 0.3
- torch.compile(fullgraph=True)

## Quantization

- Int6 uniform per-row with GPTQ-lite (5-percentile clip search per row)
- FP16 passthrough for tied embeddings
- zstd-22 compression
- Unbank -> quantize -> rebank for compatibility with parameter banking

## Legal Score-First TTT (PR #461 / #549 recipe)

Every token scored BEFORE any weight update:

```
for each 32K-token chunk:
    Phase 1 -- SCORE: sliding window eval (inference_mode, stride=64)
    Phase 2 -- TRAIN: SGD(lr=0.002, momentum=0.9), 3 epochs, all blocks unfrozen, cosine LR
```

TTT improves quantized BPB by ~0.002 (1.1235 -> 1.1215).

## Credits

- Parallel Muon / Parameter Banking: PR #399 by @abaybektursun
- LeakyReLU²: PR #493 by @parinzee, PR #518 by @sofiabod
- LN Scale: PR #315/374 by @jfprincz
- TTT recipe: PR #461 by @Christopher-Lee-McClendon (adapted: freeze=0)
- Base model stack: PR #414 by @signalrush
