# Hybrid Depth-Recurrent Transformer

**Preliminary val_bpb: 1.3323** (2×H100, single seed — 8×H100 run pending)

## Novel Architecture

Standard depth-recurrence shares all transformer weights across loop iterations. The problem: int8 quantization errors compound through the shared weights on every loop, causing catastrophic quality loss (0.40 BPB gap in our tests).

**Hybrid solution:** Keep precision-sensitive layers near input/output as unique (non-shared) weights. Only the bulk middle layers are shared and looped.

- **1 unique entry layer** — protects embedding→hidden transition from quantization compounding
- **4 shared blocks × 5 loops** — cheap depth, 20 effective middle layers from 4 weight blocks
- **1 unique exit layer** — protects hidden→logit transition
- **22 total effective layers** from only **6 unique weight blocks**
- **U-Net skip connections** across the full effective depth
- **Per-virtual-layer scalars** (attn_scale, mlp_scale, resid_mix, q_gain) give each loop its own behavior

## Additional Techniques

1. **FP16 tied embedding passthrough**: `tok_emb.weight` kept in fp16 during int8 quantization
2. **Sliding window evaluation** (stride=64, seq_len=1024)
3. **Decoupled Muon weight decay** (0.02)
4. **Overtone spectral embedding init**: SVD power-law spectrum shaping (`S_k ~ k^{-0.5}`)
5. **Phase-transition residual mixing**: Sigmoid-scheduled `resid_mix` initialization
6. **Tuned learning rates**: matrix_lr=0.03, scalar_lr=0.03, tied_embed_lr=0.04

## Results (Preliminary — 2×H100)

| Seed | val_loss | val_bpb | Steps | ms/step | GPUs |
|------|----------|---------|-------|---------|------|
| 1337 | 2.2496 | 1.3323 | 954 | 631.86 | 2 |

Artifact: ~14.2 MB | Quantization gap: -0.004 BPB (near-zero)

## Key Finding: Quantization Gap

| Architecture | Pre-quant BPB | Post-quant BPB | Gap |
|---|---|---|---|
| Pure depth-recurrence | 1.6542 | 2.0494 | **+0.3952** |
| **Hybrid (this submission)** | 1.3737 | 1.3701 | **-0.0036** |

The hybrid architecture essentially eliminates quantization degradation.

## Environment Variables for Reproduction

```bash
WARMDOWN_ITERS=2500 MATRIX_LR=0.03 SCALAR_LR=0.03 TIED_EMBED_LR=0.04 torchrun --nproc_per_node=8 train_gpt.py
```
