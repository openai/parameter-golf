# MoR SOTA: Mixture of Recurrence on Proven SOTA Base

**Date:** 2026-03-31
**Target:** Beat SOTA 1.1194 BPB (2026-03-23_LeakyReLU_LegalTTT_ParallelMuon)

## Architecture

Built directly on the proven SOTA technique stack with **Mixture of Recurrence (MoR)** weight sharing as the sole novel contribution.

### Proven SOTA Stack (unchanged)
- **GQA** (8Q/4KV), Partial RoPE (16 dims), XSA on last N blocks, FA3
- **LeakyReLU(0.5)²** MLP activation (3× expansion)
- **SmearGate** (1-token causal blend) + **BigramHash** embedding
- **LN Scale** (1/√layer+1 per virtual layer)
- **EMA(0.997)** + tight **SWA** (every 50 steps when scale<0.2)
- **Legal Score-First TTT** (inference_mode scoring before SGD)
- **Sliding window eval** (stride=64, seq=2048)
- **int6 per-row + lzma** compression (~0.726 bytes/param)
- **Parallel Muon** with 3D parameter banks (batched Newton-Schulz, `fullgraph=True`)

### Novel Contribution: MoR Weight Sharing

**K=7 unique HELIXBlocks × R=2 iterations = 14 virtual layers**

- Parameter banks sized `[K]` instead of `[L]`: `qo_bank[2K]`, `kv_bank[2K]`, `mlp_up_bank[K]`, `mlp_down_bank[K]`
- **U-Net skip connections**: encoder (r=0) pushes K tensors; decoder (r=1) injects in reverse order
- **Per-virtual-layer scalars** for decoder iterations: `v_attn_scale[K×(R-1)]`, `v_mlp_scale`, `v_resid_mix` (routed to scalar AdamW)
- **Virtual LN scale factors**: `1/√(vi+1)` for each of the K×R virtual layer positions

## Configuration

```bash
NUM_UNIQUE_BLOCKS=7     # K: unique weight sets
NUM_ITERATIONS=2        # R: iterations → 14 virtual layers
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=3
ROPE_DIMS=16
XSA_LAST_N=2
TRAIN_SEQ_LEN=2048
BIGRAM_VOCAB_SIZE=30000
BIGRAM_DIM=128
```

## Parameter Budget

- K=7 unique blocks with 3D banks → ~18M total params
- int6+lzma: ~18M × 0.726 bytes/param ≈ ~13MB (well under 16MB limit)

## Training Command

```bash
RUN_ID=mor_sota_v1 \
NUM_UNIQUE_BLOCKS=7 \
NUM_ITERATIONS=2 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Files

- `train_gpt.py`: 1918 lines — MoR SOTA implementation with full optimizer routing, serialization, and eval
- `test_smoke.py`: CPU smoke test (forward, backward, optimizer, inference, quantization roundtrip)
- `submission.json`: Metadata (val_bpb to be updated after training)
- `README.md`: This file

## Design Rationale

### Why MoR over more unique layers?
- **Parameter efficiency**: 7 unique blocks × 2 iterations = 14 virtual layers, but only 7 unique weight sets to quantize and compress
- **int6 budget**: Fewer unique weights → lower quantization error → better roundtrip BPB
- **Proven base**: All SOTA techniques preserved; MoR is the single independent variable

### Why U-Net skip connections?
- **Encoder/decoder asymmetry**: First iteration builds representations; second refines them with global skip context
- **Gradient flow**: Skip connections provide shorter gradient paths through the shared weights
- **Empirically proven**: Standard U-Net structure has strong prior in SOTA vision/language models

### Why per-virtual-layer scalars?
- **Expressiveness**: Shared weights would produce identical transformations; per-layer scalars let each virtual layer specialize
- **Low cost**: K×(R-1) = 7 additional scale/mix vectors per type — negligible parameter overhead
- **Optimizer routing**: Correctly classified as scalar AdamW (not Muon)

## Smoke Test

```bash
conda run -n ai_env python test_smoke.py
# PASS: Smoke test OK  params=155,405  loss=6.2362  K=3  R=2  virtual_layers=6
```

## Next Steps

1. Run 8×H100 training (target 10 min)
2. Evaluate on FineWeb val (compute BPB)
3. Update `submission.json` with final metrics
4. Submit if BPB < 1.1188 (beats SOTA by ≥0.001 nats)
