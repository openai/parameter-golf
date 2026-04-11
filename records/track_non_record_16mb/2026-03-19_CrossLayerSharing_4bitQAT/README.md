# Cross-Layer Parameter Sharing + 4-bit QAT (RecurrentGPT)

## Approach

This submission introduces two information-theoretically motivated techniques to maximize effective model capacity within the 16MB artifact budget:

### 1. Cross-Layer Parameter Sharing (ALBERT-style Depth Recurrence)

Instead of N unique transformer blocks, we use a **prelude → recurrent → coda** architecture:
- **2 unique prelude blocks** for input specialization
- **1 shared recurrent block** iterated 10 times with per-iteration learned gates
- **2 unique coda blocks** for output specialization
- **Effective depth: 14 layers** from only 5 unique block parameter sets

Per-iteration `iter_gate` parameters (10 × d_model) allow the shared block to differentiate its behavior across depths. U-Net skip connections are preserved from the baseline.

### 2. 4-bit Quantization-Aware Training (QAT)

Straight-through estimator (STE) based fake quantization simulates 4-bit precision (16 levels) during training. This lets the model learn robustness to aggressive post-training quantization:
- 4-bit weights stored as int8 with values in [-8, 7]
- Only 16 distinct values per row → excellent zlib compressibility
- QAT disabled for first 500 steps to let weights settle

### Combined Effect

The parameter sharing reduces unique parameters from ~28M (SOTA) to ~13.8M, while 4-bit quantization compresses these to a **7.25MB artifact** (vs 15.4MB SOTA). This leaves ~8.75MB of headroom for wider models (d=640 vs d=512) or more recurrent iterations.

## Architecture Details

| Parameter | Value |
|-----------|-------|
| model_dim | 640 |
| num_heads | 10 (head_dim=64) |
| num_kv_heads | 2 (aggressive GQA) |
| num_prelude | 2 |
| num_recurrent_iters | 10 |
| num_coda | 2 |
| effective_depth | 14 |
| unique_params | 13,786,290 |
| artifact_bytes | 7,193,920 |

## Results

| Seed | val_loss | val_bpb | Steps | ms/step |
|------|----------|---------|-------|---------|
| 1337 | 3.6447 | 2.1586 | 300 | 2001 |

**Note:** This run was on 2× GPUs in eager mode (no `torch.compile`) due to Triton SMEM constraints with the recurrent architecture. Only 300 steps were completed in the 10-minute wallclock — far too few for convergence. The approach needs 8×H100 with `torch.compile` (compiling individual blocks rather than the full model) to achieve competitive BPB.

## Key Innovations Over Baseline

1. **RecurrentGPT architecture** with configurable prelude/recurrent/coda structure
2. **Per-iteration gating** for depth-dependent behavior without per-layer parameters
3. **4-bit QAT with STE** integrated into `CastedLinear` for seamless training
4. **Unified quantization pipeline** supporting both int8 and 4-bit modes
5. **Phase-transition residual mixing** applied across effective depth (including recurrent iterations)
6. All SOTA innovations preserved: FP16 tied embedding, Muon WD, overtone init, sliding window eval
