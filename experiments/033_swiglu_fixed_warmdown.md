# Experiment 033: SwiGLU + LowLR + Fixed Warmdown + FP16 Embed + LAWA

## Status: COMPLETE (byte grouping bug crashed before sliding window eval)

## Results:
- step 500 val_bpb = **1.4500** (best step-500 result!)
- step 1000 val_bpb = 1.3725
- step 1500 val_bpb = 1.3392
- step 2000 val_bpb = 1.3074 (pre-quant, pre-LAWA)
- LAWA pre-quant val_bpb = **1.2928** (LAWA helped! averaged 9 warmdown snapshots)
- Step avg: 470ms (SwiGLU speed)
- Artifact: 13,874,068 bytes (13.9MB) ✅

## Config:
- SwiGLU(h=672) + MATRIX_LR=0.02 + SCALAR_LR=0.02 + TIED_EMBED_LR=0.03
- WARMDOWN_ITERS=400 (fixed for 2K screening — warmdown starts at step 1600)
- FP16 embedding passthrough
- LAWA every 50 steps during warmdown (9 snapshots total)
- Byte grouping enabled

## Key Findings:
1. **LAWA WORKS with correct warmdown**: 1.3074 → 1.2928 = 0.015 BPB improvement!
2. Warmdown must be < iterations for LAWA to work properly
3. SwiGLU + lowLR still gives best per-step quality but slower
4. Byte grouping decompression bug hit again (same off-by-one, old script)
