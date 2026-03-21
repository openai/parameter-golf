# Experiment 032: PR#42 Config (baseline relu² + LR=0.06 + warmdown=3600 + fp16 embed)

## Status: COMPLETE (byte grouping bug crashed before sliding window eval)

## Results:
- step 500 val_bpb = **1.4529** (baseline 1.4805 = 0.028 better!)
- step 1000 val_bpb = 1.3620
- step 1500 val_bpb = 1.3204
- step 2000 val_bpb = **1.2954** (pre-quant)
- Step avg: 460ms (baseline relu² speed)
- Artifact: 13,790,468 bytes (13.8MB) — fits 16MB ✅

## Config:
- Baseline relu² MLP (no SwiGLU)
- MATRIX_LR=0.06, SCALAR_LR=0.06, TIED_EMBED_LR=0.03
- WARMDOWN_ITERS=3600 (longer warmdown)
- FP16 embedding passthrough (tok_emb.weight kept in fp16)
- LAWA disabled
- Byte grouping enabled (saved 78KB)

## Key Finding:
PR#42's config works. Higher LR (0.06 vs 0.04) + longer warmdown (3600 vs 1200)
gives strong results at baseline relu² speed. At step 500, only 0.003 BPB worse
than SwiGLU+lowLR (1.4529 vs 1.4500) but 10% faster per step.

## Crashed: byte grouping decompression off-by-one bug (fixed in later experiments)
