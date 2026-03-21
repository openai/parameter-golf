# Experiment 031: Full Pipeline — PARTIAL (byte grouping bug crashed roundtrip)

## Training Results (complete):
- step 500 val_bpb = 1.4633 (baseline 1.4805 = 0.017 better, but exp030 was 1.4512)
- step 1000 val_bpb = 1.3718
- step 1500 val_bpb = 1.3311
- step 2000 val_bpb = **1.3074** (pre-quant, pre-LAWA)
- Step avg: 496ms (SwiGLU + lowLR)

## LAWA Result: **HURT** (1.3074 → 1.3867 = +0.08 worse)
- 21 snapshots averaged (every 100 steps from step 0)
- Warmdown started at step 0 because WARMDOWN_ITERS=3600 > ITERATIONS=2000
- So LAWA averaged snapshots from the entire training, not just warmdown
- This was a bug — LAWA should only snapshot during actual warmdown

## Quantization Results:
- Artifact: 11,002,340 bytes (11.0MB) — well under 16MB with fp16 embed
- Byte grouping saved 73,444 bytes vs standard zlib
- Code: 69,436 bytes
- **Total: 11,071,776 bytes** ✅ fits 16MB easily

## Crashed During:
- Byte grouping decompression had off-by-one bug with odd-length data
- Fixed in train_gpt_swiglu.py (even_bytes = ceil(n/2), odd_bytes = floor(n/2))
- Sliding window eval, NTK RoPE eval never ran

## Key Learnings:
1. LAWA with warmdown > iterations is BAD — averages entire training, not just warmdown
2. FP16 embedding passthrough works and fits in budget (11MB vs 15.8MB baseline)
3. Byte grouping saves ~73KB (0.5% of artifact)
4. Need warmdown_iters < iterations for 2K screening runs
