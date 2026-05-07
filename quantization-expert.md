# Quantization Expert Skill

When implementing quantization:
1. Training-time fake quantization MUST use identical formula to export-time real quantization
2. For STE: gradient = 1 for inputs within clipping range, 0 outside (not the rounded value's gradient)
3. For ternary weights: store int2 packed weights + float16 per-row alpha scales separately
4. Always verify roundtrip: load quantized weights → run eval → bpb should be within 0.005 of pre-quant
5. Enable QAT only in final 15% of training (Late QAT) to avoid disrupting Muon momentum
6. Use zstd-22 compression on quantized weights — test compressed size before submitting
