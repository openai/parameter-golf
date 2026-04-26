# Round 2 Autoresearch Results

**Date**: 2026-03-29 to 2026-03-30
**Hardware**: RTX 3070 Laptop GPU (8GB VRAM), WSL2
**Base config**: 10L/512d/3xMLP/8H/4KV/LeakyReLU(0.5)^2/cosine warmdown

## Results Table

| Experiment | fp32 bpb | int6 bpb | Quant Gap | Artifact | Status |
|------------|----------|----------|-----------|----------|--------|
| base (500 iter) | 1.3987 | 1.4157 | +0.017 | 12.76 MB | Under 16MB |
| exp4: EMA+QAT (500) | 1.3986 | 1.6196 | +0.221 | 10.27 MB | FAILED - EMA catastrophic |
| exp5: SmearGate (500) | 1.4241 | 1.7297 | +0.306 | 11.54 MB | FAILED - needs more steps |
| **exp11: 1000 iter** | **1.3254** | **1.3296** | **+0.004** | **16.11 MB** | OVER 16MB by ~100KB |
| exp12: QAT+1000 iter | 1.3253 | 1.3296 | +0.004 | 16.10 MB | QAT had no effect |

## Key Findings

### 1. More iterations is the single biggest lever
- 500 -> 1000 steps: -0.086 bpb improvement (int6)
- The loss curve still has slope at 1000 steps; 2000+ would help further
- Cost: ~90 min per run on RTX 3070

### 2. EMA without sufficient QAT is catastrophic at any step count
- EMA-averaged weights have distributions that don't survive int6 quantization
- Late QAT (last 44 steps) is insufficient to fix this
- This confirms R1 finding

### 3. SmearGate/BigramHash need >500 steps
- At 500 steps, SmearGate is +0.025 bpb WORSE in fp32
- The bigram embeddings take hundreds of steps to learn useful patterns
- May be worth testing at 2000+ steps

### 4. QAT is unnecessary when quantization gap is tiny
- At 1000 steps, int6 gap is only 0.004 bpb
- Early QAT (800 steps of fake quant) had zero measurable effect
- The weights are naturally quantization-friendly at this scale

### 5. Artifact size is the blocking constraint
- Best model (1000 iter) produces 16.1 MB artifact, 100KB over limit
- Compression: int6+zlib achieves 3.85x ratio consistently
- Need either: zstd-22 (~10% better), 9 layers, or narrower model

## Next Steps (Priority Order)

1. **Fix size**: Try zstd-22 compression (should save ~1.5MB), or reduce to 9L
2. **Try 2000 iterations**: Loss still declining at 1000; 2x more training could give another -0.03 to -0.05 bpb
3. **9L/3xMLP at 1000 iter**: Smaller model guaranteed under 16MB, test bpb tradeoff
4. **SmearGate at 1000+ iter**: May finally pay off with more training
5. **Try 11L with reduced dim (448 or 480)**: More layers with narrower width to fit under 16MB
