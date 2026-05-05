# Compression Benchmark Results (on 068 checkpoint, 21.78M params)

## Summary
FLAT+zstd wins at 14.90MB. All schemes fit under 16MB for this model size.

## Results
| Scheme | Size | vs best |
|--------|------|---------|
| **FLAT + zstd** | **14.90MB** | baseline |
| uint8 scales + FLAT + zstd | 14.98MB | +80KB |
| FLAT + LZMA | 15.06MB | +160KB |
| Blockwise64 + FLAT + zstd | 15.48MB | +580KB (worse!) |
| torch.save + zstd | 15.65MB | +750KB |

## Key Findings
1. FLAT+zstd is definitively the best compression format
2. Blockwise quantization HURTS — more scales add more bytes than they save in tighter quant
3. uint8 scales are slightly worse than fp16 — zstd already compresses fp16 scales well
4. With 1.1MB headroom, BigramHash (~400KB compressed) could be added back

## Implication
**Can run FULL config (MLP=1536 + BigramHash) with FLAT+zstd = ~15.3MB ✅**
This would recover the 0.001 BPB from BigramHash without going over budget.

## Ruled out (don't help)
- Blockwise quantization: more overhead than savings
- uint8 scales: marginal, not worth the complexity
- Outlier splitting: already tested in exp065/066, hurts compression
- Bit-packing: already tested in exp059, hurts compression
