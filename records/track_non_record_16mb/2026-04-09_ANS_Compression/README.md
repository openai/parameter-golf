# ANS Weight Compression — Replacing LZMA with Optimal Entropy Coding

**Contribution type:** Compression improvement (orthogonal to architecture/training)
**Author:** OE-GOD (Aung Maw)

## TL;DR

Every leaderboard entry uses LZMA/zstd to compress int6-quantized weights. These are general-purpose byte compressors that don't exploit the known structure of quantized weight distributions. Replacing them with **rANS (range Asymmetric Numeral Systems)** using per-layer histograms achieves near-optimal compression — **saving 1.6 MB (13.9%) losslessly**.

At int6, 1.6 MB = **2.2 million extra parameters** that fit in the same 16 MB budget.

## The Problem

Current pipeline:
```
weights → int6 quantize → pack into bytes → LZMA → artifact
                                              ↑
                                    7.6 bits/byte on packed stream
                                    but entropy is only 4.82 bits/symbol
                                    waste: 0.78 bits/param = 1.6 MB
```

LZMA operates on **bytes**, but int6 values span byte boundaries when packed. LZMA can't see the symbol structure. It's a generic compressor being applied to structured data.

## The Solution

```
weights → int6 quantize → rANS encode with per-layer histogram → artifact
                                              ↑
                                    4.82 bits/symbol (near-optimal)
                                    only 11 KB above theoretical entropy
```

rANS with the exact frequency table of each layer's int6 values encodes each symbol in exactly `-log2(frequency/total)` bits — the information-theoretic optimum.

## Results

Tested on a trained 9-layer baseline model (17M params):

| Method | Size | Bits/param | vs Theoretical |
|--------|------|------------|----------------|
| Theoretical minimum | 9.80 MB | 4.82 | — |
| **ANS (this work)** | **9.81 MB** | **4.82** | **+11 KB** |
| LZMA (current) | 11.40 MB | 5.60 | +1,638 KB |

**ANS is within 11 KB of the theoretical entropy minimum. LZMA wastes 1,638 KB.**

## What 1.6 MB Buys

At int6 quantization, 1.6 MB recovered = 2.2 million extra parameters. Concretely:

| Use recovered space for | Expected impact |
|------------------------|-----------------|
| Extra transformer layer (11→12) | More depth, ~0.01-0.02 BPB |
| MLP 3×→3.5× expansion | More capacity per layer |
| int6→int8 on top-3 sensitive layers | Better precision where it matters |
| Wider BigramHash (3072→4096) | Richer token representations |

## Methodology: How We Found This

We systematically tested where the 16 MB budget is wasted:

1. **Layer delta encoding** — Hypothesis: adjacent layers are similar, store diffs cheaply.
   Result: **Rejected.** Delta/weight ratio = 1.3 (layers are unique, not redundant). ✗

2. **Embedding factorization** — Hypothesis: embedding table is low-rank, factorize with SVD.
   Result: **Rejected.** Embedding is only 1.6% of model and high-rank. ✗

3. **Spatial correlation** — Hypothesis: adjacent weights in a row are correlated (like pixels in images).
   Result: **Rejected.** Residual entropy is 11.4% *higher* than direct entropy. Weights are not spatially correlated. ✗

4. **LZMA vs optimal coding** — Hypothesis: LZMA wastes bits on structured quantized data.
   Result: **Confirmed.** 1.6 MB gap between LZMA and theoretical entropy. ✓

Each rejected hypothesis narrowed the search. The compression gap was the real waste.

## Implementation

`ans_compress.py` — standalone tool, no dependencies beyond numpy.

```bash
# Analyze savings on any trained model
python ans_compress.py --input model.npz --analyze --bits 6

# Compress
python ans_compress.py --input model.npz --output model.ans --bits 6

# Decompress and verify lossless roundtrip
python ans_compress.py --decompress --input model.ans --output restored.npz --verify
```

The rANS implementation is ~200 lines of pure Python. Per-layer frequency tables add 64×2 = 128 bytes overhead per layer (negligible). Encode/decode is fast enough for the 10-minute training window.

## Integration Path

To integrate with the current #1 submission (PR #1019):

1. After training + GPTQ quantization, replace the LZMA serialization step with `ans_compress.compress_model()`
2. At load time, replace LZMA deserialization with `ans_compress.decompress_model()`
3. Use the freed 1.6 MB for a wider model (retrain with larger MLP or extra layer)
4. Measure BPB

This is orthogonal to all architecture and training improvements — it can be stacked on top of any submission.

## Files

- `ans_compress.py` — rANS encoder/decoder with quantization, analysis, and verification
- `delta_compress.py` — Layer similarity analysis tool (used to reject delta encoding hypothesis)
- This README

## Limitations

- Pure Python rANS — could be 10-100x faster with C/Rust implementation
- Per-layer frequency tables assume i.i.d. weights within a layer (validated: no spatial correlation)
- Overhead of ~128 bytes per layer for frequency tables (negligible at 16 MB scale)
- Not tested on the #1 submission's model yet (need 8×H100 to train it)
