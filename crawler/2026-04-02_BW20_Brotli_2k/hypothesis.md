# BW20_Brotli_2k — Hypothesis

## ONE variable changed
Compression backend: zstd (level 22) → brotli (quality 11)

## Parent
BW5 (champion): `legs/2026-03-29_BW5/train_gpt.py`

## What changed
- `import zstandard` → `import brotli`
- Compress: `zstandard.ZstdCompressor(level=22).compress(...)` → `brotli.compress(..., quality=11)`
- Decompress: `zstandard.ZstdDecompressor().decompress(...)` → `brotli.decompress(...)`

## Why
Brotli uses a larger context window and better entropy coding than zstd for
static blobs. Quantized weight tensors are a single-shot compression target
(no streaming needed), which is brotli's sweet spot. Even a modest improvement
in compression ratio directly reduces artifact size, freeing headroom for
larger models or hitting the 16MB cap more comfortably.

## What we expect
- Smaller artifact size (brotli typically beats zstd by 5-15% on dense blobs)
- Identical BPB (compression is post-training — model weights unchanged)
- Slightly slower compress/decompress (acceptable — this is a one-shot eval step)

## Gate target
- No blowups (training completes, roundtrip eval matches)
- Artifact size reduction vs zstd baseline
- BPB within noise of BW5 (identical model, only compression differs)

## Gate
1k steps, 1-GPU, seed=444. Quick signal check — compression only changes post-training serialization.
