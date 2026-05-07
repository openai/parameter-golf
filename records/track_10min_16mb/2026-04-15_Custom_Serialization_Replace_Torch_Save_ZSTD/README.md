# Non-record: Custom serialization replacing torch.save + zstd-22

**Focus: Lossless compression improvement, not BPB.**

Based on [2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271](../2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/) (PR by @jfprincz, val_bpb 1.1271). I did not run `train_gpt.py` on 8xH100, so this submission has no `.log` files. Instead, I obtained the model artifact from the base PR and iterated on the compression locally.

Since compression is lossless, this doesn't affect BPB at all — but it's a strictly better technique for trimming artifact size than lossy approaches like pruning. Anyone working on a model that's slightly over the 16MB limit could drop in this custom serialization to "buy back" ~363KB for free.

## What this PR does

Replaces `torch.save` + `zstd-22` with a custom binary format using **Asymmetric Numeral Systems (ANS)** entropy coding. The model architecture, training, and quantization in `train_gpt.py` are identical to the base submission — only the serialization layer changes.

| Method | Compressed bytes | vs Baseline |
|---|---|---|
| **Baseline** (torch.save + zstd-22) | 15,513,031 | — |
| **Custom Serialization** | 15,150,085 | **-362,946 (-2.34%)** |

### Why this works

`torch.save` + generic compressors (`zstd`, `zlib`, `lzma`) treat the serialized blob as an opaque byte stream. Since we know the format, we can do better:

- **Known value alphabets**: Int6 weights use only 64 possible symbols ([-32, 31]), int8 embeddings use 256. ANS encodes directly against the true symbol distribution, reaching within bits of the entropy floor — whereas generic compressors discover this implicitly through LZ77 pattern matching.
- **Row-level distribution structure**: Rows within the same layer type share similar value distributions. K-means clustering (K=16) on row frequency histograms produces shared ANS probability models that adapt to different weight patterns. A generic compressor can't do this because it doesn't know where one row ends and another begins.
- **Dtype-aware stream separation**: Splitting int8, fp16, and fp32 into independent streams and applying dtype-specific transforms (zigzag encoding for signed integers, byte-shuffling to group fp16 exponent bytes) makes each stream more compressible than the interleaved pickle format.
- **No pickle overhead**: `torch.save` uses pickle with per-tensor framing, ZIP containers, and metadata (~100KB). Our format stores a compact LZMA-compressed JSON header (with architecture params from which tensor shapes are derived), followed by length-prefixed compressed streams.

## Methodology: automated experiment loop

This format was developed through **62 sequential experiments** in a dedicated [serialization playground](https://github.com/joyceyan/serialization_playground), each testing a single isolated change:

1. Read prior results and notes
2. Design one change, edit `serialize.py`
3. Run `python test_serialize.py` (roundtrip correctness + size benchmark against the real H100 artifact)
4. Log results to `results.tsv`, update `notes.md` with hypothesis/result/insights
5. Keep if compressed size decreased with zero roundtrip error, otherwise revert
6. Repeat

The full experiment history (62 custom-format experiments + 25 additional torch.save fork experiments) is in the playground repo. I also attempted to fork and alter the C implementation of `torch.save` directly, but the custom binary format proved superior.

This approach is well-suited to automation: the loop is mechanical (edit, measure, keep/revert), the success criterion is objective (fewer bytes, zero error), and each iteration is fast (~2s per encode/decode cycle on the real artifact). In a production environment where even a 1-5% reduction in artifact size matters, it would make sense to have an AI agent generate custom serialization that is aware of the model's specific format and weight distributions.

### What didn't work (notable negatives from 62 experiments)

- Bit-packing / nibble-split: **+12.4%** worse (destroys byte alignment that zstd exploits)
- Delta / prediction filters: **+9.6%** worse (weights are spatially uncorrelated)
- Row interleaving across tensors: breaks within-tensor locality
- LZMA for the int8 stream: **+2%** worse than zstd-22 for large dense data
- Gray code on zigzag: +313 bytes worse

## Architecture (unchanged from base)

11L, 512D, 8H/4KV, MLP 3x, XSA on last 4 layers, EMA (0.997), bigram hash embeddings, int6 per-row quantization (MLP+attention), int8 per-row (embeddings), FlashAttention 3.

## Additional dependencies

```bash
pip install brotli constriction scipy
pip install --no-cache-dir "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
```

- `constriction`: Rust-backed ANS entropy coding library
- `scipy`: K-means clustering for row distribution grouping
- `brotli`: Brotli compression for fp16 stream
- `flash-attn-3`: Flash Attention 3 Hopper kernel (pre-built wheel for CUDA 12.8; adjust URL for your CUDA version)

## Credits

- Base submission: @jfprincz (PR #315 / record 2026-03-20)
