#!/usr/bin/env python3
"""
ANS (Asymmetric Numeral Systems) weight compressor for Parameter Golf.

The insight: LZMA operates on bytes but int6 values are packed across byte
boundaries. LZMA can't see the symbol structure. ANS with the exact histogram
of each layer's int6 values encodes each symbol in ~entropy bits — optimal.

Current pipeline:  int6 → pack to bytes → LZMA → 7.6 bits/byte
Our pipeline:      int6 → ANS per-layer  → ~5.3 bits/symbol

Usage:
  # Analyze potential savings
  python ans_compress.py --input logs/model.npz --analyze

  # Compress
  python ans_compress.py --input logs/model.npz --output model.ans

  # Decompress and verify
  python ans_compress.py --decompress --input model.ans --output model_restored.npz --verify
"""

import argparse
import json
import math
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np


# =============================================================================
# Core ANS Implementation (rANS variant)
# =============================================================================
# rANS = range ANS. Simple, fast, optimal compression.
# Each symbol costs exactly -log2(frequency/total) bits.

RANS_PRECISION = 16  # frequency table precision (2^16 = 65536 total)
RANS_TOTAL = 1 << RANS_PRECISION
RANS_LOWER = 1 << 23  # renormalization threshold


def build_frequency_table(symbols: np.ndarray, num_symbols: int) -> list:
    """Build frequency table for rANS. Every symbol gets at least freq=1."""
    counts = Counter(symbols.tolist())
    total = len(symbols)

    # Assign frequencies proportional to counts, minimum 1
    freqs = [0] * num_symbols
    assigned = 0

    for s in range(num_symbols):
        if counts.get(s, 0) > 0:
            # Proportional frequency, minimum 1
            f = max(1, round(counts[s] / total * RANS_TOTAL))
            freqs[s] = f
            assigned += f
        else:
            freqs[s] = 1  # must be >= 1 for ANS to work
            assigned += 1

    # Adjust to sum exactly to RANS_TOTAL
    diff = RANS_TOTAL - assigned
    if diff != 0:
        # Add/remove from the most common symbol
        most_common = max(range(num_symbols), key=lambda s: freqs[s])
        freqs[most_common] += diff
        if freqs[most_common] < 1:
            freqs[most_common] = 1

    # Verify
    assert sum(freqs) == RANS_TOTAL, f"Freq sum {sum(freqs)} != {RANS_TOTAL}"
    assert all(f >= 1 for f in freqs), "Zero frequency found"

    return freqs


def freqs_to_cumfreqs(freqs: list) -> list:
    """Convert frequency table to cumulative frequency table."""
    cum = [0] * (len(freqs) + 1)
    for i in range(len(freqs)):
        cum[i + 1] = cum[i] + freqs[i]
    return cum


def rans_encode(symbols: np.ndarray, freqs: list) -> bytes:
    """rANS encode a sequence of symbols. Returns compressed bytes."""
    cumfreqs = freqs_to_cumfreqs(freqs)
    n = len(symbols)

    # Encode in reverse (rANS requirement)
    state = RANS_LOWER  # initial state
    output = bytearray()

    for i in range(n - 1, -1, -1):
        s = int(symbols[i])
        freq = freqs[s]
        start = cumfreqs[s]

        # Renormalize: emit bytes while state is too large
        max_state = freq * (RANS_LOWER >> RANS_PRECISION) << 8
        while state >= max_state:
            output.append(state & 0xFF)
            state >>= 8

        # Encode symbol
        state = (state // freq) * RANS_TOTAL + (state % freq) + start

    # Flush final state (4 bytes)
    for _ in range(4):
        output.append(state & 0xFF)
        state >>= 8

    # Reverse because we encoded backwards
    output.reverse()
    return bytes(output)


def rans_decode(data: bytes, freqs: list, count: int) -> np.ndarray:
    """rANS decode compressed bytes back to symbols."""
    cumfreqs = freqs_to_cumfreqs(freqs)
    num_symbols = len(freqs)

    # Build symbol lookup table for fast decoding
    sym_table = [0] * RANS_TOTAL
    for s in range(num_symbols):
        for j in range(cumfreqs[s], cumfreqs[s + 1]):
            sym_table[j] = s

    # Read initial state (first 4 bytes)
    pos = 0
    state = 0
    for i in range(4):
        state = (state << 8) | data[pos]
        pos += 1

    symbols = np.zeros(count, dtype=np.uint8)

    for i in range(count):
        # Decode symbol
        slot = state % RANS_TOTAL
        s = sym_table[slot]
        symbols[i] = s

        # Update state
        freq = freqs[s]
        start = cumfreqs[s]
        state = freq * (state // RANS_TOTAL) + (state % RANS_TOTAL) - start

        # Renormalize: read bytes while state is too small
        while state < RANS_LOWER and pos < len(data):
            state = (state << 8) | data[pos]
            pos += 1

    return symbols


# =============================================================================
# Quantization
# =============================================================================

def quantize_uniform(values: np.ndarray, bits: int):
    """Quantize float array to uniform int levels. Returns (quantized, scale, zero)."""
    levels = (1 << bits) - 1
    vmin = float(values.min())
    vmax = float(values.max())
    scale = (vmax - vmin) / levels if levels > 0 else 1.0
    if scale == 0:
        scale = 1.0
    quantized = np.clip(np.round((values - vmin) / scale), 0, levels).astype(np.uint8)
    return quantized, scale, vmin


def dequantize_uniform(quantized: np.ndarray, scale: float, zero: float) -> np.ndarray:
    """Dequantize back to float."""
    return quantized.astype(np.float32) * scale + zero


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """Convert bfloat16 stored as V2 dtype to float32."""
    raw_bytes = raw.tobytes()
    n = len(raw_bytes) // 2
    floats = np.zeros(n, dtype=np.float32)
    for i in range(n):
        f32_bytes = b'\x00\x00' + raw_bytes[i * 2:(i + 1) * 2]
        floats[i] = struct.unpack('<f', f32_bytes)[0]
    return floats.reshape(raw.shape)


def to_float32(arr: np.ndarray) -> np.ndarray:
    """Convert any array to float32."""
    if arr.dtype == np.dtype('V2'):
        return bf16_to_f32(arr)
    try:
        return arr.astype(np.float32)
    except (ValueError, TypeError):
        return None


# =============================================================================
# Full Compression Pipeline
# =============================================================================

def compress_model(state: dict, bits: int = 6) -> bytes:
    """
    Compress a model state dict using per-layer ANS encoding.

    Format:
    [4B magic] [4B manifest_len] [manifest_json] [4B num_chunks]
    For each chunk:
      [4B key_len] [key_bytes] [4B shape_dims] [shape...] [4B freq_table_bytes] [freq_table]
      [4B data_len] [ans_compressed_data] [4B scale_bytes] [4B zero_bytes] [1B bits]
    """
    num_symbols = (1 << bits)

    chunks = []
    total_raw = 0
    total_compressed = 0

    for key in sorted(state.keys()):
        val = to_float32(state[key])
        if val is None or val.size < 4:
            continue

        flat = val.flatten()
        total_raw += flat.size * bits / 8

        # Quantize
        quantized, scale, zero = quantize_uniform(flat, bits)

        # Build frequency table
        freqs = build_frequency_table(quantized, num_symbols)

        # ANS encode
        compressed = rans_encode(quantized, freqs)
        total_compressed += len(compressed)

        chunks.append({
            'key': key,
            'shape': list(val.shape),
            'scale': scale,
            'zero': zero,
            'bits': bits,
            'freqs': freqs,
            'data': compressed,
            'count': flat.size,
        })

    # Build binary output
    result = bytearray()
    result.extend(b'ANSW')  # magic
    result.extend(struct.pack('<I', len(chunks)))

    for chunk in chunks:
        # Key
        key_bytes = chunk['key'].encode()
        result.extend(struct.pack('<I', len(key_bytes)))
        result.extend(key_bytes)

        # Shape
        result.extend(struct.pack('<I', len(chunk['shape'])))
        for dim in chunk['shape']:
            result.extend(struct.pack('<I', dim))

        # Scale, zero, bits, count
        result.extend(struct.pack('<f', chunk['scale']))
        result.extend(struct.pack('<f', chunk['zero']))
        result.extend(struct.pack('<B', chunk['bits']))
        result.extend(struct.pack('<I', chunk['count']))

        # Frequency table (num_symbols × 2 bytes each)
        for f in chunk['freqs']:
            result.extend(struct.pack('<H', f))

        # Compressed data
        result.extend(struct.pack('<I', len(chunk['data'])))
        result.extend(chunk['data'])

    print(f"  Raw int{bits} packed:  {total_raw / 1024 / 1024:.2f} MB")
    print(f"  ANS compressed:   {total_compressed / 1024 / 1024:.2f} MB")
    print(f"  + overhead:       {(len(result) - total_compressed) / 1024:.1f} KB")
    print(f"  Total artifact:   {len(result) / 1024 / 1024:.2f} MB")
    print(f"  Ratio:            {total_compressed / total_raw * 100:.1f}%")

    return bytes(result)


def decompress_model(data: bytes) -> dict:
    """Decompress ANS-encoded model back to numpy state dict."""
    pos = 0

    # Magic
    magic = data[pos:pos + 4]
    pos += 4
    assert magic == b'ANSW', f"Bad magic: {magic}"

    num_chunks = struct.unpack('<I', data[pos:pos + 4])[0]
    pos += 4

    state = {}

    for _ in range(num_chunks):
        # Key
        key_len = struct.unpack('<I', data[pos:pos + 4])[0]
        pos += 4
        key = data[pos:pos + key_len].decode()
        pos += key_len

        # Shape
        n_dims = struct.unpack('<I', data[pos:pos + 4])[0]
        pos += 4
        shape = []
        for _ in range(n_dims):
            shape.append(struct.unpack('<I', data[pos:pos + 4])[0])
            pos += 4

        # Scale, zero, bits, count
        scale = struct.unpack('<f', data[pos:pos + 4])[0]
        pos += 4
        zero = struct.unpack('<f', data[pos:pos + 4])[0]
        pos += 4
        bits = struct.unpack('<B', data[pos:pos + 1])[0]
        pos += 1
        count = struct.unpack('<I', data[pos:pos + 4])[0]
        pos += 4

        # Frequency table
        num_symbols = 1 << bits
        freqs = []
        for _ in range(num_symbols):
            freqs.append(struct.unpack('<H', data[pos:pos + 2])[0])
            pos += 2

        # Compressed data
        data_len = struct.unpack('<I', data[pos:pos + 4])[0]
        pos += 4
        compressed = data[pos:pos + data_len]
        pos += data_len

        # Decode
        quantized = rans_decode(compressed, freqs, count)

        # Dequantize
        values = dequantize_uniform(quantized, scale, zero)
        state[key] = values.reshape(shape)

    return state


def analyze(state: dict, bits: int = 6):
    """Analyze potential ANS savings vs LZMA."""
    import zlib

    num_symbols = 1 << bits
    print(f"\n{'=' * 60}")
    print(f"ANS vs LZMA Compression Analysis (int{bits})")
    print(f"{'=' * 60}\n")

    total_params = 0
    total_entropy_bits = 0
    total_lzma_bytes = 0
    total_ans_bytes = 0

    for key in sorted(state.keys()):
        val = to_float32(state[key])
        if val is None or val.size < 100:
            continue

        flat = val.flatten()
        quantized, _, _ = quantize_uniform(flat, bits)
        n = quantized.size
        total_params += n

        # Entropy (theoretical minimum)
        counts = Counter(quantized.tolist())
        ent = -sum((c / n) * math.log2(c / n) for c in counts.values())
        entropy_bits = ent * n
        total_entropy_bits += entropy_bits

        # LZMA size (pack to bytes first, like current pipeline)
        packed = bytearray()
        buf = 0
        buf_bits = 0
        for v in quantized:
            buf |= int(v) << buf_bits
            buf_bits += bits
            while buf_bits >= 8:
                packed.append(buf & 0xFF)
                buf >>= 8
                buf_bits -= 8
        if buf_bits > 0:
            packed.append(buf & 0xFF)

        lzma_compressed = zlib.compress(bytes(packed), 9)
        total_lzma_bytes += len(lzma_compressed)

        # ANS size (actual compression)
        freqs = build_frequency_table(quantized, num_symbols)
        ans_data = rans_encode(quantized, freqs)
        ans_size = len(ans_data) + num_symbols * 2  # data + freq table
        total_ans_bytes += ans_size

    print(f"Total parameters:     {total_params:>12,}")
    print(f"Theoretical minimum:  {total_entropy_bits / 8 / 1024 / 1024:>12.2f} MB ({total_entropy_bits / total_params:.2f} bits/param)")
    print(f"LZMA (current):       {total_lzma_bytes / 1024 / 1024:>12.2f} MB ({total_lzma_bytes * 8 / total_params:.2f} bits/param)")
    print(f"ANS (ours):           {total_ans_bytes / 1024 / 1024:>12.2f} MB ({total_ans_bytes * 8 / total_params:.2f} bits/param)")
    print()

    lzma_gap = total_lzma_bytes - total_entropy_bits / 8
    ans_gap = total_ans_bytes - total_entropy_bits / 8
    savings = total_lzma_bytes - total_ans_bytes

    print(f"LZMA waste:           {lzma_gap / 1024:.1f} KB above theoretical minimum")
    print(f"ANS waste:            {ans_gap / 1024:.1f} KB above theoretical minimum")
    print(f"ANS savings vs LZMA:  {savings / 1024:.1f} KB ({savings / total_lzma_bytes * 100:.1f}%)")
    print(f"Extra params at int{bits}: {int(savings * 8 / bits):,}")


def main():
    parser = argparse.ArgumentParser(description="ANS Weight Compressor")
    parser.add_argument("--input", required=True, help="Input .npz or .ans file")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--analyze", action="store_true", help="Analyze ANS vs LZMA savings")
    parser.add_argument("--bits", type=int, default=6, help="Quantization bits")
    parser.add_argument("--decompress", action="store_true", help="Decompress mode")
    parser.add_argument("--verify", action="store_true", help="Verify roundtrip")

    args = parser.parse_args()

    if args.decompress:
        print(f"Decompressing {args.input}...")
        data = Path(args.input).read_bytes()
        state = decompress_model(data)
        print(f"Restored {len(state)} tensors")
        if args.output:
            np.savez(args.output, **state)
            print(f"Saved to {args.output}")
        return

    # Load model
    print(f"Loading {args.input}...")
    state = dict(np.load(args.input, allow_pickle=True))
    n_params = sum(v.size for v in state.values() if hasattr(v, 'size') and v.dtype != object)
    print(f"Parameters: {n_params:,}")

    if args.analyze:
        analyze(state, args.bits)

    if args.output:
        print(f"\nCompressing with int{args.bits} + ANS...")
        compressed = compress_model(state, args.bits)
        Path(args.output).write_bytes(compressed)
        print(f"\nSaved to {args.output}")

        if args.verify:
            print("\nVerifying roundtrip...")
            restored = decompress_model(compressed)
            max_err = 0
            for key in state:
                orig = to_float32(state[key])
                if orig is None or key not in restored:
                    continue
                err = np.max(np.abs(orig - restored[key]))
                max_err = max(max_err, err)
            print(f"Max roundtrip error: {max_err:.6f}")
            print(f"({'OK — within quantization noise' if max_err < 0.01 else 'WARNING — large error'})")


if __name__ == "__main__":
    main()
