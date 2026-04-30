"""Test Huffman coding vs Brotli on simulated quantized weight values.

Computes exact Huffman codes for the quantized int6 distribution,
measures actual compressed sizes, and compares with Brotli.

Usage:
    BUNDLE_DIR=./local_bundle_seed42 /tmp/torch_env/bin/python3 analyze_huffman.py
"""
import os, math, struct, heapq, time
import torch
import numpy as np
from pathlib import Path
from collections import Counter

bundle_dir = Path(os.environ.get("BUNDLE_DIR", "local_bundle_seed42"))
ema = torch.load(bundle_dir / "ema_weights.pt", map_location="cpu")


# --- Huffman implementation ---

def build_huffman_table(freq_dict):
    """Build Huffman code table from frequency dict. Returns {symbol: bitstring}."""
    if len(freq_dict) <= 1:
        return {sym: "0" for sym in freq_dict}
    heap = [[count, i, sym] for i, (sym, count) in enumerate(freq_dict.items()) if count > 0]
    heapq.heapify(heap)
    counter = len(heap)
    nodes = {}
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        counter += 1
        internal = [lo[0] + hi[0], counter, None]
        nodes[counter] = (lo, hi)
        heapq.heappush(heap, internal)

    # Traverse tree to build codes
    codes = {}
    def traverse(node, prefix=""):
        idx = node[1]
        sym = node[2]
        if sym is not None:
            codes[sym] = prefix if prefix else "0"
            return
        if idx in nodes:
            left, right = nodes[idx]
            traverse(left, prefix + "0")
            traverse(right, prefix + "1")
    if heap:
        traverse(heap[0])
    return codes


def huffman_encode(values, codes):
    """Encode values using Huffman codes. Returns bytes."""
    bits = []
    for v in values:
        bits.append(codes[v])
    bitstring = "".join(bits)
    # Pad to byte boundary
    pad = (8 - len(bitstring) % 8) % 8
    bitstring += "0" * pad
    # Convert to bytes
    out = bytearray()
    for i in range(0, len(bitstring), 8):
        out.append(int(bitstring[i:i+8], 2))
    return bytes(out), len(bitstring) - pad  # return total valid bits


def huffman_decode(data, codes, n_values):
    """Decode Huffman-encoded bytes back to values."""
    # Build decode tree
    decode_tree = {}
    for sym, code in codes.items():
        node = decode_tree
        for bit in code[:-1]:
            node = node.setdefault(bit, {})
        node[code[-1]] = sym

    # Decode
    values = []
    node = decode_tree
    bit_idx = 0
    for byte in data:
        for shift in range(7, -1, -1):
            bit = "1" if (byte >> shift) & 1 else "0"
            node = node[bit]
            if not isinstance(node, dict):
                values.append(node)
                if len(values) >= n_values:
                    return values
                node = decode_tree
            bit_idx += 1
    return values


# --- Simulate quantization and test compression ---

def quantize_sdclip(w, k, clip_range):
    row_std = w.std(dim=1, keepdim=True).clamp_min(1e-10)
    scale = k * row_std / clip_range
    q = torch.clamp(torch.round(w / scale), -clip_range, clip_range)
    return q.to(torch.int8)


# Collect int6 tensors
int6_tensors = [(n, ema[n].float()) for n in sorted(ema)
                if n != "tok_emb.weight" and ema[n].numel() > 65536
                and ema[n].is_floating_point() and ema[n].ndim == 2]

configs = [
    ("int6 k=12.85", 12.85, 31),
    ("int5 k=6.0", 6.0, 15),
    ("int5 k=12.85", 12.85, 15),
]

for label, k, clip_range in configs:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    # Quantize all tensors
    all_q = []
    for name, w in int6_tensors:
        q = quantize_sdclip(w, k, clip_range)
        all_q.append(q.flatten())
    combined = torch.cat(all_q).numpy()
    n_values = len(combined)

    # Frequency table
    freq = Counter(combined.tolist())
    total = sum(freq.values())

    # Shannon entropy
    entropy = sum(-c/total * math.log2(c/total) for c in freq.values() if c > 0)
    shannon_bytes = entropy * n_values / 8

    # Build Huffman codes
    codes = build_huffman_table(freq)

    # Expected Huffman bits
    avg_code_len = sum(len(codes[sym]) * freq[sym] / total for sym in freq)
    huffman_bytes = avg_code_len * n_values / 8

    # Actually encode
    t0 = time.time()
    encoded, n_bits = huffman_encode(combined.tolist(), codes)
    encode_time = time.time() - t0
    actual_huffman_bytes = len(encoded)

    # Verify decode (first 1000 values)
    t0 = time.time()
    decoded = huffman_decode(encoded, codes, min(1000, n_values))
    decode_time_sample = time.time() - t0
    assert decoded == combined[:len(decoded)].tolist(), "Decode mismatch!"

    # Brotli comparison (on raw int8 values)
    try:
        import brotli
        raw_bytes = combined.astype(np.int8).tobytes()
        t0 = time.time()
        brotli_compressed = brotli.compress(raw_bytes, quality=11)
        brotli_time = time.time() - t0
        brotli_bytes = len(brotli_compressed)
    except ImportError:
        brotli_bytes = None
        brotli_time = 0

    # Byte-shuffled Brotli
    try:
        if brotli_bytes is not None:
            # Stride-2 byte shuffle
            src = np.frombuffer(raw_bytes, dtype=np.uint8)
            shuffled = np.empty(len(src), dtype=np.uint8)
            shuffled[:len(src)//2] = src[0::2]
            shuffled[len(src)//2:] = src[1::2]
            brotli_shuffled = brotli.compress(shuffled.tobytes(), quality=11)
            brotli_shuffled_bytes = len(brotli_shuffled)
    except:
        brotli_shuffled_bytes = None

    # Report
    print(f"  Values: {n_values:,}")
    print(f"  Unique symbols: {len(freq)}")
    print(f"  Shannon entropy: {entropy:.4f} bits/value")
    print(f"  Shannon minimum: {shannon_bytes/1e6:.3f} MB")
    print(f"  Huffman avg code: {avg_code_len:.4f} bits/value")
    print(f"  Huffman encoded: {actual_huffman_bytes/1e6:.3f} MB ({actual_huffman_bytes/shannon_bytes*100:.1f}% of Shannon)")
    print(f"  Huffman overhead: {(actual_huffman_bytes - shannon_bytes)/1e6:.3f} MB")
    print(f"  Encode time: {encode_time:.1f}s")
    print(f"  Decode time (1K): {decode_time_sample*1000:.0f}ms → est full: {decode_time_sample*n_values/1000:.1f}s")
    if brotli_bytes:
        print(f"  Brotli-11 (raw): {brotli_bytes/1e6:.3f} MB ({brotli_bytes/shannon_bytes*100:.1f}% of Shannon)")
    if brotli_shuffled_bytes:
        print(f"  Brotli-11 (shuffled): {brotli_shuffled_bytes/1e6:.3f} MB ({brotli_shuffled_bytes/shannon_bytes*100:.1f}% of Shannon)")
    print(f"  Savings Huffman vs Brotli-shuffled: {(brotli_shuffled_bytes - actual_huffman_bytes)/1e6:.3f} MB" if brotli_shuffled_bytes else "")

    # Show code lengths for top symbols
    print(f"\n  Top 10 Huffman codes:")
    for sym, cnt in sorted(freq.items(), key=lambda x: -x[1])[:10]:
        code = codes[sym]
        print(f"    q={sym:>4d}: freq={cnt/total:>6.2%} code_len={len(code)} code={code[:20]}")
