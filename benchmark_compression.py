"""
Benchmark rANS entropy coding vs Brotli/LZMA on quantized model weights.
Tests compression ratio on actual int6/int8 quantized weight distributions.
"""
import io
import lzma
import math
import struct
import time

import brotli
import constriction
import numpy as np
import torch


def _byte_shuffle(data, stride=2):
    """Byte-shuffle preprocessing (same as SOTA)."""
    if stride <= 1 or len(data) < stride:
        return data
    src = np.frombuffer(data, dtype=np.uint8)
    n = len(src)
    out = np.empty(n, dtype=np.uint8)
    dest_off = 0
    for pos in range(stride):
        chunk = src[pos::stride]
        out[dest_off : dest_off + len(chunk)] = chunk
        dest_off += len(chunk)
    return b"BSHF" + bytes([stride]) + out.tobytes()


def compress_brotli(data):
    """Brotli-11 with byte-shuffle (SOTA approach)."""
    shuffled = _byte_shuffle(data)
    return brotli.compress(shuffled, quality=11)


def compress_lzma(data):
    """LZMA with byte-shuffle."""
    shuffled = _byte_shuffle(data)
    return lzma.compress(shuffled, preset=6)


def compress_rans_per_row(q_tensor, scales):
    """
    rANS compression with per-row Laplacian entropy model.
    q_tensor: int8 tensor (rows x cols) of quantized weights
    scales: float16 tensor (rows,) of per-row scales
    Returns: compressed bytes
    """
    q_np = q_tensor.numpy().astype(np.int32)
    rows, cols = q_np.shape

    # Collect all encoded bytes
    all_compressed = bytearray()

    # Header: rows, cols
    all_compressed.extend(struct.pack("<II", rows, cols))

    # Store scales as float16 bytes
    scale_bytes = scales.numpy().tobytes()
    all_compressed.extend(scale_bytes)

    # For each row, fit Laplacian scale and encode with rANS
    # Use a quantized Laplacian model: P(x) ∝ exp(-|x| / b)
    # where b is the Laplacian scale parameter

    # Compute per-row Laplacian scale (b = mean(|x|))
    abs_vals = np.abs(q_np).astype(np.float64)
    row_b = abs_vals.mean(axis=1).clip(min=0.5)  # Laplacian scale

    # Store row_b as float32
    all_compressed.extend(row_b.astype(np.float32).tobytes())

    # Encode all rows using rANS with per-row Laplacian
    # Use constriction's stack-based ANS coder
    min_val = int(q_np.min())
    max_val = int(q_np.max())
    alphabet_size = max_val - min_val + 1

    all_compressed.extend(struct.pack("<ii", min_val, max_val))

    for r in range(rows):
        row = q_np[r]
        b = float(row_b[r])

        # Build probability table for this row's Laplacian distribution
        symbols = np.arange(min_val, max_val + 1, dtype=np.float64)
        log_probs = -np.abs(symbols) / b
        probs = np.exp(log_probs - log_probs.max())  # Normalize
        probs /= probs.sum()
        probs = probs.clip(min=1e-10)  # Avoid zero probabilities
        probs /= probs.sum()  # Renormalize

        # Convert probabilities to fixed-point for rANS
        probs_f32 = probs.astype(np.float32)

        # Encode using constriction
        encoder = constriction.stream.stack.AnsCoder()
        # Shift values to 0-based indexing
        shifted = (row - min_val).astype(np.int32)
        # Encode in reverse order (stack-based ANS)
        model = constriction.stream.model.Categorical(probs_f32)
        encoder.encode_reverse(shifted, model)

        compressed_row = encoder.get_compressed()
        # Store length + data
        all_compressed.extend(struct.pack("<I", len(compressed_row)))
        all_compressed.extend(compressed_row.tobytes())

    return bytes(all_compressed)


def decompress_rans_per_row(data):
    """Decompress rANS-encoded weights."""
    offset = 0
    rows, cols = struct.unpack_from("<II", data, offset)
    offset += 8

    # Read scales (float16, rows values)
    scale_bytes = data[offset : offset + rows * 2]
    scales = np.frombuffer(scale_bytes, dtype=np.float16)
    offset += rows * 2

    # Read Laplacian params
    row_b = np.frombuffer(data[offset : offset + rows * 4], dtype=np.float32)
    offset += rows * 4

    min_val, max_val = struct.unpack_from("<ii", data, offset)
    offset += 8

    q_np = np.zeros((rows, cols), dtype=np.int32)

    for r in range(rows):
        b = float(row_b[r])
        symbols = np.arange(min_val, max_val + 1, dtype=np.float64)
        log_probs = -np.abs(symbols) / b
        probs = np.exp(log_probs - log_probs.max())
        probs /= probs.sum()
        probs = probs.clip(min=1e-10)
        probs /= probs.sum()
        probs_f32 = probs.astype(np.float32)

        # Read compressed row
        comp_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        compressed_words = np.frombuffer(data[offset : offset + comp_len * 4], dtype=np.uint32)
        offset += comp_len * 4

        decoder = constriction.stream.stack.AnsCoder(compressed_words)
        model = constriction.stream.model.Categorical(probs_f32)
        shifted = decoder.decode(model, cols)
        q_np[r] = shifted + min_val

    return torch.from_numpy(q_np.astype(np.int8)), torch.from_numpy(scales)


def benchmark_on_synthetic_weights():
    """Test on synthetic int6 quantized weights matching typical neural net distributions."""
    print("=" * 70)
    print("COMPRESSION BENCHMARK: rANS vs Brotli vs LZMA on quantized weights")
    print("=" * 70)

    # Simulate typical GPTQ int6 quantized weights
    # Neural network weights are approximately Gaussian/Laplacian
    torch.manual_seed(42)
    configs = [
        ("MLP weight (512x2048)", 512, 2048, 6, "gaussian"),
        ("Attn QKV (512x256)", 512, 256, 6, "gaussian"),
        ("Embedding (8192x512)", 8192, 512, 8, "gaussian"),
    ]

    total_raw = 0
    total_brotli = 0
    total_lzma = 0
    total_rans = 0

    for name, rows, cols, bits, dist_type in configs:
        clip_range = 2 ** (bits - 1) - 1  # 31 for int6, 127 for int8

        # Generate weights matching typical quantized distribution
        if dist_type == "gaussian":
            w_float = torch.randn(rows, cols) * 0.02  # Typical weight std
        else:
            w_float = torch.randn(rows, cols) * 0.02

        # Simulate GPTQ quantization with SDClip
        row_std = w_float.std(dim=1)
        clip_sigmas = 12.85 if bits == 6 else 20.0
        scales = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
        q = torch.clamp(torch.round(w_float / scales.float().unsqueeze(1)), -clip_range, clip_range).to(torch.int8)

        # Raw size
        raw_bytes = q.numpy().tobytes()
        raw_size = len(raw_bytes)

        # Brotli-11 with byte-shuffle
        t0 = time.perf_counter()
        brotli_compressed = compress_brotli(raw_bytes)
        brotli_time = time.perf_counter() - t0
        brotli_size = len(brotli_compressed)

        # LZMA
        t0 = time.perf_counter()
        lzma_compressed = compress_lzma(raw_bytes)
        lzma_time = time.perf_counter() - t0
        lzma_size = len(lzma_compressed)

        # rANS per-row
        t0 = time.perf_counter()
        rans_compressed = compress_rans_per_row(q, scales)
        rans_time = time.perf_counter() - t0
        rans_size = len(rans_compressed)

        # Verify roundtrip
        q_dec, s_dec = decompress_rans_per_row(rans_compressed)
        assert torch.equal(q, q_dec), f"rANS roundtrip FAILED for {name}"

        # Entropy calculation
        q_np = q.numpy().flatten()
        values, counts = np.unique(q_np, return_counts=True)
        probs = counts / counts.sum()
        entropy_bits = -np.sum(probs * np.log2(probs))
        theoretical_min = int(math.ceil(len(q_np) * entropy_bits / 8))

        total_raw += raw_size
        total_brotli += brotli_size
        total_lzma += lzma_size
        total_rans += rans_size

        print(f"\n{name} ({rows}x{cols}, int{bits}):")
        print(f"  Raw:     {raw_size:>10,} bytes")
        print(f"  Entropy: {theoretical_min:>10,} bytes (theoretical minimum, {entropy_bits:.2f} bits/symbol)")
        print(f"  Brotli:  {brotli_size:>10,} bytes ({brotli_size/raw_size*100:.1f}%, {brotli_time*1000:.0f}ms)")
        print(f"  LZMA:    {lzma_size:>10,} bytes ({lzma_size/raw_size*100:.1f}%, {lzma_time*1000:.0f}ms)")
        print(f"  rANS:    {rans_size:>10,} bytes ({rans_size/raw_size*100:.1f}%, {rans_time*1000:.0f}ms)")
        print(f"  rANS vs Brotli: {(1 - rans_size/brotli_size)*100:+.1f}%")

    print(f"\n{'='*70}")
    print(f"TOTALS:")
    print(f"  Raw:    {total_raw:>12,} bytes")
    print(f"  Brotli: {total_brotli:>12,} bytes ({total_brotli/total_raw*100:.1f}%)")
    print(f"  LZMA:   {total_lzma:>12,} bytes ({total_lzma/total_raw*100:.1f}%)")
    print(f"  rANS:   {total_rans:>12,} bytes ({total_rans/total_raw*100:.1f}%)")
    print(f"  rANS saves {(1 - total_rans/total_brotli)*100:.1f}% over Brotli")
    print(f"  rANS saves {(1 - total_rans/total_lzma)*100:.1f}% over LZMA")
    freed_bytes = total_brotli - total_rans
    print(f"  Freed capacity: {freed_bytes:,} bytes ({freed_bytes/1e6:.2f} MB)")


def benchmark_on_real_model():
    """Test on actual trained model weights if available."""
    model_path = "final_model.pt"
    try:
        sd = torch.load(model_path, map_location="cpu")
        print(f"\n\n{'='*70}")
        print(f"REAL MODEL BENCHMARK: {model_path}")
        print(f"{'='*70}")

        total_raw = 0
        total_brotli = 0
        total_rans = 0

        for name, tensor in sd.items():
            if tensor.ndim != 2 or tensor.numel() < 65536:
                continue

            # Quantize to int6 with SDClip
            rows, cols = tensor.shape
            row_std = tensor.float().std(dim=1)
            clip_range = 31  # int6
            scales = (12.85 * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
            q = torch.clamp(
                torch.round(tensor.float() / scales.float().unsqueeze(1)),
                -clip_range, clip_range,
            ).to(torch.int8)

            raw_bytes = q.numpy().tobytes()
            raw_size = len(raw_bytes)
            brotli_size = len(compress_brotli(raw_bytes))
            rans_size = len(compress_rans_per_row(q, scales))

            total_raw += raw_size
            total_brotli += brotli_size
            total_rans += rans_size

            saving = (1 - rans_size / brotli_size) * 100
            print(f"  {name:50s} {rows:>5}x{cols:<5} brotli:{brotli_size:>8,} rans:{rans_size:>8,} ({saving:+.1f}%)")

        if total_brotli > 0:
            print(f"\n  Total: brotli={total_brotli:,} rans={total_rans:,} "
                  f"saving={(1 - total_rans/total_brotli)*100:.1f}%")
            print(f"  Freed: {(total_brotli - total_rans):,} bytes")
    except FileNotFoundError:
        print(f"\nNo model found at {model_path}, skipping real model benchmark")


if __name__ == "__main__":
    benchmark_on_synthetic_weights()
    benchmark_on_real_model()
