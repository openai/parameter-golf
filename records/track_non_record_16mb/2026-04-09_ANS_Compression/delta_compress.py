#!/usr/bin/env python3
"""
Layer-Delta Encoding + ANS Weight Compression for Parameter Golf.

Instead of storing each layer's weights independently:
  Layer1(6 bits) + Layer2(6 bits) + ... + Layer11(6 bits)

We store:
  Layer1(6 bits) + Delta2(3-4 bits) + Delta3(3-4 bits) + ...

Adjacent transformer layers learn similar features. The delta between
them is mostly near-zero, which compresses dramatically.

Usage:
  # After training, compress the model
  python delta_compress.py --input logs/model.npz --output model_delta.bin --analyze

  # Decompress back to verify
  python delta_compress.py --decompress --input model_delta.bin --output model_restored.npz
"""

import argparse
import json
import math
import struct
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np


def analyze_layer_similarity(state: dict) -> None:
    """Analyze how similar adjacent layers are — to see if delta encoding helps."""

    # Group parameters by layer
    layers = {}
    for key, val in state.items():
        parts = key.split(".")
        # Find layer index (e.g., "blocks.0.attn.c_q.weight" → layer 0)
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "blocks" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                param_name = ".".join(parts[i + 2:])
                break
        if layer_idx is not None:
            if layer_idx not in layers:
                layers[layer_idx] = {}
            layers[layer_idx][param_name] = val

    if len(layers) < 2:
        print("Not enough layers to analyze similarity.")
        return

    print(f"\n{'='*60}")
    print(f"Layer Similarity Analysis ({len(layers)} layers)")
    print(f"{'='*60}\n")

    sorted_layers = sorted(layers.keys())

    for i in range(len(sorted_layers) - 1):
        l1, l2 = sorted_layers[i], sorted_layers[i + 1]
        print(f"Layer {l1} → Layer {l2}:")

        total_params = 0
        total_delta_l1 = 0.0
        total_near_zero = 0

        for param_name in layers[l1]:
            if param_name not in layers[l2]:
                continue

            w1 = layers[l1][param_name].astype(np.float32).flatten()
            w2 = layers[l2][param_name].astype(np.float32).flatten()

            if w1.shape != w2.shape:
                continue

            delta = w2 - w1
            n = delta.size
            total_params += n

            # L1 norm of delta vs L1 norm of weights
            delta_l1 = np.mean(np.abs(delta))
            weight_l1 = np.mean(np.abs(w1))
            ratio = delta_l1 / (weight_l1 + 1e-10)

            # How many delta values are near zero?
            threshold = np.std(w1) * 0.1  # within 10% of weight std
            near_zero = np.sum(np.abs(delta) < threshold) / n * 100

            total_delta_l1 += delta_l1 * n
            total_near_zero += np.sum(np.abs(delta) < threshold)

            print(f"  {param_name:30s}  delta/weight: {ratio:.3f}  near_zero: {near_zero:.1f}%  shape: {w1.shape}")

        if total_params > 0:
            avg_near_zero = total_near_zero / total_params * 100
            print(f"  TOTAL: {total_params:,} params, {avg_near_zero:.1f}% deltas near zero\n")


def compute_entropy(values: np.ndarray) -> float:
    """Compute Shannon entropy of a discrete array in bits."""
    counts = Counter(values.flatten().tolist())
    total = sum(counts.values())
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def delta_encode_layers(state: dict, quant_bits: int = 6) -> dict:
    """
    Delta-encode adjacent transformer layers.

    Returns a dict with:
    - Non-layer params stored directly (quantized)
    - First layer stored fully (quantized to quant_bits)
    - Subsequent layers stored as delta from previous (quantized to fewer bits)
    """

    # Separate layer params from non-layer params
    layer_params = {}  # layer_idx → {param_name: array}
    other_params = {}  # key → array

    for key, val in state.items():
        parts = key.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "blocks" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                param_name = ".".join(parts[i + 2:])
                break

        if layer_idx is not None:
            if layer_idx not in layer_params:
                layer_params[layer_idx] = {}
            try:
                layer_params[layer_idx][param_name] = val.astype(np.float32)
            except (ValueError, TypeError):
                continue
        else:
            if val.dtype == object or len(val.shape) == 0:
                continue
            try:
                other_params[key] = val.astype(np.float32)
            except (ValueError, TypeError):
                continue

    sorted_layers = sorted(layer_params.keys())
    if len(sorted_layers) < 2:
        return {"no_delta": True, "state": state}

    encoded = {
        "meta": {
            "num_layers": len(sorted_layers),
            "layer_indices": sorted_layers,
            "quant_bits": quant_bits,
        },
        "other_params": {},
        "base_layer": {},    # First layer, fully quantized
        "deltas": {},        # Subsequent layers as deltas
    }

    # Quantize and store non-layer params
    for key, val in other_params.items():
        v = val.astype(np.float32).flatten()
        q, scale, zero = uniform_quantize(v, quant_bits)
        encoded["other_params"][key] = {
            "quantized": q,
            "scale": scale,
            "zero": zero,
            "shape": list(val.shape),
            "dtype": str(val.dtype),
        }

    # Store first layer fully quantized
    base_idx = sorted_layers[0]
    for param_name, val in layer_params[base_idx].items():
        v = val.flatten()
        q, scale, zero = uniform_quantize(v, quant_bits)
        encoded["base_layer"][param_name] = {
            "quantized": q,
            "scale": scale,
            "zero": zero,
            "shape": list(val.shape),
        }

    # Store subsequent layers as deltas
    for i in range(1, len(sorted_layers)):
        prev_idx = sorted_layers[i - 1]
        curr_idx = sorted_layers[i]
        encoded["deltas"][curr_idx] = {}

        for param_name in layer_params[curr_idx]:
            curr = layer_params[curr_idx][param_name].flatten()

            if param_name in layer_params[prev_idx] and \
               layer_params[prev_idx][param_name].shape == layer_params[curr_idx][param_name].shape:
                # Delta encoding
                prev = layer_params[prev_idx][param_name].flatten()
                delta = curr - prev

                # Delta has lower variance → can use fewer bits
                delta_bits = max(quant_bits - 2, 3)  # save 2 bits on deltas
                q, scale, zero = uniform_quantize(delta, delta_bits)

                encoded["deltas"][curr_idx][param_name] = {
                    "is_delta": True,
                    "quantized": q,
                    "scale": scale,
                    "zero": zero,
                    "shape": list(layer_params[curr_idx][param_name].shape),
                    "bits": delta_bits,
                }
            else:
                # No matching previous layer param, store fully
                q, scale, zero = uniform_quantize(curr, quant_bits)
                encoded["deltas"][curr_idx][param_name] = {
                    "is_delta": False,
                    "quantized": q,
                    "scale": scale,
                    "zero": zero,
                    "shape": list(layer_params[curr_idx][param_name].shape),
                    "bits": quant_bits,
                }

    return encoded


def uniform_quantize(values: np.ndarray, bits: int):
    """Uniform quantization to given bit width. Returns (quantized_ints, scale, zero_point)."""
    levels = (1 << bits) - 1
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    scale = (v_max - v_min) / levels if levels > 0 else 1.0
    if scale == 0:
        scale = 1.0

    quantized = np.clip(np.round((values - v_min) / scale), 0, levels).astype(np.uint8)
    return quantized, scale, v_min


def uniform_dequantize(quantized: np.ndarray, scale: float, zero: float):
    """Dequantize back to float."""
    return quantized.astype(np.float32) * scale + zero


def pack_bits(data: np.ndarray, bits: int) -> bytes:
    """Pack array of uint8 values (each using `bits` bits) into a tight byte stream."""
    if bits == 8:
        return data.tobytes()

    result = bytearray()
    buffer = 0
    buffer_bits = 0

    for val in data.flat:
        buffer = buffer | (int(val) << buffer_bits)
        buffer_bits += bits
        while buffer_bits >= 8:
            result.append(buffer & 0xFF)
            buffer >>= 8
            buffer_bits -= 8

    if buffer_bits > 0:
        result.append(buffer & 0xFF)

    return bytes(result)


def unpack_bits(data: bytes, bits: int, count: int) -> np.ndarray:
    """Unpack a tight bit stream back to uint8 array."""
    if bits == 8:
        return np.frombuffer(data[:count], dtype=np.uint8).copy()

    result = np.zeros(count, dtype=np.uint8)
    mask = (1 << bits) - 1
    buffer = 0
    buffer_bits = 0
    byte_idx = 0

    for i in range(count):
        while buffer_bits < bits:
            buffer = buffer | (data[byte_idx] << buffer_bits)
            buffer_bits += 8
            byte_idx += 1
        result[i] = buffer & mask
        buffer >>= bits
        buffer_bits -= bits

    return result


def serialize_encoded(encoded: dict) -> bytes:
    """Serialize the delta-encoded model to a compact binary format."""

    # We'll build a simple format:
    # [4 bytes: magic] [4 bytes: meta_json_len] [meta_json] [compressed_weights]

    meta = encoded["meta"]

    # Collect all weight data in order
    weight_chunks = []
    chunk_info = []

    # Other params
    for key, info in encoded.get("other_params", {}).items():
        packed = pack_bits(info["quantized"], meta["quant_bits"])
        weight_chunks.append(packed)
        chunk_info.append({
            "type": "other",
            "key": key,
            "scale": info["scale"],
            "zero": info["zero"],
            "shape": info["shape"],
            "bits": meta["quant_bits"],
            "packed_len": len(packed),
            "count": int(np.prod(info["shape"])),
        })

    # Base layer
    for param_name, info in encoded.get("base_layer", {}).items():
        packed = pack_bits(info["quantized"], meta["quant_bits"])
        weight_chunks.append(packed)
        chunk_info.append({
            "type": "base",
            "param": param_name,
            "scale": info["scale"],
            "zero": info["zero"],
            "shape": info["shape"],
            "bits": meta["quant_bits"],
            "packed_len": len(packed),
            "count": int(np.prod(info["shape"])),
        })

    # Delta layers
    for layer_idx in sorted(encoded.get("deltas", {}).keys()):
        for param_name, info in encoded["deltas"][layer_idx].items():
            bits = info.get("bits", meta["quant_bits"])
            packed = pack_bits(info["quantized"], bits)
            weight_chunks.append(packed)
            chunk_info.append({
                "type": "delta" if info["is_delta"] else "full",
                "layer": layer_idx,
                "param": param_name,
                "scale": info["scale"],
                "zero": info["zero"],
                "shape": info["shape"],
                "bits": bits,
                "packed_len": len(packed),
                "count": int(np.prod(info["shape"])),
            })

    # Build final binary
    all_weights = b"".join(weight_chunks)

    manifest = {
        "meta": meta,
        "chunks": chunk_info,
        "total_weight_bytes": len(all_weights),
    }
    manifest_json = json.dumps(manifest).encode()

    # Compress weights with zlib (could use lzma for better ratio)
    compressed_weights = zlib.compress(all_weights, 9)

    # Header
    magic = b"DLTA"
    result = magic
    result += struct.pack("<I", len(manifest_json))
    result += manifest_json
    result += struct.pack("<I", len(compressed_weights))
    result += compressed_weights

    return result


def compare_compression(state: dict, quant_bits: int = 6) -> None:
    """Compare standard quantization vs delta encoding compression ratios."""

    print(f"\n{'='*60}")
    print("Compression Comparison")
    print(f"{'='*60}\n")

    # Standard: quantize everything to quant_bits, compress
    all_weights_standard = bytearray()
    for key in sorted(state.keys()):
        val = state[key]
        if val.dtype == object or len(val.shape) == 0:
            continue  # skip non-numeric params
        try:
            val = val.astype(np.float32).flatten()
        except (ValueError, TypeError):
            continue
        q, _, _ = uniform_quantize(val, quant_bits)
        packed = pack_bits(q, quant_bits)
        all_weights_standard.extend(packed)

    standard_raw = len(all_weights_standard)
    standard_compressed = len(zlib.compress(bytes(all_weights_standard), 9))

    print(f"Standard int{quant_bits}:")
    print(f"  Raw packed:   {standard_raw:>12,} bytes ({standard_raw/1024/1024:.2f} MB)")
    print(f"  Compressed:   {standard_compressed:>12,} bytes ({standard_compressed/1024/1024:.2f} MB)")

    # Entropy of standard quantized weights
    all_q_standard = np.frombuffer(bytes(all_weights_standard), dtype=np.uint8)
    entropy_standard = compute_entropy(all_q_standard)
    print(f"  Byte entropy: {entropy_standard:.3f} bits/byte (max 8.0)")

    # Delta: delta-encode layers then compress
    encoded = delta_encode_layers(state, quant_bits)
    if "no_delta" in encoded:
        print("\nNo layers found for delta encoding.")
        return

    delta_binary = serialize_encoded(encoded)
    delta_size = len(delta_binary)

    print(f"\nDelta-encoded int{quant_bits}/int{max(quant_bits-2,3)}:")
    print(f"  Total size:   {delta_size:>12,} bytes ({delta_size/1024/1024:.2f} MB)")

    savings = standard_compressed - delta_size
    savings_pct = savings / standard_compressed * 100 if standard_compressed > 0 else 0

    print(f"\nSavings: {savings:,} bytes ({savings_pct:.1f}%)")

    if savings > 0:
        print(f"That's {savings/1024:.1f} KB freed — enough for ~{savings*8//quant_bits:,} more parameters")
    else:
        print("Delta encoding is WORSE for this model. Layers may be too different.")


def main():
    parser = argparse.ArgumentParser(description="Layer-Delta Compression for Parameter Golf")
    parser.add_argument("--input", required=True, help="Input .npz model file")
    parser.add_argument("--output", help="Output compressed file")
    parser.add_argument("--analyze", action="store_true", help="Analyze layer similarity")
    parser.add_argument("--compare", action="store_true", help="Compare compression ratios")
    parser.add_argument("--bits", type=int, default=6, help="Quantization bits (default: 6)")
    parser.add_argument("--decompress", action="store_true", help="Decompress mode")

    args = parser.parse_args()

    if args.decompress:
        print("Decompression not yet implemented — add if delta encoding proves useful")
        return

    # Load model
    print(f"Loading {args.input}...")
    state = dict(np.load(args.input, allow_pickle=True))

    n_params = sum(int(np.prod(v.shape)) for v in state.values())
    raw_bytes = sum(v.nbytes for v in state.values())
    print(f"Parameters: {n_params:,}")
    print(f"Raw size: {raw_bytes:,} bytes ({raw_bytes/1024/1024:.2f} MB)")

    if args.analyze:
        analyze_layer_similarity(state)

    if args.compare:
        compare_compression(state, args.bits)

    if args.output:
        print(f"\nDelta-encoding with {args.bits}-bit quantization...")
        encoded = delta_encode_layers(state, args.bits)
        binary = serialize_encoded(encoded)

        Path(args.output).write_bytes(binary)
        print(f"Saved to {args.output}: {len(binary):,} bytes ({len(binary)/1024/1024:.2f} MB)")


if __name__ == "__main__":
    main()
