"""Int6/Int8 packing and zlib compression utilities for artifact size optimization."""

import io
import math
import struct
import zlib

import torch
import torch.nn as nn
import numpy as np


def quantize_to_int8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric int8 quantization.

    Returns:
        (quantized_int8, scales_fp16) where scales has shape (n_rows,)
    """
    if tensor.dim() < 2:
        tensor = tensor.unsqueeze(0)
    orig_shape = tensor.shape
    flat = tensor.reshape(-1, tensor.shape[-1])

    # Per-row absmax scaling
    absmax = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = absmax / 127.0
    quantized = (flat / scale).round().clamp(-127, 127).to(torch.int8)

    return quantized.reshape(orig_shape), scale.squeeze(-1).half()


def dequantize_int8(quantized: torch.Tensor, scales: torch.Tensor,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Dequantize int8 tensor back to float."""
    if quantized.dim() < 2:
        quantized = quantized.unsqueeze(0)
    flat = quantized.reshape(-1, quantized.shape[-1]).float()
    s = scales.float().unsqueeze(-1)
    return (flat * s).reshape(quantized.shape).to(dtype)


def quantize_to_int6(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric int6 quantization (range [-31, 31]).

    Returns:
        (quantized_int8_storage, scales_fp16)
        Values are stored in int8 but clamped to [-31, 31] (6-bit range).
    """
    if tensor.dim() < 2:
        tensor = tensor.unsqueeze(0)
    orig_shape = tensor.shape
    flat = tensor.reshape(-1, tensor.shape[-1])

    absmax = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = absmax / 31.0
    quantized = (flat / scale).round().clamp(-31, 31).to(torch.int8)

    return quantized.reshape(orig_shape), scale.squeeze(-1).half()


def dequantize_int6(quantized: torch.Tensor, scales: torch.Tensor,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Dequantize int6 tensor back to float."""
    return dequantize_int8(quantized, scales, dtype)  # same math, different range


def pack_int6_bits(values: np.ndarray) -> bytes:
    """Pack int6 values (stored as int8, range [-31,31]) into 6-bit packed bytes.

    Packs 4 values into 3 bytes (4 * 6 bits = 24 bits = 3 bytes).
    Values are offset to unsigned range [0, 62] before packing.
    """
    # Offset to unsigned: [-31, 31] -> [0, 62]
    unsigned = (values.astype(np.int16) + 31).astype(np.uint8)
    flat = unsigned.flatten()

    # Pad to multiple of 4
    pad_len = (4 - len(flat) % 4) % 4
    if pad_len:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.uint8)])

    result = bytearray()
    for i in range(0, len(flat), 4):
        a, b, c, d = flat[i], flat[i+1], flat[i+2], flat[i+3]
        # Pack 4 x 6-bit values into 3 bytes
        byte0 = (a & 0x3F) | ((b & 0x03) << 6)
        byte1 = ((b >> 2) & 0x0F) | ((c & 0x0F) << 4)
        byte2 = ((c >> 4) & 0x03) | ((d & 0x3F) << 2)
        result.extend([byte0, byte1, byte2])

    return bytes(result)


def unpack_int6_bits(packed: bytes, n_values: int) -> np.ndarray:
    """Unpack 6-bit packed bytes back to int6 values (as int8, range [-31,31])."""
    data = np.frombuffer(packed, dtype=np.uint8)
    result = []

    for i in range(0, len(data), 3):
        if i + 2 >= len(data):
            break
        b0, b1, b2 = data[i], data[i+1], data[i+2]
        a = b0 & 0x3F
        b = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2)
        c = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4)
        d = (b2 >> 2) & 0x3F
        result.extend([a, b, c, d])

    # Convert back to signed: [0, 62] -> [-31, 31]
    arr = np.array(result[:n_values], dtype=np.int8)
    arr = arr.astype(np.int16) - 31
    return arr.astype(np.int8)


def save_compressed(model: nn.Module, path: str, bits: int = 6,
                    compression_level: int = 9):
    """Save model with quantization and zlib compression.

    Args:
        model: Model to save.
        path: Output file path.
        bits: Quantization bits (6 or 8).
        compression_level: zlib compression level (1-9).
    """
    quantize_fn = quantize_to_int6 if bits == 6 else quantize_to_int8

    buf = io.BytesIO()
    state = {}
    metadata = {}

    seen = set()
    for name, param in model.named_parameters():
        if param.data_ptr() in seen:
            metadata[name] = {"alias": True}
            continue
        seen.add(param.data_ptr())

        if param.dim() >= 2:
            q, s = quantize_fn(param.data)
            state[f"{name}.quant"] = q.numpy()
            state[f"{name}.scale"] = s.numpy()
            metadata[name] = {
                "shape": list(param.shape),
                "bits": bits,
                "quantized": True,
            }
        else:
            # Store 1D params (norms, biases, scales) as fp16
            state[f"{name}.fp16"] = param.data.half().numpy()
            metadata[name] = {
                "shape": list(param.shape),
                "quantized": False,
            }

    np.savez(buf, **{k: v for k, v in state.items()})
    raw_bytes = buf.getvalue()

    compressed = zlib.compress(raw_bytes, compression_level)

    with open(path, "wb") as f:
        # Header: magic + metadata size + metadata + compressed data
        import json
        meta_bytes = json.dumps(metadata).encode("utf-8")
        f.write(b"PGOLF1")  # magic
        f.write(struct.pack("<I", len(meta_bytes)))
        f.write(meta_bytes)
        f.write(compressed)

    raw_mb = len(raw_bytes) / (1024 * 1024)
    comp_mb = os.path.getsize(path) / (1024 * 1024) if hasattr(os, 'path') else len(compressed) / (1024*1024)
    print(f"[packing] Saved {path}: {raw_mb:.2f}MB raw -> {comp_mb:.2f}MB compressed "
          f"({bits}-bit, level={compression_level})")

    return os.path.getsize(path)


# Need os for getsize
import os
