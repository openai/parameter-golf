"""
Trit packing: serialize ternary {-1, 0, +1} weights at 5 trits per byte.

3^5 = 243 < 256, so 5 ternary values fit in one byte.
This gives 1.6 bits/param — already at entropy (zstd/zlib achieve ~0% further compression).

For comparison:
  - int8: 8 bits/param
  - int6: 6 bits/param (stored as int8, but only 6 bits used)
  - int5: 5 bits/param (stored as int8, but only 5 bits used)
  - ternary packed: 1.6 bits/param

The density advantage is 3-4x over int5/int6 in the same 16MB budget.
"""

from __future__ import annotations

import io
import math
from typing import Any

import torch
from torch import Tensor

try:
    import zstandard
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False
import zlib


# ─── Trit packing / unpacking ───────────────────────────────────────────────

_TRITS_PER_BYTE = 5
_POWERS = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32)


def pack_trits(tensor: Tensor) -> tuple[Tensor, tuple[int, ...], int]:
    """Pack ternary {-1, 0, +1} tensor into bytes. 5 trits per byte.

    Args:
        tensor: int8 tensor with values in {-1, 0, +1}

    Returns:
        packed: uint8 tensor of packed bytes
        shape: original tensor shape
        pad: number of padding trits added (to make length divisible by 5)
    """
    shape = tuple(tensor.shape)
    # Map: -1 → 0, 0 → 1, +1 → 2
    flat = (tensor.reshape(-1).to(torch.int32) + 1)

    # Pad to multiple of 5
    n = flat.numel()
    pad = (_TRITS_PER_BYTE - n % _TRITS_PER_BYTE) % _TRITS_PER_BYTE
    if pad > 0:
        flat = torch.cat([flat, torch.ones(pad, dtype=torch.int32)])  # pad with 1 (=0 in ternary)

    # Pack: groups of 5 → single byte
    # value = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4
    groups = flat.reshape(-1, _TRITS_PER_BYTE)
    powers = _POWERS.to(groups.device)
    packed = (groups * powers).sum(dim=1).to(torch.uint8)

    return packed, shape, pad


def unpack_trits(packed: Tensor, shape: tuple[int, ...], pad: int) -> Tensor:
    """Unpack bytes back to ternary {-1, 0, +1} tensor.

    Args:
        packed: uint8 tensor from pack_trits
        shape: original tensor shape
        pad: padding count from pack_trits

    Returns:
        Tensor of int8 with values in {-1, 0, +1}, reshaped to original shape
    """
    # Unpack each byte into 5 trits
    values = packed.to(torch.int32)
    trits = torch.zeros(packed.numel() * _TRITS_PER_BYTE, dtype=torch.int32,
                        device=packed.device)

    for i in range(_TRITS_PER_BYTE):
        trits[i::_TRITS_PER_BYTE] = values % 3
        values = values // 3

    # Remove padding
    total = math.prod(shape)
    trits = trits[:total]

    # Map back: 0 → -1, 1 → 0, 2 → +1
    return (trits - 1).to(torch.int8).reshape(shape)


# ─── Vectorized unpack (faster for large tensors) ───────────────────────────

def unpack_trits_vectorized(packed: Tensor, shape: tuple[int, ...], pad: int) -> Tensor:
    """Vectorized version of unpack_trits. ~3x faster for large tensors."""
    values = packed.to(torch.int32).unsqueeze(1).expand(-1, _TRITS_PER_BYTE)
    divisors = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32, device=packed.device)
    trits = (values // divisors) % 3
    trits = trits.reshape(-1)

    total = math.prod(shape)
    trits = trits[:total]
    return (trits - 1).to(torch.int8).reshape(shape)


# ─── Model serialization ────────────────────────────────────────────────────

# Parameters matching these patterns are kept in fp32 (control tensors)
CONTROL_PATTERNS = (
    "attn_scale", "mlp_scale", "resid_mix", "q_gain",
    "skip_weight", "skip_weights", "smear", "bigram.scale",
)

# Parameters matching these patterns are kept in fp16 (embeddings etc)
PASSTHROUGH_PATTERNS = ("tok_emb", "lm_head", "bigram.embed", "bigram.proj")


def is_ternary_param(name: str, tensor: Tensor) -> bool:
    """Determine if a parameter should be serialized as ternary."""
    # Skip small tensors
    if tensor.numel() <= 8192:
        return False
    # Skip control tensors
    if any(p in name for p in CONTROL_PATTERNS):
        return False
    # Skip embeddings/head
    if any(p in name for p in PASSTHROUGH_PATTERNS):
        return False
    # Only 2D weight matrices
    if tensor.ndim != 2:
        return False
    return True


def serialize_ternary_model(
    state_dict: dict[str, Tensor],
    per_row: bool = True,
) -> dict[str, Any]:
    """Serialize a model with ternary weights for the 16MB artifact.

    Args:
        state_dict: model state dict (with continuous weights from training)
        per_row: use per-row scales (recommended)

    Returns:
        Serializable dict with packed ternary weights + passthrough params
    """
    obj: dict[str, Any] = {
        "__format__": "ternary_packed_v1",
        "ternary": {},
        "scales": {},
        "metadata": {},
        "passthrough": {},
    }

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()

        if is_ternary_param(name, t):
            t_float = t.float()

            # Compute scale
            if per_row and t_float.ndim == 2:
                scale = t_float.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
            else:
                scale = t_float.abs().mean().clamp_min(1e-8)

            # Quantize to ternary
            ternary = torch.clamp(torch.round(t_float / scale), -1, 1).to(torch.int8)

            # Pack trits
            packed, shape, pad = pack_trits(ternary)

            obj["ternary"][name] = packed
            obj["scales"][name] = scale.to(torch.float16)
            obj["metadata"][name] = {"shape": list(shape), "pad": pad}

        elif any(p in name for p in CONTROL_PATTERNS):
            # Control tensors: keep fp32
            obj["passthrough"][name] = t.float()

        else:
            # Everything else (embeddings, small tensors): keep fp16
            if t.is_floating_point():
                obj["passthrough"][name] = t.to(torch.float16)
            else:
                obj["passthrough"][name] = t

    return obj


def deserialize_ternary_model(
    obj: dict[str, Any],
    template_sd: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Deserialize packed ternary model back to a state dict.

    Args:
        obj: dict from serialize_ternary_model (after torch.load)
        template_sd: reference state dict for dtype/shape info

    Returns:
        Reconstituted state dict with bf16/fp32 weights
    """
    out: dict[str, Tensor] = {}

    for name, orig in template_sd.items():
        if name in obj["ternary"]:
            # Unpack ternary
            packed = obj["ternary"][name]
            meta = obj["metadata"][name]
            shape = tuple(meta["shape"])
            pad = meta["pad"]

            ternary = unpack_trits_vectorized(packed, shape, pad)
            scale = obj["scales"][name].float()

            # Dequantize: ternary * scale → bf16
            if scale.ndim > 1:
                # Per-row: scale is (out_features, 1)
                dequant = (ternary.float() * scale).to(orig.dtype)
            else:
                dequant = (ternary.float() * scale.item()).to(orig.dtype)

            out[name] = dequant

        elif name in obj["passthrough"]:
            t = obj["passthrough"][name]
            if t.is_floating_point() and t.dtype != orig.dtype:
                out[name] = t.to(orig.dtype)
            else:
                out[name] = t
        else:
            raise KeyError(f"Parameter '{name}' not found in serialized model")

    return out


def compress_artifact(obj: dict[str, Any], level: int = 22) -> bytes:
    """Serialize and compress the ternary model artifact."""
    buf = io.BytesIO()
    torch.save(obj, buf)
    raw = buf.getvalue()

    if _HAS_ZSTD:
        return zstandard.ZstdCompressor(level=level).compress(raw)
    else:
        return zlib.compress(raw, 9)


def decompress_artifact(blob: bytes) -> dict[str, Any]:
    """Decompress and deserialize a ternary model artifact."""
    try:
        if _HAS_ZSTD:
            raw = zstandard.ZstdDecompressor().decompress(blob)
        else:
            raw = zlib.decompress(blob)
    except Exception:
        # Try the other compressor
        try:
            raw = zlib.decompress(blob)
        except Exception:
            if _HAS_ZSTD:
                raw = zstandard.ZstdDecompressor().decompress(blob)
            else:
                raise

    return torch.load(io.BytesIO(raw), map_location="cpu")


# ─── Roundtrip validation ───────────────────────────────────────────────────

def validate_roundtrip(state_dict: dict[str, Tensor], per_row: bool = True) -> dict[str, float]:
    """Validate serialize → compress → decompress → deserialize roundtrip.

    Returns dict with size info and max reconstruction error.
    """
    # Serialize
    obj = serialize_ternary_model(state_dict, per_row=per_row)

    # Compress
    blob = compress_artifact(obj)

    # Decompress + deserialize
    obj2 = decompress_artifact(blob)
    recon = deserialize_ternary_model(obj2, state_dict)

    # Check reconstruction
    max_err = 0.0
    ternary_params = 0
    passthrough_params = 0
    for name, orig in state_dict.items():
        if name in recon:
            err = (orig.float() - recon[name].float()).abs().max().item()
            max_err = max(max_err, err)
            if name in obj["ternary"]:
                ternary_params += orig.numel()
            else:
                passthrough_params += orig.numel()

    total_params = ternary_params + passthrough_params
    compressor = "zstd" if _HAS_ZSTD else "zlib"

    return {
        "compressed_bytes": len(blob),
        "compressed_mb": len(blob) / 1_000_000,
        "ternary_params": ternary_params,
        "passthrough_params": passthrough_params,
        "total_params": total_params,
        "bits_per_ternary_param": len(blob) * 8 / max(ternary_params, 1),
        "max_reconstruction_error": max_err,
        "compressor": compressor,
    }
