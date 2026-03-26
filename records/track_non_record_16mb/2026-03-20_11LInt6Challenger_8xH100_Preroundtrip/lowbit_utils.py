from __future__ import annotations

import io
import zlib

import torch
from torch import Tensor

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - optional dependency
    zstd = None


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def quant_qmax(num_bits: int) -> int:
    if num_bits not in {5, 6, 7, 8}:
        raise ValueError(f"quantization bits must be one of (5, 6, 7, 8), got {num_bits}")
    return (1 << (num_bits - 1)) - 1


def _keep_fp32(name: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in name for pattern in patterns)


def resolve_num_bits(name: str, default_bits: int, bit_overrides: tuple[tuple[str, int], ...]) -> int:
    for pattern, num_bits in bit_overrides:
        if pattern and pattern in name:
            return num_bits
    return default_bits


def clipped_abs_max(t: Tensor, *, clip_q: float, dim: int | None = None) -> Tensor:
    # The competitive configs use clip_q extremely close to 1.0, where torch.quantile is much slower
    # than amax but produces nearly identical clipping thresholds. Switch to max in that regime so
    # export and roundtrip validation stay practical on 1x GPU proxy runs.
    if clip_q >= 0.9999:
        return t.abs().amax(dim=dim)
    if dim is None:
        return torch.quantile(t.abs().flatten(), clip_q)
    return torch.quantile(t.abs(), clip_q, dim=dim)


def keep_float_tensor(
    name: str,
    t: Tensor,
    passthrough_orig_dtypes: dict[str, str],
    *,
    keep_float_fp32_name_patterns: tuple[str, ...],
    keep_float_store_dtype: torch.dtype,
) -> Tensor:
    if _keep_fp32(name, keep_float_fp32_name_patterns):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=keep_float_store_dtype).contiguous()
    return t


def quantize_float_tensor(
    t: Tensor,
    *,
    num_bits: int,
    clip_q: float,
    per_row_scale_dtype: torch.dtype,
    grouped_int8: bool = False,
    group_size: int = 64,
) -> tuple[Tensor, Tensor, dict[str, object]]:
    qmax = quant_qmax(num_bits)
    t32 = t.float()
    if grouped_int8:
        if num_bits != 8 or t32.ndim != 2:
            raise ValueError("grouped_int8 requires 2D tensors with num_bits=8")
        rows, cols = t32.shape
        groups = (cols + group_size - 1) // group_size
        q = torch.empty_like(t32, dtype=torch.int8)
        scales = torch.empty((rows, groups), dtype=torch.float32)
        for g in range(groups):
            start = g * group_size
            end = min(start + group_size, cols)
            chunk = t32[:, start:end]
            clip_abs = clipped_abs_max(chunk, clip_q=clip_q, dim=1) if chunk.numel() else torch.zeros((rows,), dtype=torch.float32)
            scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
            clipped = torch.maximum(torch.minimum(chunk, clip_abs[:, None]), -clip_abs[:, None])
            q[:, start:end] = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8)
            scales[:, g] = scale
        return q.contiguous(), scales.to(dtype=per_row_scale_dtype).contiguous(), {"scheme": "per_group", "group_size": group_size}
    if t32.ndim == 2:
        clip_abs = clipped_abs_max(t32, clip_q=clip_q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / float(qmax)).clamp_min(1.0 / float(qmax))
        q = torch.clamp(torch.round(clipped / scale[:, None]), -qmax, qmax).to(torch.int8).contiguous()
        return q, scale.to(dtype=per_row_scale_dtype).contiguous(), {"scheme": "per_row"}
    clip_abs = float(clipped_abs_max(t32, clip_q=clip_q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / float(qmax) if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -qmax, qmax).to(torch.int8).contiguous()
    return q, scale, {"scheme": "per_tensor"}


def dequantize_tensor(q: Tensor, s: Tensor, dtype: torch.dtype, meta: dict[str, object]) -> Tensor:
    scheme = meta.get("scheme", "per_tensor")
    if scheme == "per_group":
        group_size = int(meta["group_size"])
        out = torch.empty_like(q, dtype=torch.float32)
        groups = s.shape[1]
        for g in range(groups):
            start = g * group_size
            end = min(start + group_size, q.shape[1])
            out[:, start:end] = q[:, start:end].float() * s[:, g].to(dtype=torch.float32)[:, None]
        return out.to(dtype=dtype).contiguous()
    if scheme == "per_row" or s.ndim > 0:
        return (q.float() * s.to(dtype=torch.float32).view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
    return (q.float() * float(s.item())).to(dtype=dtype).contiguous()


def quantize_state_dict(
    state_dict: dict[str, Tensor],
    *,
    weight_quant_bits: int,
    embed_quant_bits: int,
    lowbit_name_patterns: tuple[str, ...],
    keep_float_name_patterns: tuple[str, ...],
    grouped_int8_name_patterns: tuple[str, ...],
    group_size: int,
    keep_float_max_numel: int,
    keep_float_fp32_name_patterns: tuple[str, ...],
    keep_float_store_dtype: torch.dtype,
    per_row_scale_dtype: torch.dtype,
    clip_q: float,
    fp16_embed_export: bool,
    bit_overrides: tuple[tuple[str, int], ...] = (),
) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        force_keep = any(pattern in name for pattern in keep_float_name_patterns)
        embed_keep = (fp16_embed_export and name == "tok_emb.weight") or (name == "tok_emb.weight" and embed_quant_bits == 16)
        if force_keep or embed_keep or t.numel() <= keep_float_max_numel:
            kept = keep_float_tensor(
                name,
                t,
                passthrough_orig_dtypes,
                keep_float_fp32_name_patterns=keep_float_fp32_name_patterns,
                keep_float_store_dtype=keep_float_store_dtype,
            )
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        default_bits = embed_quant_bits if name == "tok_emb.weight" else (weight_quant_bits if any(pattern in name for pattern in lowbit_name_patterns) else 8)
        bits = resolve_num_bits(name, default_bits, bit_overrides)
        grouped = any(pattern in name for pattern in grouped_int8_name_patterns)
        q, s, meta = quantize_float_tensor(
            t,
            num_bits=bits,
            clip_q=clip_q,
            per_row_scale_dtype=per_row_scale_dtype,
            grouped_int8=grouped,
            group_size=group_size,
        )
        quantized[name] = q
        scales[name] = s
        qmeta[name] = {**meta, "bits": bits}
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": "lowbit_clean_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
        "qmeta": qmeta,
        "passthrough_orig_dtypes": passthrough_orig_dtypes,
    }
    return obj, stats


def dequantize_state_dict(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        out[name] = dequantize_tensor(q, obj["scales"][name], getattr(torch, obj["dtypes"][name]), qmeta.get(name, {}))
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def fake_quantize_tensor(t: Tensor, *, num_bits: int, clip_q: float) -> Tensor:
    if not t.is_floating_point():
        return t
    q, s, meta = quantize_float_tensor(
        t,
        num_bits=num_bits,
        clip_q=clip_q,
        per_row_scale_dtype=torch.float32,
    )
    return dequantize_tensor(q, s, t.dtype, meta).to(dtype=t.dtype)


def export_state_dict_without_mtp(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: tensor for name, tensor in state_dict.items() if "mtp_heads" not in name}


def load_export_state_dict(model, state_dict: dict[str, Tensor]) -> None:
    incompat = model.load_state_dict(state_dict, strict=False)
    non_mtp_missing = [k for k in incompat.missing_keys if "mtp_heads" not in k]
    unexpected = [k for k in incompat.unexpected_keys if "mtp_heads" not in k]
    if non_mtp_missing or unexpected:
        raise RuntimeError(f"Export state load mismatch missing={non_mtp_missing} unexpected={unexpected}")


def compress_quantized(obj: dict[str, object], compressor: str) -> tuple[bytes, int]:
    buf = io.BytesIO()
    torch.save(obj, buf)
    raw = buf.getvalue()
    if compressor == "zlib":
        return zlib.compress(raw, level=9), len(raw)
    if compressor == "zstd":
        if zstd is None:
            raise RuntimeError("zstandard is required for SERIAL_COMPRESSOR=zstd")
        return zstd.ZstdCompressor(level=22).compress(raw), len(raw)
    raise ValueError(f"Unsupported SERIAL_COMPRESSOR={compressor!r}")


def decompress_quantized(blob: bytes, compressor: str) -> dict[str, object]:
    if compressor == "zlib":
        return torch.load(io.BytesIO(zlib.decompress(blob)), map_location="cpu")
    if compressor == "zstd":
        if zstd is None:
            raise RuntimeError("zstandard is required for SERIAL_COMPRESSOR=zstd")
        return torch.load(io.BytesIO(zstd.ZstdDecompressor().decompress(blob)), map_location="cpu")
    raise ValueError(f"Unsupported SERIAL_COMPRESSOR={compressor!r}")
