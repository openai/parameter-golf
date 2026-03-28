from __future__ import annotations

import pickle

import mlx.core as mx
import numpy as np
import zlib


KEEP_FLOAT_MAX_NUMEL = 65_536
SCALE_STORE_DTYPE = np.float16
CLIP_PERCENTILE = 99.99984
CLIP_Q = CLIP_PERCENTILE / 100.0


def bits_per_token_from_loss(token_loss_nats: float) -> float:
    import math

    return token_loss_nats / math.log(2.0)


def _np_float32(arr: mx.array) -> np.ndarray:
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)


def _quantize_float_array(arr: mx.array, bits: int) -> tuple[np.ndarray, np.ndarray]:
    f32 = _np_float32(arr)
    qmax = max((1 << (bits - 1)) - 1, 1)
    if f32.ndim == 2:
        clip_abs = np.quantile(np.abs(f32), CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / qmax, 1.0 / qmax).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -qmax, qmax).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(SCALE_STORE_DTYPE, copy=False))

    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / qmax if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -qmax, qmax).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale.astype(SCALE_STORE_DTYPE, copy=False)


def _dequantize_float_array(q: np.ndarray, scale: np.ndarray, dtype: mx.Dtype) -> mx.array:
    if scale.ndim > 0:
        out_arr = q.astype(np.float32) * scale.astype(np.float32).reshape((q.shape[0],) + (1,) * (q.ndim - 1))
    else:
        out_arr = q.astype(np.float32) * float(scale.astype(np.float32))
    return mx.array(out_arr, dtype=dtype)


def estimate_trainable_payload_bytes(full_state: dict[str, mx.array], trainable_names: set[str]) -> float:
    return sum(float(full_state[name].nbytes) for name in trainable_names)


def quantize_trainable_params(
    full_state: dict[str, mx.array],
    trainable_names: set[str],
    bits: int,
) -> tuple[dict[str, mx.array], dict[str, float]]:
    updated = dict(full_state)
    payload_bytes = 0.0
    quantized_params = 0
    kept_params = 0
    for name in trainable_names:
        arr = full_state[name]
        if not mx.issubdtype(arr.dtype, mx.floating):
            payload_bytes += float(arr.nbytes)
            continue
        if int(arr.size) <= KEEP_FLOAT_MAX_NUMEL:
            kept = np.array(arr.astype(mx.float16), dtype=np.float16, copy=False)
            updated[name] = mx.array(kept, dtype=arr.dtype)
            payload_bytes += float(kept.nbytes)
            kept_params += int(arr.size)
            continue
        q, scale = _quantize_float_array(arr, bits)
        updated[name] = _dequantize_float_array(q, scale, arr.dtype)
        payload_bytes += float((arr.size * bits) / 8.0 + scale.nbytes)
        quantized_params += int(arr.size)
    stats = {
        "bits": float(bits),
        "payload_bytes_est": payload_bytes,
        "payload_mb_est": payload_bytes / (1024.0 * 1024.0),
        "quantized_params": quantized_params,
        "kept_params": kept_params,
    }
    return updated, stats


def pack_trainable_params(
    full_state: dict[str, mx.array],
    trainable_names: set[str],
    bits: int,
) -> tuple[dict[str, object], dict[str, float]]:
    packed: dict[str, object] = {}
    payload_bytes = 0.0
    quantized_params = 0
    kept_params = 0
    for name, arr in full_state.items():
        if name not in trainable_names:
            continue
        if not mx.issubdtype(arr.dtype, mx.floating):
            raw = np.array(arr, copy=False)
            packed[name] = {
                "type": "raw",
                "dtype": str(arr.dtype),
                "data": raw,
            }
            payload_bytes += float(raw.nbytes)
            continue
        if int(arr.size) <= KEEP_FLOAT_MAX_NUMEL:
            kept = np.array(arr.astype(mx.float16), dtype=np.float16, copy=False)
            packed[name] = {
                "type": "fp16",
                "data": kept,
            }
            payload_bytes += float(kept.nbytes)
            kept_params += int(arr.size)
            continue
        q, scale = _quantize_float_array(arr, bits)
        packed[name] = {
            "type": "quant",
            "bits": int(bits),
            "q": q,
            "scale": scale,
            "dtype": str(arr.dtype),
        }
        payload_bytes += float((arr.size * bits) / 8.0 + scale.nbytes)
        quantized_params += int(arr.size)
    stats = {
        "bits": float(bits),
        "payload_bytes_est": payload_bytes,
        "payload_mb_est": payload_bytes / (1024.0 * 1024.0),
        "quantized_params": quantized_params,
        "kept_params": kept_params,
    }
    return packed, stats


def dequantize_packed_params(packed: dict[str, object]) -> dict[str, mx.array]:
    restored: dict[str, mx.array] = {}
    for name, entry_obj in packed.items():
        entry = dict(entry_obj)
        kind = entry["type"]
        if kind == "quant":
            q = np.array(entry["q"], copy=False)
            scale = np.array(entry["scale"], copy=False)
            restored[name] = _dequantize_float_array(q, scale, mx.float32)
        elif kind == "fp16":
            data = np.array(entry["data"], copy=False)
            restored[name] = mx.array(data.astype(np.float32, copy=False), dtype=mx.float32)
        elif kind == "raw":
            restored[name] = mx.array(np.array(entry["data"], copy=False))
        else:
            raise ValueError(f"Unknown packed param type: {kind}")
    return restored


def serialize_packed_params_zlib(
    packed: dict[str, object],
    level: int = 9,
) -> tuple[bytes, int]:
    raw = pickle.dumps(packed, protocol=pickle.HIGHEST_PROTOCOL)
    return zlib.compress(raw, level=level), len(raw)
