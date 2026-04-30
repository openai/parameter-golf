#!/usr/bin/env python3
from __future__ import annotations

import argparse
import lzma
import pickle
import zlib
from pathlib import Path

import numpy as np


def softplus(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def binary_bits(arr: np.ndarray, center_mode: str, group_size: int) -> np.ndarray:
    if center_mode == "none":
        centered = arr
    elif center_mode == "group_mean" and arr.shape[-1] % group_size == 0:
        groups = arr.reshape(*arr.shape[:-1], arr.shape[-1] // group_size, group_size).astype(np.float32, copy=False)
        centered = (groups - np.mean(groups, axis=-1, keepdims=True)).reshape(arr.shape)
    elif center_mode == "group_mean":
        centered = arr
    else:
        raise ValueError(f"unsupported binary center mode: {center_mode}")
    return (centered.reshape(-1) >= 0).astype(np.uint8, copy=False)


def pack_binary_signs(arr: np.ndarray, center_mode: str, group_size: int) -> np.ndarray:
    bits = binary_bits(arr, center_mode, group_size)
    return np.packbits(bits, bitorder="little")


def normalize_ternary_groups(arr: np.ndarray, norm_mode: str, group_size: int) -> np.ndarray:
    if norm_mode == "none":
        return arr
    if arr.shape[-1] % group_size != 0:
        return arr
    groups = arr.reshape(*arr.shape[:-1], arr.shape[-1] // group_size, group_size).astype(np.float32, copy=False)
    if norm_mode == "group_rms":
        denom = np.sqrt(np.mean(groups * groups, axis=-1, keepdims=True) + 1e-8)
    elif norm_mode == "group_absmean":
        denom = np.mean(np.abs(groups), axis=-1, keepdims=True) + 1e-8
    else:
        raise ValueError(f"unsupported ternary norm mode: {norm_mode}")
    return (groups / denom).reshape(arr.shape)


def ternary_trits(arr: np.ndarray, threshold: float, norm_mode: str, group_size: int) -> np.ndarray:
    flat = normalize_ternary_groups(arr, norm_mode, group_size).reshape(-1)
    trits = np.zeros(flat.shape, dtype=np.uint8)
    trits[flat > threshold] = 2
    trits[flat < -threshold] = 0
    trits[(flat >= -threshold) & (flat <= threshold)] = 1
    return trits


def pack_ternary_trits(arr: np.ndarray, threshold: float, norm_mode: str, group_size: int) -> np.ndarray:
    trits = ternary_trits(arr, threshold, norm_mode, group_size)
    pad = (-trits.size) % 5
    if pad:
        trits = np.concatenate([trits, np.ones((pad,), dtype=np.uint8)])
    trits = trits.reshape(-1, 5).astype(np.uint16)
    packed = trits[:, 0] + 3 * trits[:, 1] + 9 * trits[:, 2] + 27 * trits[:, 3] + 81 * trits[:, 4]
    return packed.astype(np.uint8, copy=False)


def tensor_payload(
    name: str,
    arr: np.ndarray,
    arrays: dict[str, np.ndarray],
    mode: str,
    threshold: float,
    ternary_norm_mode: str,
    binary_center_mode: str,
    group_size: int,
) -> tuple[str, object]:
    if name.endswith(".weight_logits"):
        if mode == "binary":
            data = pack_binary_signs(arr, binary_center_mode, group_size)
            kind = "bin1"
        elif mode == "ternary":
            data = pack_ternary_trits(arr, threshold, ternary_norm_mode, group_size)
            kind = "tri5"
        else:
            raise ValueError(f"unsupported mode: {mode}")
        return name, {"kind": kind, "shape": arr.shape, "data": data}

    if name.endswith(".raw_scales"):
        scales = (softplus(arr) + 1e-6).astype(np.float16)
        return name, {"kind": "fp16", "shape": scales.shape, "data": scales}

    if np.issubdtype(arr.dtype, np.floating):
        fp16 = arr.astype(np.float16)
        return name, {"kind": "fp16", "shape": fp16.shape, "data": fp16}

    return name, {"kind": str(arr.dtype), "shape": arr.shape, "data": arr}


def packet_for_npz(path: Path, mode: str, threshold: float, ternary_norm_mode: str, binary_center_mode: str, group_size: int) -> dict[str, object]:
    z = np.load(path)
    tensors = {}
    for name in z.files:
        key, payload = tensor_payload(name, z[name], z, mode, threshold, ternary_norm_mode, binary_center_mode, group_size)
        tensors[key] = payload
    return {
        "format": f"native_{mode}_v1",
        "threshold": threshold,
        "ternary_norm_mode": ternary_norm_mode,
        "binary_center_mode": binary_center_mode,
        "group_size": group_size,
        "tensors": tensors,
    }


def ternary_counts(path: Path, threshold: float, ternary_norm_mode: str, group_size: int) -> tuple[int, int, int]:
    neg = zero = pos = 0
    z = np.load(path)
    for name in z.files:
        if not name.endswith(".weight_logits"):
            continue
        trits = ternary_trits(z[name], threshold, ternary_norm_mode, group_size)
        counts = np.bincount(trits, minlength=3)
        neg += int(counts[0])
        zero += int(counts[1])
        pos += int(counts[2])
    return neg, zero, pos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--mode", choices=("binary", "ternary"), default="binary")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--ternary-norm-mode", choices=("none", "group_rms", "group_absmean"), default="none")
    ap.add_argument("--binary-center-mode", choices=("none", "group_mean"), default="none")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--code", type=Path, default=Path("train_qwen_mlx.py"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    packet = packet_for_npz(args.npz, args.mode, args.threshold, args.ternary_norm_mode, args.binary_center_mode, args.group_size)
    raw = pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL)
    zlib_blob = zlib.compress(raw, level=9)
    lzma_blob = lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)

    code_bytes = args.code.stat().st_size if args.code.exists() else 0
    if args.out:
        args.out.write_bytes(lzma_blob)

    print(f"npz={args.npz}")
    print(f"mode={args.mode}")
    print(f"ternary_norm_mode={args.ternary_norm_mode}")
    print(f"binary_center_mode={args.binary_center_mode}")
    print(f"group_size={args.group_size}")
    if args.mode == "ternary":
        neg, zero, pos = ternary_counts(args.npz, args.threshold, args.ternary_norm_mode, args.group_size)
        total = max(neg + zero + pos, 1)
        print(f"ternary_neg_frac={neg / total:.6f}")
        print(f"ternary_zero_frac={zero / total:.6f}")
        print(f"ternary_pos_frac={pos / total:.6f}")
    print(f"raw_packet_bytes={len(raw)}")
    print(f"zlib_packet_bytes={len(zlib_blob)}")
    print(f"lzma_packet_bytes={len(lzma_blob)}")
    print(f"code_bytes={code_bytes}")
    print(f"total_zlib_with_code={len(zlib_blob) + code_bytes}")
    print(f"total_lzma_with_code={len(lzma_blob) + code_bytes}")
    print(f"limit_bytes=16000000")
    print(f"zlib_under_limit={len(zlib_blob) + code_bytes <= 16_000_000}")
    print(f"lzma_under_limit={len(lzma_blob) + code_bytes <= 16_000_000}")


if __name__ == "__main__":
    main()
