#!/usr/bin/env python3
"""Reconstruct model.onnx.data from model.safetensors[.gz] + onnx_data_manifest.json.

This is the Python reference implementation of the JS-side reconstruction
that runs in the browser. Both must produce a byte-identical buffer for the
same inputs; we assert SHA-256 equality against the manifest's reference
hash (which was captured from the committed model.onnx.data when the
manifest was generated).

Usage:
  python reconstruct_onnx_data.py --model-dir visualization/public/models/ar

  # Optionally write the reconstructed .onnx.data to disk for ORT smoke
  # tests; not normally needed because the goal is in-memory reconstruction.
  python reconstruct_onnx_data.py --model-dir ... --write /tmp/onnx.data
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "visualization" / "scripts"))


def _read_safetensors(model_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Read either model.safetensors or model.safetensors.gz; return (state, metadata)."""
    raw_path = model_dir / "model.safetensors"
    gz_path = model_dir / "model.safetensors.gz"
    if raw_path.exists():
        path_to_open = raw_path
        cleanup = None
    elif gz_path.exists():
        # safe_open requires a file path, so we decompress to a tmp file.
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
        with gzip.open(gz_path, "rb") as src:
            tmp.write(src.read())
        tmp.flush()
        tmp.close()
        path_to_open = Path(tmp.name)
        cleanup = lambda: Path(tmp.name).unlink(missing_ok=True)
    else:
        raise FileNotFoundError(f"no model.safetensors or model.safetensors.gz in {model_dir}")
    try:
        with safe_open(str(path_to_open), framework="pt") as f:
            metadata = f.metadata() or {}
            state = {k: f.get_tensor(k) for k in f.keys()}
        return state, metadata
    finally:
        if cleanup:
            cleanup()


def dequantize(state: dict[str, torch.Tensor], metadata: dict[str, str]) -> dict[str, np.ndarray]:
    """Dequant safetensors → {name → fp32 ndarray}.

    Mirrors the byte-for-byte arithmetic the JS loader will perform.
    """
    bases: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in state.items():
        base, role = k.rsplit(".", 1)
        bases.setdefault(base, {})[role] = v
    out: dict[str, np.ndarray] = {}
    for base, parts in bases.items():
        if "p" in parts:
            out[base] = parts["p"].numpy().copy()
        elif "q" in parts and "s" in parts:  # AR int8 + fp16 scale
            q, s = parts["q"], parts["s"]
            f32 = q.float() * s.float().view(s.shape[0], *([1] * (q.ndim - 1)))
            target = metadata.get(f"target:{base}", "float32")
            if target != "float32":
                f32 = f32.to(getattr(torch, target)).to(torch.float32)
            out[base] = f32.numpy().astype(np.float32, copy=False).copy()
        elif "q" in parts:  # MDLM fp8_e4m3fn (exact via 256-entry LUT in JS)
            out[base] = parts["q"].to(torch.float32).numpy().copy()
        else:
            raise ValueError(f"bad parts for {base}: {sorted(parts.keys())}")
    return out


def reconstruct(model_dir: Path) -> bytes:
    manifest_path = model_dir / "onnx_data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    state, metadata = _read_safetensors(model_dir)
    deq = dequantize(state, metadata)

    buf = bytearray(manifest["total_bytes"])
    for entry in manifest["layout"]:
        kind = entry["kind"]
        offset = entry["offset"]
        length = entry["length"]
        if kind == "zeros":
            # bytearray is initialised to zeros — no-op write, but check bounds.
            assert offset + length <= len(buf)
            continue
        if kind == "gap":
            blob = base64.b64decode(entry["data_b64"])
            assert len(blob) == length, f"{kind} length mismatch at {offset}"
            buf[offset : offset + length] = blob
            continue
        if kind != "tensor":
            raise ValueError(f"unknown layout kind: {kind}")

        if entry["source"] == "constant":
            blob = base64.b64decode(entry["data_b64"])
            assert len(blob) == length, f"constant length mismatch for {entry['name']}"
            buf[offset : offset + length] = blob
        elif entry["source"] == "weight":
            arr = deq[entry["safetensors_base"]]
            transform = entry.get("transform", "identity")
            if transform == "identity":
                blob = arr.tobytes()
            elif transform == "transpose":
                if arr.ndim != 2:
                    raise ValueError(
                        f"transpose requested for non-2D tensor {entry['safetensors_base']} (ndim={arr.ndim})"
                    )
                blob = np.ascontiguousarray(arr.T).tobytes()
            else:
                raise ValueError(f"unknown transform: {transform}")
            assert len(blob) == length, (
                f"length mismatch for {entry['name']}: blob={len(blob)} expected={length}"
            )
            buf[offset : offset + length] = blob
        else:
            raise ValueError(f"unknown source: {entry['source']!r} on {entry['name']}")

    return bytes(buf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--write", help="optionally write reconstructed bytes to this path")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    manifest = json.loads((model_dir / "onnx_data_manifest.json").read_text())

    rebuilt = reconstruct(model_dir)
    rebuilt_sha = hashlib.sha256(rebuilt).hexdigest()
    print(f"  reconstructed {len(rebuilt):,} B  sha256={rebuilt_sha}")
    print(f"  manifest ref:   sha256={manifest['sha256']}")
    if rebuilt_sha != manifest["sha256"]:
        raise SystemExit("✗ reconstructed sha256 does not match the manifest reference")
    print("  ✓ matches manifest reference")

    # Cross-check against the actual on-disk model.onnx.data if it's present.
    data_path = model_dir / "model.onnx.data"
    if data_path.exists():
        actual_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
        print(f"  on-disk model.onnx.data sha256={actual_sha}")
        if actual_sha != rebuilt_sha:
            raise SystemExit("✗ reconstructed bytes don't match committed model.onnx.data")
        print("  ✓ matches committed model.onnx.data")

    if args.write:
        Path(args.write).write_bytes(rebuilt)
        print(f"  wrote {args.write}")


if __name__ == "__main__":
    main()
