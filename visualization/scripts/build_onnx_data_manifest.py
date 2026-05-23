#!/usr/bin/env python3
"""Build onnx_data_manifest.json that documents how model.onnx.data is laid out.

Why this file exists
--------------------
The browser's ONNX Runtime path consumes a fp32 byte buffer ("external data")
referenced by offsets in model.onnx. Today that buffer is `model.onnx.data`,
~65–75 MB on disk. The new pipeline keeps `model.safetensors.gz` (≤16 MB) as
the on-disk artifact and reconstructs the fp32 buffer at load time. Both
Python and JS need to produce a *byte-identical* buffer from the same
safetensors so that:

  SHA-256(JS-reconstructed onnx.data) ==
    SHA-256(Python-reconstructed onnx.data) ==
    SHA-256(committed model.onnx.data)

The reconstruction is straightforward: for each external initializer in
model.onnx, write its bytes at the exact offset model.onnx expects. The
manifest tells you, for every such initializer:

  - name, dims, dtype_enum (for sanity)
  - offset, length (where it lives in model.onnx.data)
  - source: "weight"   → bytes come from dequantizing safetensors[name],
                         optionally transformed (see "transform" field)
            "constant" → bytes are inlined as base64 below
                         (folded constants from torch.onnx.export, e.g.
                         RoPE caches, fused norm-residual scales)
  - if source=="weight":
      safetensors_base — the ".q/.s/.p" base name to dequant
      transform        — "identity" or "transpose"
                         (torch.onnx exports Linear weights transposed for
                          the ONNX MatMul convention; we record the
                          transform so the reconstruction is byte-exact.)
  - if source=="constant": data_b64 — the raw bytes verbatim

Plus the gap regions between initializers (torch.onnx.export emits aligned
blocks; observed gaps so far are either ≤ a few bytes of padding or ~50 KB
of zeros) — recorded as "gap" entries with the literal bytes, so we can
faithfully reproduce the file.

Generated once (or whenever model.onnx is regenerated). Committed as
public/models/<kind>/onnx_data_manifest.json.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "visualization" / "scripts"))

from pack_safetensors import dequantize_safetensors  # noqa: E402


def build(model_dir: Path, *, gap_max_bytes: int = 65536) -> dict:
    onnx_path = model_dir / "model.onnx"
    data_path = model_dir / "model.onnx.data"
    raw_path = model_dir / "model.safetensors"
    gz_path = model_dir / "model.safetensors.gz"
    if raw_path.exists():
        safetensors_path = raw_path
    elif gz_path.exists():
        safetensors_path = gz_path
    else:
        raise FileNotFoundError(
            f"need model.safetensors or model.safetensors.gz in {model_dir}"
        )

    model = onnx.load(str(onnx_path), load_external_data=False)
    data_bytes = data_path.read_bytes()
    deq = dequantize_safetensors(safetensors_path)

    # Build a content index: hash → (safetensors_base, transform)
    # torch.onnx.export commonly transposes 2-D Linear weights to fit the
    # ONNX MatMul A·B convention, so we register both orientations.
    by_hash: dict[str, tuple[str, str]] = {}
    for name, arr in deq.items():
        h_id = hashlib.sha256(arr.tobytes()).hexdigest()
        by_hash.setdefault(h_id, (name, "identity"))
        if arr.ndim == 2:
            t = np.ascontiguousarray(arr.T)
            by_hash.setdefault(hashlib.sha256(t.tobytes()).hexdigest(), (name, "transpose"))

    n_identity = 0
    n_transpose = 0
    n_constant = 0
    constant_bytes_total = 0

    entries: list[dict] = []
    for init in model.graph.initializer:
        ext = {kv.key: kv.value for kv in init.external_data}
        if not ext:
            continue  # inline initializer, lives in the protobuf, not the sidecar
        offset = int(ext.get("offset", 0))
        length = int(ext.get("length", 0))
        location = ext.get("location", "")
        if location != "model.onnx.data":
            raise ValueError(f"unexpected location={location!r} for {init.name}")
        chunk = data_bytes[offset : offset + length]
        if len(chunk) != length:
            raise ValueError(f"truncated read for {init.name}: got {len(chunk)} of {length}")

        chunk_hash = hashlib.sha256(chunk).hexdigest()
        if chunk_hash in by_hash:
            base_name, transform = by_hash[chunk_hash]
            src = {
                "source": "weight",
                "safetensors_base": base_name,
                "transform": transform,
            }
            if transform == "identity":
                n_identity += 1
            else:
                n_transpose += 1
        else:
            src = {"source": "constant", "data_b64": base64.b64encode(chunk).decode("ascii")}
            n_constant += 1
            constant_bytes_total += length

        entries.append({
            "name": init.name,
            "dims": list(init.dims),
            "dtype_enum": int(init.data_type),
            "offset": offset,
            "length": length,
            **src,
        })

    # Detect gaps (and the trailing tail) so reconstruction is byte-exact even
    # with torch's alignment padding. Sort by offset, then walk. Gaps so far
    # have always been all-zero — record them as kind="zeros" so the manifest
    # doesn't carry MBs of base64'd zeros. A gap with non-zero bytes would be
    # encoded as data_b64 and would imply we missed something — flag loudly.
    entries_sorted = sorted(entries, key=lambda e: e["offset"])
    layout: list[dict] = []
    cursor = 0

    def append_gap(start: int, blob: bytes) -> None:
        if not blob:
            return
        if all(b == 0 for b in blob):
            layout.append({"kind": "zeros", "offset": start, "length": len(blob)})
        else:
            if len(blob) > gap_max_bytes:
                raise ValueError(
                    f"non-zero gap {start}..{start+len(blob)} is {len(blob)} bytes, "
                    f"exceeds gap_max_bytes={gap_max_bytes}; investigate before inlining"
                )
            layout.append({
                "kind": "gap",
                "offset": start,
                "length": len(blob),
                "data_b64": base64.b64encode(blob).decode("ascii"),
            })

    for e in entries_sorted:
        if e["offset"] > cursor:
            append_gap(cursor, data_bytes[cursor : e["offset"]])
        layout.append({"kind": "tensor", **e})
        cursor = e["offset"] + e["length"]
    if cursor < len(data_bytes):
        append_gap(cursor, data_bytes[cursor:])

    manifest = {
        "version": 1,
        "comment": (
            "Documents the byte layout of model.onnx.data so it can be "
            "reconstructed deterministically from model.safetensors. Each "
            "tensor is either a 'weight' (bytes = dequantized safetensors entry "
            "named by safetensors_base) or a 'constant' (bytes = decoded "
            "data_b64). Gaps (alignment padding from torch.onnx.export) are "
            "recorded verbatim so reconstruction is byte-identical."
        ),
        "total_bytes": len(data_bytes),
        "sha256": hashlib.sha256(data_bytes).hexdigest(),
        "layout": layout,
    }
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    manifest = build(model_dir)
    out = model_dir / "onnx_data_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    n_layout = len(manifest["layout"])
    n_identity = sum(1 for e in manifest["layout"] if e.get("source") == "weight" and e.get("transform") == "identity")
    n_transpose = sum(1 for e in manifest["layout"] if e.get("source") == "weight" and e.get("transform") == "transpose")
    n_constant = sum(1 for e in manifest["layout"] if e.get("source") == "constant")
    n_gap = sum(1 for e in manifest["layout"] if e["kind"] == "gap")
    n_zeros = sum(1 for e in manifest["layout"] if e["kind"] == "zeros")
    inlined_b64_bytes = sum(e["length"] for e in manifest["layout"]
                            if e.get("source") == "constant" or e["kind"] == "gap")
    zeros_bytes = sum(e["length"] for e in manifest["layout"] if e["kind"] == "zeros")
    print(f"  wrote {out}  ({out.stat().st_size:,} B)")
    print(f"  layout: {n_layout} entries — {n_identity} identity-weight, "
          f"{n_transpose} transpose-weight, {n_constant} folded-constant, "
          f"{n_gap} non-zero gap, {n_zeros} zero-block")
    print(f"  inlined-b64 bytes (constants + non-zero gaps): {inlined_b64_bytes:,}")
    print(f"  zero-block bytes (length-only):                 {zeros_bytes:,}")
    print(f"  reference sha256: {manifest['sha256']}")
    print(f"  reference total:  {manifest['total_bytes']:,} bytes")


if __name__ == "__main__":
    main()
