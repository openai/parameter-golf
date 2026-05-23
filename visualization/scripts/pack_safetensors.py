#!/usr/bin/env python3
"""Pack a .ptz artifact to .safetensors (+ optional gzip).

Replaces the fp32-expanding `convert_artifact.py` for distribution. The
goal is a single on-disk file ≤ 16 MiB that the browser can load and
losslessly dequantize to the same fp32 weights `convert_artifact.py`
used to write into `model.bin`.

Layout in the .safetensors file:

  AR (int8 per-row + fp16 scale, fp32 passthrough):
    "<name>.q"  int8     shape == weight
    "<name>.s"  float16  shape == (rows,)
    "<name>.p"  float32  passthrough tensors
    metadata["target:<name>"] = "bfloat16"  if dequant must round-trip
                                            through bf16 before fp32

  MDLM (raw float8_e4m3fn, fp32 passthrough):
    "<name>.q"  float8_e4m3fn   shape == weight
    "<name>.p"  float32         passthrough tensors

The metadata "kind" key disambiguates the two schemes for the loader.

Round-trip self-check: dequantize the new .safetensors and assert
byte-equality vs the existing public/models/<kind>/model.bin (which is
already known good — node_parity.js + cross_check_bpb_compare.py both
pass on those bytes).
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import lzma
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "text_diffusion"))

# Architecture constants — duplicated from convert_artifact.py so the new
# pipeline doesn't depend on the old script. These are model hyperparameters
# the JS forward needs to match (e.g., num_heads, rope_base).
MODEL_ARGS = dict(
    vocab_size=1024, num_heads=8, num_kv_heads=4, mlp_mult=2,
    logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
)


def _inflate_ptz(path: Path) -> dict:
    raw = path.read_bytes()
    try:
        data = zlib.decompress(raw)
    except zlib.error:
        data = lzma.decompress(raw)
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


def detect_arch(state: dict) -> dict:
    """Mirror experiments/text_diffusion/bpb_compare.py:detect_architecture.

    Looks at the dequantized state dict to figure out num_layers / model_dim /
    is_diffusion. We only need to peek at the int8 / fp8 quantized weights.
    """
    block_indices = set()
    for k in state:
        if k.startswith("blocks."):
            block_indices.add(int(k.split(".")[1]))
    num_layers = max(block_indices) + 1 if block_indices else 0
    emb = state["tok_emb.weight"]
    return {
        "num_layers": num_layers,
        "model_dim": emb.shape[1],
        "is_diffusion": emb.shape[0] > 1024,
    }


def pack(artifact_path: Path, out_path: Path) -> tuple[dict, dict]:
    obj = _inflate_ptz(artifact_path)
    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}

    if "quantized" in obj:  # int8 per-row scheme (AR baseline today)
        metadata["kind"] = "ar_int8_per_row"
        for name, q in obj["quantized"].items():
            assert q.dtype == torch.int8
            tensors[f"{name}.q"] = q.contiguous()
            scale = obj["scales"][name].to(torch.float16).contiguous()
            tensors[f"{name}.s"] = scale
            target = obj["dtypes"].get(name, "float32")
            if target != "float32":
                metadata[f"target:{name}"] = target
        for name, t in obj["passthrough"].items():
            tensors[f"{name}.p"] = t.to(torch.float32).contiguous()
        # Build arch by peeking at shapes (no need to dequant fully).
        peek = {**obj["quantized"], **obj["passthrough"]}
        arch_raw = detect_arch(peek)
    elif "entries" in obj:  # fp8_e4m3fn scheme (MDLM submission today)
        metadata["kind"] = "mdlm_fp8_e4m3fn"
        for name, entry in obj["entries"].items():
            t = entry["data"].contiguous()
            assert t.dtype == torch.float8_e4m3fn, (name, t.dtype)
            tensors[f"{name}.q"] = t
        for name, t in obj["passthrough"].items():
            tensors[f"{name}.p"] = t.to(torch.float32).contiguous()
        peek = {name: e["data"] for name, e in obj["entries"].items()}
        peek.update(obj["passthrough"])
        arch_raw = detect_arch(peek)
    else:
        raise ValueError(f"Unknown .ptz layout for {artifact_path} (keys: {sorted(obj.keys())})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata=metadata)

    arch = {
        "kind": "mdlm" if arch_raw["is_diffusion"] else "ar",
        "num_layers": arch_raw["num_layers"],
        "model_dim": arch_raw["model_dim"],
        **MODEL_ARGS,
        "mask_id": MODEL_ARGS["vocab_size"] if arch_raw["is_diffusion"] else None,
    }
    arch_doc = {
        "label": (f"MDLM {arch['num_layers']}L-{arch['model_dim']}d"
                  if arch_raw["is_diffusion"]
                  else f"AR {arch['num_layers']}L-{arch['model_dim']}d"),
        "arch": arch,
        "weights": "model.safetensors.gz",
    }
    return metadata, arch_doc


def dequantize_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Mirror what JS will do; produce {name -> fp32 ndarray} byte-equivalent
    to the dequantized state-dict produced by load_artifact() on the .ptz.

    Accepts either a raw .safetensors file or a .safetensors.gz; the .gz is
    decompressed to a temp file because safe_open requires a file path.
    """
    path = Path(path)
    cleanup = None
    if path.suffix == ".gz":
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
        with gzip.open(path, "rb") as src:
            tmp.write(src.read())
        tmp.flush()
        tmp.close()
        path = Path(tmp.name)
        cleanup = lambda: path.unlink(missing_ok=True)
    try:
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
            sd = {k: f.get_tensor(k) for k in f.keys()}
    finally:
        if cleanup:
            cleanup()

    bases: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        base, role = k.rsplit(".", 1)
        bases.setdefault(base, {})[role] = v

    out: dict[str, np.ndarray] = {}
    for base, parts in bases.items():
        if "p" in parts:
            t = parts["p"]
            assert t.dtype == torch.float32
            out[base] = t.numpy().copy()
        elif "q" in parts and "s" in parts:  # AR int8 + fp16 scale
            q, s = parts["q"], parts["s"]
            assert q.dtype == torch.int8 and s.dtype == torch.float16
            f32 = q.float() * s.float().view(s.shape[0], *([1] * (q.ndim - 1)))
            target = meta.get(f"target:{base}", "float32")
            if target != "float32":
                f32 = f32.to(getattr(torch, target)).to(torch.float32)
            out[base] = f32.numpy().astype(np.float32, copy=False).copy()
        elif "q" in parts:  # MDLM fp8 (e4m3fn → fp32 is exact)
            q = parts["q"]
            assert q.dtype == torch.float8_e4m3fn
            out[base] = q.to(torch.float32).numpy().copy()
        else:
            raise ValueError(f"bad parts for {base}: {sorted(parts.keys())}")
    return out


def assert_matches_model_bin(safetensors_path: Path, model_dir: Path) -> tuple[int, int]:
    manifest = json.loads((model_dir / "model.json").read_text())
    raw = (model_dir / "model.bin").read_bytes()
    deq = dequantize_safetensors(safetensors_path)
    ok = 0
    failures: list[tuple[str, str]] = []
    for entry in manifest["tensors"]:
        n = entry["nbytes"] // 4
        existing = np.frombuffer(raw, dtype=np.float32, count=n, offset=entry["offset"]).copy()
        ours = deq[entry["name"]].reshape(-1).astype(np.float32, copy=False)
        if existing.tobytes() == ours.tobytes():
            ok += 1
        else:
            d = np.abs(existing - ours)
            failures.append((entry["name"], f"max abs={d.max():.3e}  mean={d.mean():.3e}"))
    if failures:
        for name, why in failures[:10]:
            print(f"  FAIL {name}: {why}")
        raise AssertionError(f"{len(failures)}/{len(manifest['tensors'])} tensors differ from model.bin")
    return ok, len(manifest["tensors"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", required=True, help=".ptz path")
    p.add_argument("--out-dir", required=True,
                   help="visualization/public/models/<kind> — where to write model.safetensors[.gz]")
    p.add_argument("--no-gzip", action="store_true", help="skip the .gz step (debug)")
    p.add_argument("--gzip-level", type=int, default=9, choices=range(1, 10))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    raw_path = out_dir / "model.safetensors"
    print(f"packing {args.artifact}")
    metadata, arch_doc = pack(Path(args.artifact), raw_path)
    print(f"  wrote {raw_path}  ({raw_path.stat().st_size:,} B)")
    print(f"  metadata: {metadata}")
    print(f"  arch:     {arch_doc['label']}")
    arch_path = out_dir / "model.json"
    arch_path.write_text(json.dumps(arch_doc, indent=2))
    print(f"  wrote {arch_path}  ({arch_path.stat().st_size:,} B)")

    if not args.no_gzip:
        gz_path = out_dir / "model.safetensors.gz"
        # mtime=0, no filename, deterministic so the file's hash is stable.
        with open(raw_path, "rb") as fin, gzip.GzipFile(
            filename="", fileobj=open(gz_path, "wb"),
            mode="wb", compresslevel=args.gzip_level, mtime=0,
        ) as gz:
            gz.write(fin.read())
        sz = gz_path.stat().st_size
        limit = 16 * 1024 * 1024
        flag = "✓" if sz <= limit else "✗ OVER 16 MiB"
        print(f"  wrote {gz_path}  ({sz:,} B = {sz/limit*100:.2f}% of 16 MiB) {flag}")
        # The .gz is the artifact of record; drop the uncompressed copy.
        raw_path.unlink()


if __name__ == "__main__":
    main()
