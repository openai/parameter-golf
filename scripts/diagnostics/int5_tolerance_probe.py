#!/usr/bin/env python3
"""Int5 tolerance probe for mixed-precision quantization feasibility.

Determines whether mixed int5/int6 naive quantization on 05c-plus can free
enough REAL compressed bytes (under custom-serialization + brotli-10) to
raise the MLP width ceiling above 3.15x.

Two input modes:
  --float-checkpoint  Full fidelity: float→int6 and float→int5 damage.
  --int6-artifact     Lower-fidelity proxy: incremental int6→int5 damage
                      from dequantized int6 weights.

Usage:
  python scripts/diagnostics/int5_tolerance_probe.py \\
    --float-checkpoint diagnostics/2026-03-31_05c_plus/final_model.pt \\
    --output-dir diagnostics/2026-03-31_int5_probe

  python scripts/diagnostics/int5_tolerance_probe.py \\
    --int6-artifact diagnostics/2026-03-31_05c_plus/final_model.int6.ptz \\
    --output-dir diagnostics/2026-03-31_int5_probe
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import struct
import time
from collections import defaultdict

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Optional compressors
# ---------------------------------------------------------------------------
try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import brotli
    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False

# ---------------------------------------------------------------------------
# Constants — match train_gpt.py (05c-plus) exactly
# ---------------------------------------------------------------------------
CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain",
    "skip_weight", "skip_weights", "smear",
    "ve_layer_scales", "ve_shared.scale",
)
INT8_CLIP_Q = 99.99984 / 100.0  # INT8_CLIP_PERCENTILE / 100

CAP = 16_000_000
MODEL_DIM = 512
CURRENT_MLP_MULT = 3.0
NUM_LAYERS = 11
BLOCK_GROUPS = [(0, 3), (4, 8), (9, 10)]


# ---------------------------------------------------------------------------
# Tensor classification — exact match of train_gpt.py
# ---------------------------------------------------------------------------
def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"


def is_int6_eligible(name: str, tensor: torch.Tensor) -> bool:
    """Would this tensor be quantized to int6 in the training export?"""
    if not tensor.is_floating_point() or tensor.numel() <= 65536:
        return False
    if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
        return False
    return _classify_param(name) in ("mlp", "attn")


def tensor_family(name: str) -> str:
    if name.startswith("bigram."):
        return "bigram"
    if name.startswith("ve_") or name.startswith("ve_shared."):
        return "ve"
    if ".attn." in name:
        return "attn"
    if ".mlp." in name:
        return "mlp"
    if "embed.weight" in name or "lm_head" in name:
        return "embeddings"
    return "other"


def block_index(name: str) -> int | None:
    m = re.match(r"blocks\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def block_group_label(name: str) -> str | None:
    idx = block_index(name)
    if idx is None:
        return None
    for lo, hi in BLOCK_GROUPS:
        if lo <= idx <= hi:
            return f"blocks.{lo}-{hi}"
    return f"blocks.{idx}"


# ---------------------------------------------------------------------------
# Quantizers — exact match of train_gpt.py semantics
# ---------------------------------------------------------------------------
def quantize_int6_per_row(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact match of train_gpt.py: range [-32,31], scale = row_max/31."""
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / 31.0).clamp_min(1.0 / 31.0).to(torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -32, 31).to(torch.int8)
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / 31.0 if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -32, 31).to(torch.int8)
    return q, scale


def quantize_int5_per_row(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Same pattern as int6 but range [-16, 15], scale = row_max/15."""
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / 15.0).clamp_min(1.0 / 15.0).to(torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -16, 15).to(torch.int8)
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / 15.0 if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -16, 15).to(torch.int8)
    return q, scale


def quantize_float_tensor(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Int8 with percentile clipping — exact match of train_gpt.py."""
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=torch.float16).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(
        torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127
    ).to(torch.int8).contiguous()
    return q, scale


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if scale.ndim > 0:
        return q.float() * scale.float().view(q.shape[0], *([1] * (q.ndim - 1)))
    return q.float() * float(scale.item())


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def _decompress_ptz(blob: bytes) -> bytes:
    if HAS_ZSTD:
        try:
            return zstandard.ZstdDecompressor().decompress(blob)
        except Exception:
            pass
    try:
        import zlib
        return zlib.decompress(blob)
    except Exception:
        pass
    raise RuntimeError("Cannot decompress .ptz (need zstandard or zlib)")


def load_float_checkpoint(path: str) -> dict[str, torch.Tensor]:
    print(f"Loading float checkpoint: {path}")
    sd = torch.load(path, map_location="cpu", weights_only=False)
    # Reject quantized checkpoints
    if "w" in sd and "m" in sd:
        first_v = next(iter(sd["m"].values()), None)
        if isinstance(first_v, dict) and "type" in first_v:
            raise ValueError(f"{path} looks quantized. Use --int6-artifact.")
    print(f"  Loaded {len(sd)} tensors")
    return sd


def load_int6_artifact(path: str) -> tuple[dict[str, torch.Tensor], dict, dict]:
    """Returns (dequantized_sd, quant_result, quant_meta)."""
    print(f"Loading int6 artifact (lower-fidelity proxy): {path}")
    with open(path, "rb") as f:
        blob = f.read()
    print(f"  File size: {len(blob):,} bytes")
    raw = _decompress_ptz(blob)
    qs = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    qr, qm = qs["w"], qs["m"]
    deq = {}
    for name, info in qm.items():
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            deq[name] = qr[name].float()
        elif name + ".q" in qr and name + ".scale" in qr:
            deq[name] = dequantize(qr[name + ".q"], qr[name + ".scale"])
    print(f"  Dequantized {len(deq)} tensors")
    return deq, qr, qm


# ---------------------------------------------------------------------------
# Per-tensor int5 vs int6 analysis
# ---------------------------------------------------------------------------
def analyze_tensor(name: str, float_tensor: torch.Tensor) -> dict:
    t = float_tensor.float()
    flat = t.reshape(-1)
    numel = flat.numel()
    if numel == 0:
        return {"name": name, "status": "empty"}

    norm_f = torch.linalg.norm(flat).item()
    rms_f = math.sqrt(flat.square().mean().item())

    q6, s6 = quantize_int6_per_row(t)
    d6 = dequantize(q6, s6).reshape(-1)
    q5, s5 = quantize_int5_per_row(t)
    d5 = dequantize(q5, s5).reshape(-1)

    def _m(deq, tag):
        delta = flat - deq
        mse = delta.square().mean().item()
        rmse = math.sqrt(mse)
        cos = float((flat * deq).sum() / (norm_f * torch.linalg.norm(deq) + 1e-12))
        return {
            f"mse_{tag}": mse,
            f"rmse_{tag}": rmse,
            f"rel_rmse_{tag}": rmse / (rms_f + 1e-12),
            f"cosine_{tag}": cos,
            f"zero_frac_{tag}": (deq == 0.0).float().mean().item(),
            f"sign_flip_{tag}": int(
                ((torch.sign(flat) != torch.sign(deq))
                 & (flat != 0) & (deq != 0)).sum().item()
            ),
        }

    m6 = _m(d6, "int6")
    m5 = _m(d5, "int5")

    mse_ratio = m5["mse_int5"] / (m6["mse_int6"] + 1e-20)
    added_rel = max(m5["rel_rmse_int5"] - m6["rel_rmse_int6"], 1e-12)
    raw_bytes = numel  # 1 byte per int8 value

    return {
        "name": name, "status": "ok",
        "shape": list(t.shape), "numel": numel,
        "family": tensor_family(name),
        "block_group": block_group_label(name),
        "block_idx": block_index(name),
        "raw_bytes": raw_bytes,
        **m6, **m5,
        "int5_mse_ratio": mse_ratio,
        "int5_extra_mse": m5["mse_int5"] - m6["mse_int6"],
        "int5_cosine_drop": m6["cosine_int6"] - m5["cosine_int5"],
        "int5_rel_rmse_ratio": m5["rel_rmse_int5"] / (m6["rel_rmse_int6"] + 1e-12),
        "added_rel_damage": added_rel,
        "utility_score": raw_bytes / added_rel,
    }


# ---------------------------------------------------------------------------
# Custom binary serialization (from compress_probe.py)
# ---------------------------------------------------------------------------
DTYPE_TO_STR = {
    torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16",
    torch.int8: "i8", torch.int16: "i16", torch.int32: "i32",
    torch.int64: "i64", torch.bool: "bool",
}
DTYPE_ELEM_SIZE = {
    "f32": 4, "f16": 2, "bf16": 2,
    "i8": 1, "i16": 2, "i32": 4, "i64": 8, "bool": 1,
}


def _byte_shuffle(data: bytes, elem_size: int) -> bytes:
    if elem_size <= 1:
        return data
    n = len(data) // elem_size
    arr = np.frombuffer(data, dtype=np.uint8).reshape(n, elem_size)
    return arr.T.copy().tobytes()


def _custom_pack(state_dict: dict[str, torch.Tensor], meta: dict,
                 shuffle: bool = True) -> bytes:
    header_entries = {}
    chunks = []
    offset = 0
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        ds = DTYPE_TO_STR.get(tensor.dtype)
        if ds is None:
            raise ValueError(f"Unsupported dtype {tensor.dtype} for {name}")
        raw = (tensor.numpy().tobytes() if tensor.dtype != torch.bfloat16
               else tensor.float().numpy().tobytes())
        es = DTYPE_ELEM_SIZE[ds]
        if shuffle and es > 1:
            raw = _byte_shuffle(raw, es)
        header_entries[name] = {
            "s": list(tensor.shape), "d": ds,
            "o": offset, "n": len(raw), "e": tensor.numel(),
        }
        chunks.append(raw)
        offset += len(raw)
    hdr = json.dumps({"t": header_entries, "m": meta}, separators=(",", ":")).encode()
    return struct.pack("<I", len(hdr)) + hdr + b"".join(chunks)


# ---------------------------------------------------------------------------
# Mixed quantization + compression measurement
# ---------------------------------------------------------------------------
def build_quant_from_float(
    float_sd: dict[str, torch.Tensor], int5_names: set[str],
) -> tuple[dict[str, torch.Tensor], dict]:
    """Full quantization from float, mirroring mixed_quantize_int6."""
    result: dict[str, torch.Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in float_sd.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if cat in ("mlp", "attn") and t.ndim >= 1:
            if name in int5_names:
                q, s = quantize_int5_per_row(t)
                meta[name] = {"type": "int5"}
            else:
                q, s = quantize_int6_per_row(t)
                meta[name] = {"type": "int6"}
            result[name + ".q"] = q
            result[name + ".scale"] = s
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta


def build_mixed_from_artifact(
    qr: dict, qm: dict, deq_sd: dict, int5_names: set[str],
) -> tuple[dict, dict]:
    """Swap selected int6 tensors to int5 in artifact state."""
    new_r = dict(qr)
    new_m = dict(qm)
    for name in int5_names:
        if name not in deq_sd:
            continue
        q5, s5 = quantize_int5_per_row(deq_sd[name])
        new_r[name + ".q"] = q5
        new_r[name + ".scale"] = s5
        new_m[name] = {"type": "int5"}
    return new_r, new_m


def compress_quant_state(qr: dict, qm: dict) -> int:
    """Pack + brotli-10 compress, return size in bytes. -1 if brotli missing."""
    if not HAS_BROTLI:
        return -1
    jm = {k: (v if isinstance(v, dict) else str(v)) for k, v in qm.items()}
    packed = _custom_pack(qr, jm, shuffle=True)
    return len(brotli.compress(packed, quality=10))


# ---------------------------------------------------------------------------
# Schedule generation
# ---------------------------------------------------------------------------
def _sched_quality(stats: list[dict]) -> dict:
    if not stats:
        return {"worst_cosine_int5": 1.0, "mean_cosine_int5": 1.0,
                "worst_mse_ratio": 1.0, "mean_mse_ratio": 1.0}
    return {
        "worst_cosine_int5": min(s["cosine_int5"] for s in stats),
        "mean_cosine_int5": sum(s["cosine_int5"] for s in stats) / len(stats),
        "worst_mse_ratio": max(s["int5_mse_ratio"] for s in stats),
        "mean_mse_ratio": sum(s["int5_mse_ratio"] for s in stats) / len(stats),
    }


def generate_schedules(tensor_stats: list[dict], total_int6_bytes: int) -> list[dict]:
    ok = [s for s in tensor_stats if s["status"] == "ok"]
    ranked = sorted(ok, key=lambda s: s.get("utility_score", 0), reverse=True)
    schedules = []

    # Fraction-based (by utility ranking)
    for tag, frac in [("conservative", 0.25), ("moderate", 0.50), ("aggressive", 0.75)]:
        target = total_int6_bytes * frac
        sel, cum = [], 0
        for s in ranked:
            if cum >= target:
                break
            sel.append(s["name"])
            cum += s["raw_bytes"]
        sel_s = [s for s in ok if s["name"] in set(sel)]
        schedules.append({
            "name": tag, "target_fraction": round(frac, 4),
            "tensor_count": len(sel), "tensor_names": sel,
            "int5_bytes": cum, "int6_bytes_remaining": total_int6_bytes - cum,
            **_sched_quality(sel_s),
        })

    # Per-family
    by_fam = defaultdict(list)
    for s in ok:
        by_fam[s["family"]].append(s)
    for fam, stats in sorted(by_fam.items()):
        byt = sum(s["raw_bytes"] for s in stats)
        schedules.append({
            "name": f"family_{fam}_all_int5",
            "target_fraction": round(byt / total_int6_bytes, 4) if total_int6_bytes else 0,
            "tensor_count": len(stats),
            "tensor_names": [s["name"] for s in stats],
            "int5_bytes": byt, "int6_bytes_remaining": total_int6_bytes - byt,
            **_sched_quality(stats),
        })

    # Per-block-group
    by_bg = defaultdict(list)
    for s in ok:
        bg = s.get("block_group")
        if bg:
            by_bg[bg].append(s)
    for bg, stats in sorted(by_bg.items()):
        byt = sum(s["raw_bytes"] for s in stats)
        schedules.append({
            "name": f"block_{bg}_int5",
            "target_fraction": round(byt / total_int6_bytes, 4) if total_int6_bytes else 0,
            "tensor_count": len(stats),
            "tensor_names": [s["name"] for s in stats],
            "int5_bytes": byt, "int6_bytes_remaining": total_int6_bytes - byt,
            **_sched_quality(stats),
        })

    return schedules


# ---------------------------------------------------------------------------
# Width estimation
# ---------------------------------------------------------------------------
def estimate_widths(baseline_sz: int, schedule_sz: int, code_bytes: int) -> dict:
    savings = baseline_sz - schedule_sz
    headroom = CAP - baseline_sz - code_bytes
    new_hr = headroom + savings

    def _mlp_raw(mult):
        h = int(mult * MODEL_DIM)
        per_layer = h * MODEL_DIM + h * 2 + MODEL_DIM * h + MODEL_DIM * 2
        return per_layer * NUM_LAYERS

    base_mlp = _mlp_raw(CURRENT_MLP_MULT)
    est_cr = 0.60  # conservative, from compress_probe measurements

    widths = {}
    for mult in [3.10, 3.15, 3.20, 3.25, 3.30, 3.50, 4.00]:
        extra = int((_mlp_raw(mult) - base_mlp) * est_cr)
        margin = new_hr - extra
        widths[f"{mult:.2f}x"] = {
            "hidden": int(mult * MODEL_DIM),
            "extra_compressed_est": extra,
            "fits": margin >= 0, "margin_bytes": margin,
        }
    return {"savings": savings, "current_headroom": headroom,
            "new_headroom": new_hr, "est_cr": est_cr, "widths": widths}


def theoretical_bit_packing(schedules: list[dict]) -> list[dict]:
    """Separate informational section: savings if bit-packing were implemented."""
    out = []
    for sc in schedules:
        i5 = sc["int5_bytes"]
        i6 = sc["int6_bytes_remaining"]
        out.append({
            "schedule": sc["name"],
            "raw_savings_int5_packed": int(i5 * 3 / 8),
            "raw_savings_both_packed": int(i5 * 3 / 8) + int(i6 * 2 / 8),
        })
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Wrote: {path} ({os.path.getsize(path):,} bytes)")


def write_markdown(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    L = []
    src = data["source_mode"]
    sm = data["summary"]

    L.append("# Int5 Tolerance Probe Results\n")
    L.append(f"**Source mode**: `{src}`  ")
    L.append(f"**Source path**: `{data['source_path']}`  ")
    L.append(f"**Timestamp**: {data['timestamp']}\n")

    if src == "int6-artifact-proxy":
        L.append("> **WARNING**: This analysis uses dequantized int6 weights as the")
        L.append("> float reference. Results are *incremental int6->int5 proxies*,")
        L.append("> NOT true float->int5 fidelity. The decision is based on a")
        L.append("> lower-fidelity proxy.\n")

    L.append("## Summary\n")
    L.append(f"- Int6-eligible tensors analyzed: **{sm['total_int6_tensors']}**")
    L.append(f"- Total int6 weight bytes: **{sm['total_int6_bytes']:,}**")
    if sm.get("baseline_compressed") is not None:
        bl = sm["baseline_compressed"]
        hr = CAP - bl - sm["code_bytes"]
        L.append(f"- Baseline compressed (all int6, custom+brotli-10): **{bl:,}** bytes")
        L.append(f"- Current headroom vs 16MB cap: **{hr:+,}** bytes")
    L.append("")

    # Family summary
    if data.get("family_summary"):
        L.append("## Per-family int5 damage\n")
        L.append("| Family | Tensors | Bytes | Mean MSE ratio | Worst cos(int5) | Mean cos(int5) |")
        L.append("|--------|---------|-------|----------------|-----------------|----------------|")
        for fam, fs in sorted(data["family_summary"].items()):
            L.append(f"| {fam} | {fs['tensors']} | {fs['bytes']:,} | "
                     f"{fs['mean_mse_ratio']:.2f} | {fs['worst_cosine']:.6f} | "
                     f"{fs['mean_cosine']:.6f} |")
        L.append("")

    # Block summary
    if data.get("block_summary"):
        L.append("## Per-block int5 damage\n")
        L.append("| Block | Tensors | Bytes | Mean MSE ratio | Worst cos(int5) |")
        L.append("|-------|---------|-------|----------------|-----------------|")
        for bg, bs in sorted(data["block_summary"].items()):
            L.append(f"| {bg} | {bs['tensors']} | {bs['bytes']:,} | "
                     f"{bs['mean_mse_ratio']:.2f} | {bs['worst_cosine']:.6f} |")
        L.append("")

    # Top candidates
    if data.get("top_candidates"):
        L.append("## Top 15 int5 candidates (by utility = bytes / added_damage)\n")
        L.append("| # | Tensor | Bytes | MSE ratio | Cos(int5) | Utility |")
        L.append("|---|--------|-------|-----------|-----------|---------|")
        for i, c in enumerate(data["top_candidates"][:15], 1):
            L.append(f"| {i} | `{c['name']}` | {c['raw_bytes']:,} | "
                     f"{c['int5_mse_ratio']:.2f} | {c['cosine_int5']:.6f} | "
                     f"{c['utility_score']:.0f} |")
        L.append("")

    # Schedules with compression results
    if data.get("schedules"):
        L.append("## Candidate schedules\n")
        for sc in data["schedules"]:
            pct = f"{sc['target_fraction']:.0%}" if sc['target_fraction'] else "N/A"
            L.append(f"### {sc['name']} ({sc['tensor_count']} tensors, {pct} of int6 bytes)\n")
            L.append(f"- Int5 bytes: {sc['int5_bytes']:,}")
            if sc.get("compressed_size") is not None:
                L.append(f"- Compressed size: **{sc['compressed_size']:,}** bytes")
                L.append(f"- **Real savings vs baseline: {sc.get('savings_vs_baseline', 0):+,} bytes**")
            if sc.get("width_estimate"):
                we = sc["width_estimate"]
                L.append(f"- New headroom: {we['new_headroom']:+,} bytes")
                L.append("- Width estimates:")
                for mk in ["3.10x", "3.15x", "3.20x", "3.25x", "3.30x", "3.50x"]:
                    wi = we["widths"].get(mk)
                    if wi:
                        st = "FITS" if wi["fits"] else "OVER"
                        L.append(f"  - MLP {mk}: **{st}** (margin {wi['margin_bytes']:+,})")
            L.append(f"- Quality: worst cos={sc['worst_cosine_int5']:.6f}, "
                     f"mean MSE ratio={sc['mean_mse_ratio']:.2f}")
            L.append("")

    # Theoretical bit-packing — SEPARATE SECTION
    if data.get("theoretical_bit_packing"):
        L.append("## Theoretical bit-packing (INFORMATIONAL ONLY)\n")
        L.append("> These estimates assume 5 bits/weight for int5, 6 bits/weight for int6,")
        L.append("> packed into fewer raw bytes. **Requires implementing a bit-packing")
        L.append("> export path.** If only bit-packing makes the fork attractive, the next")
        L.append("> step is \"bit-packing feasibility,\" NOT directly a training fork.\n")
        L.append("| Schedule | Raw savings (int5 packed) | Raw savings (both packed) |")
        L.append("|----------|--------------------------|--------------------------|")
        for bp in data["theoretical_bit_packing"]:
            L.append(f"| {bp['schedule']} | {bp['raw_savings_int5_packed']:,} | "
                     f"{bp['raw_savings_both_packed']:,} |")
        L.append("")

    # Decision
    L.append("## Decision assessment\n")
    if data.get("decision"):
        d = data["decision"]
        L.append(f"**Recommendation**: {d['recommendation']}\n")
        L.append(f"**Reasoning**: {d['reasoning']}\n")
        if d.get("bit_packing_note"):
            L.append(f"**Bit-packing note**: {d['bit_packing_note']}\n")
        if src == "int6-artifact-proxy":
            L.append("*Caveat: This decision is based on a lower-fidelity int6-artifact proxy. "
                     "Confirm with a float checkpoint run if the result is borderline.*\n")
    else:
        L.append("Run without `--limit` for compression measurements and automated assessment.\n")

    with open(path, "w") as f:
        f.write("\n".join(L))
    print(f"Wrote: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Int5 tolerance probe for mixed-precision quantization feasibility")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--float-checkpoint", help="Float checkpoint (.pt) — full fidelity")
    grp.add_argument("--int6-artifact", help="Int6 artifact (.ptz) — lower-fidelity proxy")
    parser.add_argument("--output-dir", default="diagnostics/2026-03-31_int5_probe")
    parser.add_argument("--code-bytes", type=int, default=69000)
    parser.add_argument("--limit", type=int, default=None,
                        help="Analyze only N tensors (skips compression measurement)")
    args = parser.parse_args()

    # Auto-detect .ptz passed as float checkpoint
    src = args.float_checkpoint or args.int6_artifact
    if args.float_checkpoint and src.endswith(".ptz"):
        print(f"NOTE: {src} is .ptz — switching to int6-artifact mode.")
        args.int6_artifact = args.float_checkpoint
        args.float_checkpoint = None

    # ---- Load ----
    float_sd = None
    qr_orig = qm_orig = deq_sd = None

    if args.float_checkpoint:
        source_mode = "float-checkpoint"
        float_sd = load_float_checkpoint(args.float_checkpoint)
        eligible = {n: t for n, t in float_sd.items() if is_int6_eligible(n, t)}
    else:
        source_mode = "int6-artifact-proxy"
        deq_sd, qr_orig, qm_orig = load_int6_artifact(args.int6_artifact)
        eligible = {}
        for name, info in qm_orig.items():
            if isinstance(info, dict) and info.get("type") == "int6" and name in deq_sd:
                eligible[name] = deq_sd[name]

    total_int6_bytes = sum(t.numel() for t in eligible.values())
    print(f"\nSource mode: {source_mode}")
    print(f"Int6-eligible tensors: {len(eligible)}")
    print(f"Total int6 weight bytes: {total_int6_bytes:,}")

    if args.limit:
        names = sorted(eligible.keys())[:args.limit]
        eligible = {k: eligible[k] for k in names}
        print(f"Limited to {len(eligible)} tensors (compression skipped)")

    # ---- Per-tensor analysis ----
    print(f"\nAnalyzing {len(eligible)} tensors...")
    tensor_stats = []
    for i, (name, tensor) in enumerate(sorted(eligible.items())):
        tensor_stats.append(analyze_tensor(name, tensor))
        if (i + 1) % 10 == 0 or i == len(eligible) - 1:
            print(f"  {i+1}/{len(eligible)}")

    ok = [s for s in tensor_stats if s["status"] == "ok"]
    print(f"  {len(ok)} tensors OK")

    # ---- Summaries ----
    by_fam = defaultdict(list)
    for s in ok:
        by_fam[s["family"]].append(s)
    family_summary = {
        fam: {
            "tensors": len(ss), "bytes": sum(s["raw_bytes"] for s in ss),
            "mean_mse_ratio": sum(s["int5_mse_ratio"] for s in ss) / len(ss),
            "worst_cosine": min(s["cosine_int5"] for s in ss),
            "mean_cosine": sum(s["cosine_int5"] for s in ss) / len(ss),
        } for fam, ss in sorted(by_fam.items())
    }

    by_bg = defaultdict(list)
    for s in ok:
        bg = s.get("block_group")
        if bg:
            by_bg[bg].append(s)
    block_summary = {
        bg: {
            "tensors": len(ss), "bytes": sum(s["raw_bytes"] for s in ss),
            "mean_mse_ratio": sum(s["int5_mse_ratio"] for s in ss) / len(ss),
            "worst_cosine": min(s["cosine_int5"] for s in ss),
        } for bg, ss in sorted(by_bg.items())
    }

    top_candidates = sorted(ok, key=lambda s: s.get("utility_score", 0), reverse=True)
    schedules = generate_schedules(tensor_stats, total_int6_bytes)

    # ---- Compression measurement (full mode only) ----
    baseline_compressed = None
    if args.limit is None and HAS_BROTLI:
        print("\nMeasuring compressed sizes (custom-shuffle + brotli-10)...")

        print("  Baseline (all int6)...")
        t0 = time.perf_counter()
        if float_sd is not None:
            b_qr, b_qm = build_quant_from_float(float_sd, set())
        else:
            b_qr, b_qm = qr_orig, qm_orig
        baseline_compressed = compress_quant_state(b_qr, b_qm)
        print(f"    {baseline_compressed:,} bytes ({time.perf_counter()-t0:.1f}s)")

        for sc in schedules:
            print(f"  Schedule '{sc['name']}' ({sc['tensor_count']} tensors)...")
            t0 = time.perf_counter()
            if float_sd is not None:
                m_qr, m_qm = build_quant_from_float(float_sd, set(sc["tensor_names"]))
            else:
                m_qr, m_qm = build_mixed_from_artifact(
                    qr_orig, qm_orig, deq_sd, set(sc["tensor_names"]))
            sz = compress_quant_state(m_qr, m_qm)
            sc["compressed_size"] = sz
            sc["savings_vs_baseline"] = baseline_compressed - sz
            sc["width_estimate"] = estimate_widths(
                baseline_compressed, sz, args.code_bytes)
            dt = time.perf_counter() - t0
            print(f"    {sz:,} bytes, savings={baseline_compressed - sz:+,} ({dt:.1f}s)")

    elif args.limit is None and not HAS_BROTLI:
        print("\nWARNING: brotli not available — skipping compression measurement")

    # ---- Theoretical bit-packing ----
    bp = theoretical_bit_packing(schedules)

    # ---- Decision ----
    decision = None
    if baseline_compressed is not None:
        best = None
        for sc in schedules:
            sav = sc.get("savings_vs_baseline", 0)
            cos = sc["worst_cosine_int5"]
            if sav > 0 and cos > 0.995:
                if best is None or sav > best.get("savings_vs_baseline", 0):
                    best = sc

        if best and best["savings_vs_baseline"] >= 100_000:
            we = best.get("width_estimate", {})
            max_mult = CURRENT_MLP_MULT
            for mk, wi in we.get("widths", {}).items():
                mv = float(mk.replace("x", ""))
                if wi["fits"] and mv > max_mult:
                    max_mult = mv
            if max_mult > 3.15:
                decision = {
                    "recommendation": "REOPEN mixed-bit width work",
                    "reasoning": (
                        f"Schedule '{best['name']}' saves {best['savings_vs_baseline']:,} "
                        f"real compressed bytes, pushing width to {max_mult:.2f}x "
                        f"(worst cosine={best['worst_cosine_int5']:.6f})."
                    ),
                }
            else:
                decision = {
                    "recommendation": "MARGINAL — consider bit-packing feasibility first",
                    "reasoning": (
                        f"Best schedule '{best['name']}' saves {best['savings_vs_baseline']:,} "
                        f"bytes but not enough to push above 3.15x under current export."
                    ),
                }
        elif best and best["savings_vs_baseline"] > 0:
            decision = {
                "recommendation": "STOP — go to 3.10x + brotli + MTP",
                "reasoning": (
                    f"Best acceptable schedule saves only {best['savings_vs_baseline']:,} "
                    f"bytes (<100 KB) — not enough to matter."
                ),
            }
        else:
            decision = {
                "recommendation": "STOP — go to 3.10x + brotli + MTP",
                "reasoning": "No schedule with acceptable quality (cosine>0.995) yields positive savings.",
            }

        # Bit-packing annotation
        if decision and not decision["recommendation"].startswith("REOPEN"):
            max_bp = max((b["raw_savings_both_packed"] for b in bp), default=0)
            if max_bp > 500_000:
                decision["bit_packing_note"] = (
                    f"Theoretical bit-packing could save up to {max_bp:,} raw bytes. "
                    f"Consider a bit-packing feasibility phase before fully committing to 3.10x."
                )

    # ---- Build output ----
    output = {
        "source_mode": source_mode,
        "source_path": args.float_checkpoint or args.int6_artifact,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_int6_tensors": len(ok),
            "total_int6_bytes": total_int6_bytes,
            "baseline_compressed": baseline_compressed,
            "code_bytes": args.code_bytes,
        },
        "tensor_stats": tensor_stats,
        "family_summary": family_summary,
        "block_summary": block_summary,
        "top_candidates": top_candidates[:20],
        "schedules": [{k: v for k, v in sc.items() if k != "tensor_names"}
                      for sc in schedules],
        "schedules_full": schedules,  # with tensor_names, for machine consumption
        "theoretical_bit_packing": bp,
        "decision": decision,
    }

    json_path = os.path.join(args.output_dir, "int5_tolerance_probe.json")
    md_path = os.path.join(args.output_dir, "int5_tolerance_probe.md")
    write_json(json_path, output)
    write_markdown(md_path, output)

    # ---- Console summary ----
    print("\n" + "=" * 70)
    print("INT5 TOLERANCE PROBE SUMMARY")
    print("=" * 70)
    print(f"Source: {source_mode}")
    print(f"Tensors: {len(ok)} analyzed, {total_int6_bytes:,} int6 bytes")
    if baseline_compressed:
        print(f"Baseline (custom+brotli-10): {baseline_compressed:,} bytes")
    print()

    if family_summary:
        print("Per-family:")
        for fam, fs in sorted(family_summary.items()):
            print(f"  {fam:8s}: {fs['tensors']:3d}t, mse_ratio={fs['mean_mse_ratio']:.2f}, "
                  f"worst_cos={fs['worst_cosine']:.6f}")
        print()

    if any(sc.get("compressed_size") is not None for sc in schedules):
        print("Real savings (custom+brotli-10):")
        for sc in schedules:
            if sc.get("compressed_size") is not None:
                print(f"  {sc['name']:30s}: {sc['savings_vs_baseline']:+8,} bytes "
                      f"(cos>={sc['worst_cosine_int5']:.4f})")
        print()

    if decision:
        print(f">>> {decision['recommendation']}")
        print(f"    {decision['reasoning']}")
        if decision.get("bit_packing_note"):
            print(f"    NOTE: {decision['bit_packing_note']}")

    print(f"\nOutputs: {json_path}")
    print(f"         {md_path}")


if __name__ == "__main__":
    main()
