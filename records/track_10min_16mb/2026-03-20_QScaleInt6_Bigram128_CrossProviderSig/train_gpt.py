"""
PR135 export variant with quantized per-row scales.

The training recipe is unchanged. We only compress the mixed int6 export more
aggressively by quantizing large per-row scale vectors into uint8 log-space.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parent
BASE_PATH = ROOT / "pr135_base_train_gpt.py"
SPEC = importlib.util.spec_from_file_location("pr135_base_qscale", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Unable to load base trainer from {BASE_PATH}")
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)


def quantize_scale_vector(scale: Tensor) -> tuple[Tensor, Tensor]:
    s = scale.float().clamp_min(1e-12)
    log_s = s.log()
    lo = log_s.min()
    hi = log_s.max()
    if float((hi - lo).abs()) < 1e-12:
        q = torch.zeros_like(s, dtype=torch.uint8)
    else:
        q = torch.clamp(torch.round((log_s - lo) * (255.0 / (hi - lo))), 0, 255).to(torch.uint8)
    return q.contiguous(), torch.stack((lo, hi)).to(torch.float16).contiguous()


def dequantize_scale_vector(scale_q: Tensor, scale_meta: Tensor) -> Tensor:
    lo, hi = scale_meta.float()
    if float((hi - lo).abs()) < 1e-12:
        return torch.full(scale_q.shape, torch.exp(lo).item(), dtype=torch.float32)
    return torch.exp(lo + (scale_q.float() / 255.0) * (hi - lo))


def _store_scale(result: dict[str, Tensor], name: str, scale: Tensor, kind: str) -> dict[str, str]:
    if scale.ndim > 0 and scale.numel() > 32:
        scale_q, scale_meta = quantize_scale_vector(scale)
        result[name + ".scale_q"] = scale_q
        result[name + ".scale_meta"] = scale_meta
        return {"type": kind + "_qscale"}
    result[name + ".scale"] = scale
    return {"type": kind}


def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str]):
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = base._classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in base.CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if any(pattern in name for pattern in base.FP16_KEEP_NAME_PATTERNS):
            result[name] = t.to(dtype=torch.float16).contiguous()
            meta[name] = "passthrough_fp16"
            continue
        if cat in int6_cats and t.ndim >= 1:
            q, s = base.quantize_int6_per_row(t)
            result[name + ".q"] = q
            meta[name] = _store_scale(result, name, s, "int6")
        else:
            q, s = base.quantize_float_tensor(t)
            result[name + ".q"] = q
            meta[name] = _store_scale(result, name, s, "int8")
    return result, meta


def dequantize_mixed_int6(result: dict[str, Tensor], meta: dict[str, object], template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta[name]
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q = result[name + ".q"]
        scale_type = info.get("type") if isinstance(info, dict) else None
        if scale_type in {"int6_qscale", "int8_qscale"}:
            s = dequantize_scale_vector(result[name + ".scale_q"], result[name + ".scale_meta"])
        else:
            s = result[name + ".scale"]
        if isinstance(s, Tensor) and s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            scalar = float(s.item()) if isinstance(s, Tensor) else float(s)
            out[name] = (q.float() * scalar).to(orig_dtype)
    return out


base.mixed_quantize_int6 = mixed_quantize_int6
base.dequantize_mixed_int6 = dequantize_mixed_int6


if __name__ == "__main__":
    base.main()
