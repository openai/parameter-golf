"""
Mixed-precision GPTQ-lite: variable bit-width per layer, keeping clip search.

Drop-in replacement for mixed_quantize_int6(). Same clip percentile search,
same per-row scaling, but different bits per layer based on sensitivity.

One-line core change: clip_range = (1 << (bits-1)) - 1 instead of hardcoded 31.
"""

import torch
from torch import Tensor
from typing import Dict, Tuple

# ── Layer sensitivity classification ──

def get_bits_for_layer(name: str, num_layers: int) -> int:
    """Assign bit-width based on layer position and component type.

    Based on experimental data + Q-BERT + CLASE-Quant + Gemini/Opus/Grok consensus:
    - Embeddings: FP16 passthrough (handled separately)
    - Q/K attention in boundary layers (0,1,last): int8 (most sensitive)
    - Q/K attention in middle layers: int6
    - V/Out attention: int5
    - MLP down (error accumulator): int5
    - MLP up in boundary layers: int6
    - MLP up in middle layers: int4 (most compressible)
    """
    # Extract layer index
    layer_idx = -1
    parts = name.split('.')
    for p in parts:
        if p.isdigit():
            layer_idx = int(p)
            break

    sensitive_layers = {0, 1, num_layers - 1}
    is_sensitive = layer_idx in sensitive_layers

    # Q projections
    if 'c_q.weight' in name:
        return 8 if is_sensitive else 6

    # K projections
    if 'c_k.weight' in name:
        return 8 if is_sensitive else 6

    # V projections
    if 'c_v.weight' in name:
        return 6 if is_sensitive else 5

    # Output projections
    if 'proj.weight' in name and 'mlp' not in name:
        return 6 if is_sensitive else 5

    # MLP up (most compressible)
    if 'mlp.fc.weight' in name:
        return 6 if is_sensitive else 4

    # MLP down (error accumulator)
    if 'mlp.proj.weight' in name:
        return 6 if is_sensitive else 5

    # Default
    return 6


def quantize_per_row_variable(t: Tensor, bits: int = 6) -> Tuple[Tensor, Tensor]:
    """GPTQ-lite with variable bit-width. Same clip search, different clip_range."""
    clip_range = (1 << (bits - 1)) - 1  # bits=8->127, bits=6->31, bits=5->15, bits=4->7
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale


def mixed_quantize_adaptive(
    state_dict: Dict[str, Tensor],
    num_layers: int,
    control_patterns: tuple,
    fp16_names: set = None,
    log_fn=print,
) -> Tuple[Dict[str, Tensor], Dict]:
    """Drop-in replacement for mixed_quantize_int6.

    Same interface, same output format. Just smarter bit allocation.
    """
    if fp16_names is None:
        fp16_names = {"tok_emb.weight"}

    result: Dict[str, Tensor] = {}
    meta: Dict[str, object] = {}
    total_size = 0
    total_params = 0

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()

        # Non-float or tiny: passthrough
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            total_size += result[name].numel() * result[name].element_size()
            continue

        # Control tensors: float32 passthrough
        if control_patterns and any(p in name for p in control_patterns):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            total_size += result[name].numel() * 4
            continue

        # FP16 passthrough for embeddings
        if name in fp16_names:
            result[name] = t.to(torch.float16).contiguous()
            meta[name] = "passthrough_fp16"
            total_size += result[name].numel() * 2
            log_fn(f"  {name:45s} → FP16 passthrough")
            continue

        # Adaptive bit-width quantization with GPTQ-lite clip search
        bits = get_bits_for_layer(name, num_layers)
        q, s = quantize_per_row_variable(t, bits=bits)
        result[name + ".q"] = q
        result[name + ".scale"] = s

        layer_size = q.numel() * q.element_size() + s.numel() * s.element_size()
        total_size += layer_size
        total_params += t.numel()

        meta[name] = {"type": f"int{bits}", "bits": bits}
        log_fn(f"  {name:45s} → int{bits} ({layer_size:>9,} bytes)")

    log_fn(f"\n  Total quantized size: {total_size:,} bytes")
    log_fn(f"  Total params quantized: {total_params:,}")

    return result, meta


def dequantize_adaptive(
    result: Dict[str, Tensor],
    meta: Dict,
    template_sd: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Dequantize adaptive state dict. Same as dequantize_mixed_int6."""
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue

        orig_dtype = orig.dtype

        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype != orig_dtype:
                t = t.to(orig_dtype)
            out[name] = t
            continue

        if isinstance(info, dict) and "bits" in info:
            q = result[name + ".q"]
            s = result[name + ".scale"]
            if s.ndim > 0:
                out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
            else:
                out[name] = (q.float() * float(s.item())).to(orig_dtype)
            continue

        if name in result:
            out[name] = result[name]

    return out
