from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:  # pragma: no cover - older torch builds use the legacy API
    SDPBackend = None
    sdpa_kernel = None


def _expand_gqa_heads(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.size(-2) == k.size(-2):
        return q, k, v
    if q.size(-2) % k.size(-2) != 0:
        raise ValueError(
            f"Incompatible GQA head counts: q_heads={q.size(-2)} kv_heads={k.size(-2)}"
        )
    repeat = q.size(-2) // k.size(-2)
    return q, k.repeat_interleave(repeat, dim=-2), v.repeat_interleave(repeat, dim=-2)


def _math_sdpa_context():
    if not torch.cuda.is_available():
        return nullcontext()
    if sdpa_kernel is not None and SDPBackend is not None:
        return sdpa_kernel([SDPBackend.MATH])
    if hasattr(torch.backends.cuda, "sdp_kernel"):
        try:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
                enable_cudnn=False,
            )
        except TypeError:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            )
    return nullcontext()


def flash_attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    q, k, v = _expand_gqa_heads(q, k, v)
    q = q.permute(0, 2, 1, 3).contiguous()
    k = k.permute(0, 2, 1, 3).contiguous()
    v = v.permute(0, 2, 1, 3).contiguous()
    try:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    except RuntimeError as exc:
        msg = str(exc)
        if "Invalid backend" not in msg and "No available kernel" not in msg:
            raise
        with _math_sdpa_context():
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.permute(0, 2, 1, 3).contiguous()
