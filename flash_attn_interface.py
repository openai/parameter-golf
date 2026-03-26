"""FlashAttention compatibility and runtime backend selection.

The frontier trainers prefer FlashAttention when the runtime tensors and device
actually satisfy its requirements. Otherwise they fall back to PyTorch SDP math.
"""

from __future__ import annotations

import contextlib
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


try:
    from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func

    _FLASH_ATTN_SOURCE = "flash_attn.flash_attn_interface"
    _FLASH_ATTN_IMPORT_ERROR = None
except ImportError:
    try:
        from flash_attn import flash_attn_func as _flash_attn_func

        _FLASH_ATTN_SOURCE = "flash_attn"
        _FLASH_ATTN_IMPORT_ERROR = None
    except ImportError as exc:  # pragma: no cover - depends on installed flash-attn version
        _flash_attn_func = None
        _FLASH_ATTN_SOURCE = None
        _FLASH_ATTN_IMPORT_ERROR = exc


_ATTENTION_LOG_FN: Callable[[str], None] | None = None
_LOGGED_MESSAGES: set[str] = set()


def configure_attention_logging(log_fn: Callable[[str], None] | None) -> None:
    global _ATTENTION_LOG_FN
    _ATTENTION_LOG_FN = log_fn


def flash_attention_import_summary() -> dict[str, object]:
    return {
        "flash_attn_available": _flash_attn_func is not None,
        "flash_attn_source": _FLASH_ATTN_SOURCE,
        "flash_attn_import_error": None if _FLASH_ATTN_IMPORT_ERROR is None else str(_FLASH_ATTN_IMPORT_ERROR),
    }


def _log_once(message: str) -> None:
    if _ATTENTION_LOG_FN is None or message in _LOGGED_MESSAGES:
        return
    _LOGGED_MESSAGES.add(message)
    _ATTENTION_LOG_FN(message)


def _dtype_supported_for_flash(dtype: torch.dtype) -> bool:
    return dtype in {torch.float16, torch.bfloat16}


def _flash_attention_eligibility(q: Tensor, k: Tensor, v: Tensor) -> tuple[bool, str]:
    if _flash_attn_func is None:
        return False, "flash_attn_import_unavailable"
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False, "attention_tensors_not_on_cuda"
    if q.dtype != k.dtype or q.dtype != v.dtype:
        return False, f"mixed_attention_dtypes:q={q.dtype} k={k.dtype} v={v.dtype}"
    if not _dtype_supported_for_flash(q.dtype):
        return False, f"unsupported_attention_dtype:{q.dtype}"
    major, minor = torch.cuda.get_device_capability(q.device)
    if major < 8:
        return False, f"unsupported_compute_capability:sm_{major}{minor}"
    return True, "ok"


def _math_sdp_context():
    attention_mod = getattr(torch.nn, "attention", None)
    if attention_mod is not None and hasattr(attention_mod, "sdpa_kernel") and hasattr(attention_mod, "SDPBackend"):
        return attention_mod.sdpa_kernel(attention_mod.SDPBackend.MATH)
    sdp_kernel = getattr(torch.backends.cuda, "sdp_kernel", None)
    if sdp_kernel is None:
        return contextlib.nullcontext()
    try:
        return sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False, enable_cudnn=False)
    except TypeError:
        return sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)


def _expand_gqa_heads(q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    q_heads = q.size(-2)
    kv_heads = k.size(-2)
    if kv_heads != v.size(-2):
        raise RuntimeError(f"Mismatched KV heads for attention fallback: k={kv_heads} v={v.size(-2)}")
    if q_heads == kv_heads:
        return q, k, v
    if q_heads % kv_heads != 0:
        raise RuntimeError(f"Query heads must be divisible by KV heads for attention fallback: q={q_heads} kv={kv_heads}")
    group = q_heads // kv_heads
    return q, k.repeat_interleave(group, dim=-2), v.repeat_interleave(group, dim=-2)


def _sdp_math_attention(q: Tensor, k: Tensor, v: Tensor, *, enable_gqa: bool) -> Tensor:
    if enable_gqa:
        q, k, v = _expand_gqa_heads(q, k, v)
    q_sdp = q.transpose(1, 2)
    k_sdp = k.transpose(1, 2)
    v_sdp = v.transpose(1, 2)
    with _math_sdp_context():
        y = F.scaled_dot_product_attention(
            q_sdp,
            k_sdp,
            v_sdp,
            attn_mask=None,
            is_causal=True,
        )
    return y.transpose(1, 2).contiguous()


def causal_attention(q: Tensor, k: Tensor, v: Tensor, *, enable_gqa: bool) -> Tensor:
    can_use_flash, reason = _flash_attention_eligibility(q, k, v)
    if can_use_flash:
        try:
            y = _flash_attn_func(q, k, v, causal=True)
            _log_once(f"attention_backend:flash_attn attention_dtype:{q.dtype}")
            return y
        except RuntimeError as exc:
            reason = f"flash_attn_runtime_error:{str(exc).strip().replace(chr(10), ' ')}"
    _log_once(f"attention_backend:sdp_math attention_dtype:{q.dtype} fallback_reason:{reason}")
    return _sdp_math_attention(q, k, v, enable_gqa=enable_gqa)


flash_attn_func = _flash_attn_func

__all__ = [
    "causal_attention",
    "configure_attention_logging",
    "flash_attention_import_summary",
    "flash_attn_func",
]
