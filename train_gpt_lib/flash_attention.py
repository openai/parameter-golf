"""FlashAttention-2 / FlashAttention-3 wrapper.

Usage (in CausalSelfAttention.forward):
    from .flash_attention import get_flash_attn_fn, FLASH_ATTN_VERSION
    fn = get_flash_attn_fn(version=3)   # or version=2
    out = fn(q, k, v, causal=True)      # q/k/v in (B, S, H, D) layout

Layout note
-----------
flash_attn expects tensors in **(B, S, H, D)** (sequence dim second), while
PyTorch's ``scaled_dot_product_attention`` uses **(B, H, S, D)** (heads
second).  The attention module keeps tensors in flash_attn's layout when
flash_attn is active to avoid extra transposes.

FlashAttention-3
----------------
``flash_attn.flash_attn_interface.flash_attn_func`` is the Hopper-optimised
FA3 kernel path (only beneficial on SM90+ / H100).  The API is identical to
the FA2 function so we can swap them transparently.

GQA
---
Both FA2 and FA3 handle GQA natively: pass q with ``num_heads`` heads and
k/v with ``num_kv_heads`` heads — no manual expansion needed.
"""
from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------

_FA2_FN = None
_FA3_FN = None

try:
    from flash_attn import flash_attn_func as _fa2  # noqa: F401

    _FA2_FN = _fa2
except ImportError:
    pass

try:
    # FA3 Hopper kernels — available in flash-attn >= 2.7 on SM90
    from flash_attn.flash_attn_interface import flash_attn_func as _fa3  # noqa: F401

    # Only use FA3 if actually running on Hopper (SM90+)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
        _FA3_FN = _fa3
except ImportError:
    pass


def is_available(version: int = 2) -> bool:
    """Return True if the requested flash_attn version is importable and runnable."""
    if version == 3:
        return _FA3_FN is not None
    return _FA2_FN is not None


def get_flash_attn_fn(version: int = 2):
    """Return the flash_attn callable for the requested version.

    The returned function has the signature::

        fn(q, k, v, causal=True) -> Tensor

    where q/k/v are in **(B, S, H, D)** layout (bfloat16 or float16).

    Parameters
    ----------
    version:
        2  — FlashAttention-2 (``flash_attn.flash_attn_func``)
        3  — FlashAttention-3 Hopper kernel (``flash_attn.flash_attn_interface.flash_attn_func``).
             Automatically falls back to FA2 if not on SM90.

    Raises
    ------
    RuntimeError
        If the requested version is not installed.
    """
    if version == 3:
        if _FA3_FN is not None:
            return _fa3_wrapper
        if _FA2_FN is not None:
            # Graceful fallback: FA2 on non-Hopper GPU
            return _fa2_wrapper
        raise RuntimeError(
            "FlashAttention-3 requested but flash_attn is not installed. "
            "Run: pip install flash-attn"
        )
    if version == 2:
        if _FA2_FN is not None:
            return _fa2_wrapper
        raise RuntimeError(
            "FlashAttention-2 requested but flash_attn is not installed. "
            "Run: pip install flash-attn"
        )
    raise ValueError(f"Unknown flash_attn version={version}. Use 2 or 3.")


def _fa2_wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """Call FA2 with (B, S, H, D) inputs, return (B, S, H, D)."""
    assert _FA2_FN is not None
    return _FA2_FN(q, k, v, causal=causal)


def _fa3_wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """Call FA3 Hopper kernel with (B, S, H, D) inputs, return (B, S, H, D)."""
    assert _FA3_FN is not None
    return _FA3_FN(q, k, v, causal=causal)


def describe() -> str:
    """Human-readable string describing what's available."""
    parts = []
    if _FA3_FN is not None:
        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else "?"
        parts.append(f"FA3(Hopper SM{cap[0]}{cap[1]})")
    if _FA2_FN is not None:
        parts.append("FA2")
    if not parts:
        parts.append("unavailable (flash-attn not installed)")
    return " ".join(parts)
