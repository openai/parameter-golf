"""Strict FlashAttention wrapper for the April 20 LaCT record path.

This folder intentionally does not provide any SDPA/math fallback. Priority and
record runs must use the same FlashAttention installation family as the April 9
record. If the backend is missing, fail immediately with an explicit message.
"""

from __future__ import annotations

FLASH_ATTN_INSTALL_HINT = (
    "FlashAttention is required for this record path. Install it with the same "
    "command used by the April 9 record:\n"
    "pip install flash_attn_3 --no-deps --find-links "
    "https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/"
)

_IMPORT_ERRORS: list[str] = []
_flash_attn_func = None

for _module_name in ("flash_attn.flash_attn_interface", "flash_attn"):
    try:
        _module = __import__(_module_name, fromlist=["flash_attn_func"])
        _flash_attn_func = getattr(_module, "flash_attn_func")
        break
    except Exception as exc:  # pragma: no cover - startup path only
        _IMPORT_ERRORS.append(f"{_module_name}: {exc!r}")

if _flash_attn_func is None:
    details = "\n".join(_IMPORT_ERRORS) if _IMPORT_ERRORS else "no import attempts recorded"
    raise RuntimeError(f"{FLASH_ATTN_INSTALL_HINT}\nImport errors:\n{details}")


def flash_attn_func(q, k, v, causal=False):
    if not causal:
        raise ValueError("This record path only supports causal FlashAttention")
    return _flash_attn_func(q, k, v, causal=causal)
