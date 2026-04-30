"""Strict FlashAttention wrapper for the April 20 LaCT record path.

This folder intentionally does not provide any SDPA/math fallback. Priority and
record runs must use the same FlashAttention installation family as the April 9
record. If the backend is missing, fail immediately with an explicit message.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

FLASH_ATTN_INSTALL_HINT = (
    "FlashAttention is required for this record path. Install it with the same "
    "command used by the April 9 record:\n"
    "pip install flash_attn_3 --no-deps --find-links "
    "https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/"
)

_IMPORT_ERRORS: list[str] = []
_flash_attn_func = None
_flash_attn_module = None


def _load_site_module(module_name: str):
    current_dir = str(Path(__file__).resolve().parent)
    search_path = [p for p in sys.path if Path(p or ".").resolve() != Path(current_dir)]
    spec = importlib.machinery.PathFinder.find_spec(module_name, search_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"No module named '{module_name}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


for _module_name in ("flash_attn.flash_attn_interface", "flash_attn"):
    try:
        _module = __import__(_module_name, fromlist=["flash_attn_func"])
        _flash_attn_module = _module
        _flash_attn_func = getattr(_module, "flash_attn_func")
        break
    except Exception as exc:  # pragma: no cover - startup path only
        _IMPORT_ERRORS.append(f"{_module_name}: {exc!r}")

if _flash_attn_func is None:
    try:
        _module = _load_site_module("flash_attn_interface")
        _flash_attn_module = _module
        _flash_attn_func = getattr(_module, "flash_attn_func")
    except Exception as exc:  # pragma: no cover - startup path only
        _IMPORT_ERRORS.append(f"flash_attn_interface(site-packages): {exc!r}")

if _flash_attn_func is None:
    details = "\n".join(_IMPORT_ERRORS) if _IMPORT_ERRORS else "no import attempts recorded"
    raise RuntimeError(f"{FLASH_ATTN_INSTALL_HINT}\nImport errors:\n{details}")

if _flash_attn_module is not None:
    for _attr_name in (
        "FlashAttnFunc",
        "flash_attn_varlen_func",
        "flash_attn_with_kvcache",
    ):
        if hasattr(_flash_attn_module, _attr_name):
            globals()[_attr_name] = getattr(_flash_attn_module, _attr_name)


def flash_attn_func(q, k, v, causal=False):
    if not causal:
        raise ValueError("This record path only supports causal FlashAttention")
    return _flash_attn_func(q, k, v, causal=causal)


def __getattr__(name: str):
    if _flash_attn_module is not None and hasattr(_flash_attn_module, name):
        return getattr(_flash_attn_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
