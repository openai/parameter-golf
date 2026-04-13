from __future__ import annotations

from pathlib import Path
import torch


def _load_extension() -> None:
    here = Path(__file__).resolve().parent
    candidates = sorted(here.glob("cutlass_evt_fusion*.so"))
    if not candidates:
        raise ImportError(f"No compiled cutlass_evt_fusion extension found in {here}")
    torch.ops.load_library(str(candidates[0]))


_load_extension()
