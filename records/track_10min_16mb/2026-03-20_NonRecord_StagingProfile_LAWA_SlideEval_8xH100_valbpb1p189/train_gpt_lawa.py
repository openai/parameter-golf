"""LAWA (lane 7) helpers for train_gpt.py — keeps train_gpt.py under the line budget."""
from __future__ import annotations

from collections import deque
from collections.abc import Callable

import torch
import torch.distributed as dist
from torch import Tensor, nn


def lawa_float_state_cpu(model: nn.Module) -> dict[str, Tensor]:
    """CPU snapshot for checkpoint deque (bounded K copies)."""
    return {
        k: v.detach().float().cpu().clone()
        for k, v in model.state_dict().items()
        if v.is_floating_point()
    }


def lawa_ema_shadow_init(model: nn.Module) -> dict[str, Tensor]:
    """Float32 shadow on each tensor's device — fast per-step EMA (avoids CPU sync)."""
    return {
        k: v.detach().float().clone()
        for k, v in model.state_dict().items()
        if v.is_floating_point()
    }


def lawa_ema_update(shadow: dict[str, Tensor], model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        sd = model.state_dict()
        for k, t in sd.items():
            if not t.is_floating_point() or k not in shadow:
                continue
            cur = t.detach().float()
            shadow[k].mul_(decay).add_(cur, alpha=(1.0 - decay))


def lawa_apply_float_cpu_to_model(
    model: nn.Module, averaged: dict[str, Tensor], log0: Callable[[str], None]
) -> None:
    with torch.no_grad():
        for k, t in model.state_dict().items():
            if k in averaged and t.is_floating_point():
                t.copy_(averaged[k].to(device=t.device, dtype=t.dtype))
    log0("lawa: applied averaged weights to base_model before export")


def lawa_broadcast_float_state(model: nn.Module) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for t in model.state_dict().values():
        if t.is_floating_point():
            dist.broadcast(t, src=0)


def lawa_finalize_to_model(
    model: nn.Module,
    lawa_shadow: dict[str, Tensor] | None,
    lawa_ckpt_deque: deque | None,
    lawa_window: int,
    log0: Callable[[str], None],
) -> None:
    """Apply EMA shadow or mean of last checkpoint snapshots (before final val / export)."""
    if lawa_shadow is not None:
        lawa_apply_float_cpu_to_model(model, lawa_shadow, log0)
        return
    if lawa_ckpt_deque is not None and len(lawa_ckpt_deque) > 0:
        keys = lawa_ckpt_deque[0].keys()
        averaged: dict[str, Tensor] = {}
        for k in keys:
            stacked = torch.stack([snap[k] for snap in lawa_ckpt_deque], dim=0)
            averaged[k] = stacked.mean(dim=0)
        if len(lawa_ckpt_deque) < lawa_window:
            log0(
                f"lawa: checkpoint mode averaged {len(lawa_ckpt_deque)} snapshot(s) "
                f"(window={lawa_window})"
            )
        lawa_apply_float_cpu_to_model(model, averaged, log0)
