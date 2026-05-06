"""Train-time ternary layers shared by the sub-4MB experiments."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


_TERNARY_WEIGHT_CACHE_STACK: list[dict[tuple[int, torch.device, torch.dtype, int], Any]] = []
_PACKED_KERNEL_WARNED = False
_DENSE_KERNEL_WARNED = False


@contextmanager
def ternary_weight_cache(enabled: bool = True):
    """Reuse train-time ternary views for repeated HRC block calls within one forward."""

    if not enabled:
        yield
        return
    cache: dict[tuple[int, torch.device, torch.dtype, int], Tensor] = {}
    _TERNARY_WEIGHT_CACHE_STACK.append(cache)
    try:
        yield
    finally:
        _TERNARY_WEIGHT_CACHE_STACK.pop()


def _active_ternary_weight_cache() -> dict[tuple[int, torch.device, torch.dtype, int], Any] | None:
    if not _TERNARY_WEIGHT_CACHE_STACK:
        return None
    return _TERNARY_WEIGHT_CACHE_STACK[-1]


def _packed_ternary_kernel_requested() -> str:
    return os.environ.get("TRAIN_TERNARY_PACKED_KERNEL", "0").strip().lower()


def _dense_ternary_kernel_requested() -> str:
    return os.environ.get("TRAIN_TERNARY_DENSE_KERNEL", "0").strip().lower()


def _packed_ternary_kernel_enabled(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    mirror_mode: int,
    group_size: int,
    scale_stat: str,
) -> bool:
    mode = _packed_ternary_kernel_requested()
    if mode in {"", "0", "false", "no", "off", "none"}:
        return False
    if (
        x.device.type != "cuda"
        or weight.device.type != "cuda"
        or bias is not None
        or int(mirror_mode) != 0
        or str(scale_stat).strip().lower() != "mean"
        or int(group_size) % 16 != 0
        or x.dtype not in {torch.float16, torch.bfloat16, torch.float32}
        or weight.dtype not in {torch.float16, torch.bfloat16, torch.float32}
    ):
        return False
    return True


def _packed_ternary_linear_or_none(
    module_id: int,
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    group_size: int,
    scale_stat: str,
    mirror_mode: int,
    cache: dict[tuple[int, torch.device, torch.dtype, int], Any] | None,
) -> Tensor | None:
    global _PACKED_KERNEL_WARNED
    if not _packed_ternary_kernel_enabled(x, weight, bias, mirror_mode, group_size, scale_stat):
        return None
    try:
        from . import packed_cuda

        packed_key = (module_id, weight.device, weight.dtype, -1000)
        packed_pair: tuple[Tensor, Tensor] | None = None
        if cache is not None:
            cached = cache.get(packed_key)
            if cached is not None:
                packed_pair = cached
        if packed_pair is None:
            packed_pair = packed_cuda.pack_ternary_weight(weight, group_size)
            if cache is not None:
                cache[packed_key] = packed_pair
        packed, scales = packed_pair
        return packed_cuda.packed_ternary_linear(x, weight, packed, scales, group_size)
    except Exception:
        if _packed_ternary_kernel_requested() in {"1", "true", "yes", "on"}:
            raise
        if not _PACKED_KERNEL_WARNED:
            print("TRAIN_TERNARY_PACKED_KERNEL:auto disabled after CUDA extension failure", flush=True)
            _PACKED_KERNEL_WARNED = True
        return None


def _dense_ternary_weight_or_none(
    weight: Tensor,
    group_size: int,
    work_dtype: torch.dtype,
    scale_stat: str,
) -> Tensor | None:
    global _DENSE_KERNEL_WARNED
    mode = _dense_ternary_kernel_requested()
    if mode in {"", "0", "false", "no", "off", "none"}:
        return None
    if (
        weight.device.type != "cuda"
        or str(scale_stat).strip().lower() != "mean"
        or int(group_size) <= 0
        or int(group_size) > 1024
        or work_dtype not in {torch.float16, torch.bfloat16, torch.float32}
        or weight.dtype not in {torch.float16, torch.bfloat16, torch.float32}
    ):
        return None
    try:
        from . import packed_cuda

        return packed_cuda.dense_ternary_weight(weight, group_size, work_dtype=work_dtype)
    except Exception:
        if mode in {"1", "true", "yes", "on"}:
            raise
        if not _DENSE_KERNEL_WARNED:
            print("TRAIN_TERNARY_DENSE_KERNEL:auto disabled after CUDA extension failure", flush=True)
            _DENSE_KERNEL_WARNED = True
        return None


def mirror_weight(weight: Tensor, mirror_mode: int) -> Tensor:
    if mirror_mode == 0:
        return weight
    if mirror_mode == 1:
        row_idx = torch.arange(weight.size(0) - 1, -1, -1, device=weight.device)
        col_idx = torch.arange(weight.size(1) - 1, -1, -1, device=weight.device)
        row_signs = 1.0 - 2.0 * (torch.arange(weight.size(0), device=weight.device, dtype=torch.float32) % 2.0)
        flipped = weight.index_select(0, row_idx).index_select(1, col_idx)
        return flipped * row_signs.to(dtype=weight.dtype)[:, None]
    if mirror_mode == 2:
        cols = int(weight.size(1))
        if cols <= 1:
            return weight
        idx = torch.arange(cols, device=weight.device, dtype=torch.float32) + 1.0
        vec = torch.cos(idx * 0.61803398875)
        vec = vec / vec.norm().clamp_min(1e-6)
        return weight - 2.0 * (weight @ vec[:, None]) * vec[None, :]
    raise ValueError(f"Unsupported mirror mode code: {mirror_mode}")


def _ternary_scale_from_abs(abs_groups: Tensor, dim: int, scale_stat: str) -> Tensor:
    if scale_stat == "median":
        return abs_groups.median(dim=dim, keepdim=True).values.clamp_min(1e-8)
    return abs_groups.mean(dim=dim, keepdim=True).clamp_min(1e-8)


def ternary_ste_weight(
    weight: Tensor,
    group_size: int,
    work_dtype: torch.dtype | None = None,
    scale_stat: str = "mean",
) -> Tensor:
    group_size = max(int(group_size), 1)
    scale_stat = str(scale_stat).strip().lower()
    if scale_stat not in {"mean", "median"}:
        raise ValueError(f"scale_stat must be mean|median, got {scale_stat!r}")
    if work_dtype is None:
        work_dtype = torch.bfloat16 if weight.device.type == "cuda" else torch.float32
    w = weight.to(dtype=work_dtype)
    if w.ndim != 2:
        flat = w.reshape(-1)
        pad = (-int(flat.numel())) % group_size
        if pad:
            flat = F.pad(flat, (0, pad))
        groups = flat.view(-1, group_size)
        abs_groups = groups.abs()
        scale = _ternary_scale_from_abs(abs_groups, dim=1, scale_stat=scale_stat)
        q = groups.sign() * (abs_groups > (0.5 * scale)).to(dtype=groups.dtype)
        ternary = groups + ((q * scale) - groups).detach()
        if pad:
            ternary = ternary.flatten()[:-pad]
        return ternary.reshape_as(weight)
    rows, cols = w.shape
    pad = (-int(cols)) % group_size
    if pad:
        w = F.pad(w, (0, pad))
    groups = w.view(rows, -1, group_size)
    abs_groups = groups.abs()
    scale = _ternary_scale_from_abs(abs_groups, dim=2, scale_stat=scale_stat)
    # Equivalent to round(w / absmean).clamp(-1, 1) away from exact ties,
    # but avoids a per-forward divide+round on the training hot path.
    q = groups.sign() * (abs_groups > (0.5 * scale)).to(dtype=groups.dtype)
    ternary = groups + ((q * scale) - groups).detach()
    if pad:
        ternary = ternary.reshape(rows, cols + pad)[:, :cols]
    return ternary.reshape_as(weight)


def dequantize_ternary_groups_with_shrinkage(groups: Tensor, scale: Tensor) -> Tensor:
    q_absmean = groups.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    return groups * (scale.to(torch.float32)[:, None] / q_absmean)


class TernaryLinear(nn.Linear):
    """Linear layer that trains through a BitNet-style ternary STE forward pass."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 64,
        scale_stat: str = "mean",
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.group_size = max(int(group_size), 1)
        self.scale_stat = str(scale_stat).strip().lower()
        if self.scale_stat not in {"mean", "median"}:
            raise ValueError(f"scale_stat must be mean|median, got {scale_stat!r}")
        self.register_buffer(
            "_row_reverse_idx",
            torch.arange(out_features - 1, -1, -1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_col_reverse_idx",
            torch.arange(in_features - 1, -1, -1, dtype=torch.long),
            persistent=False,
        )
        row_signs = 1.0 - 2.0 * (torch.arange(out_features, dtype=torch.float32) % 2.0)
        self.register_buffer("_row_signs", row_signs, persistent=False)
        if in_features > 1:
            idx = torch.arange(in_features, dtype=torch.float32) + 1.0
            vec = torch.cos(idx * 0.61803398875)
            vec = vec / vec.norm().clamp_min(1e-6)
        else:
            vec = torch.ones((in_features,), dtype=torch.float32)
        self.register_buffer("_householder_vec", vec, persistent=False)
        self.register_buffer("lqer_A", None, persistent=True)
        self.register_buffer("lqer_B", None, persistent=True)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        lqer_A = state_dict.pop(prefix + "lqer_A", None)
        lqer_B = state_dict.pop(prefix + "lqer_B", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if lqer_A is None and lqer_B is None:
            self.lqer_A = None
            self.lqer_B = None
            return
        if lqer_A is None or lqer_B is None:
            error_msgs.append(f"{prefix} received incomplete LQER sidecar")
            return
        if lqer_A.ndim != 2 or lqer_B.ndim != 2:
            error_msgs.append(f"{prefix} LQER sidecar tensors must be rank-2")
            return
        if lqer_A.shape[0] != self.out_features or lqer_B.shape[1] != self.in_features or lqer_A.shape[1] != lqer_B.shape[0]:
            error_msgs.append(
                f"{prefix} LQER sidecar shape mismatch: A={tuple(lqer_A.shape)} B={tuple(lqer_B.shape)} "
                f"for weight={(self.out_features, self.in_features)}"
            )
            return
        self.lqer_A = lqer_A.detach().to(device=self.weight.device, dtype=self.weight.dtype).contiguous()
        self.lqer_B = lqer_B.detach().to(device=self.weight.device, dtype=self.weight.dtype).contiguous()

    def _mirror_weight(self, weight: Tensor, mirror_mode: int) -> Tensor:
        if mirror_mode == 0:
            return weight
        if mirror_mode == 1:
            flipped = weight.index_select(0, self._row_reverse_idx).index_select(1, self._col_reverse_idx)
            return flipped * self._row_signs.to(dtype=weight.dtype)[:, None]
        if mirror_mode == 2:
            if weight.size(1) <= 1:
                return weight
            vec = self._householder_vec.to(dtype=weight.dtype)
            return weight - 2.0 * (weight @ vec[:, None]) * vec[None, :]
        raise ValueError(f"Unsupported mirror mode code: {mirror_mode}")

    def forward(self, x: Tensor, mirror_mode: int = 0) -> Tensor:
        work_dtype = x.dtype if x.dtype in {torch.float16, torch.bfloat16} else torch.float32
        cache = _active_ternary_weight_cache()
        packed_out = _packed_ternary_linear_or_none(
            id(self),
            x,
            self.weight,
            self.bias,
            self.group_size,
            self.scale_stat,
            mirror_mode,
            cache,
        )
        if packed_out is not None:
            return packed_out
        if cache is not None:
            raw_key = (id(self), self.weight.device, x.dtype, -1)
            weight = cache.get(raw_key)
            if weight is None:
                weight = _dense_ternary_weight_or_none(
                    self.weight,
                    self.group_size,
                    work_dtype,
                    self.scale_stat,
                )
                if weight is None:
                    weight = ternary_ste_weight(
                        self.weight,
                        self.group_size,
                        work_dtype=work_dtype,
                        scale_stat=self.scale_stat,
                    ).to(dtype=x.dtype)
                cache[raw_key] = weight
            if mirror_mode:
                mirror_key = (id(self), self.weight.device, x.dtype, int(mirror_mode))
                mirrored_weight = cache.get(mirror_key)
                if mirrored_weight is None:
                    mirrored_weight = self._mirror_weight(weight, mirror_mode)
                    cache[mirror_key] = mirrored_weight
                weight = mirrored_weight
        else:
            weight = _dense_ternary_weight_or_none(
                self.weight,
                self.group_size,
                work_dtype,
                self.scale_stat,
            )
            if weight is None:
                weight = ternary_ste_weight(
                    self.weight,
                    self.group_size,
                    work_dtype=work_dtype,
                    scale_stat=self.scale_stat,
                ).to(dtype=x.dtype)
            if mirror_mode:
                weight = self._mirror_weight(weight, mirror_mode)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        out = F.linear(x, weight, bias)
        if self.lqer_A is None or self.lqer_B is None:
            return out
        A = self.lqer_A.to(device=x.device, dtype=x.dtype)
        B = self.lqer_B.to(device=x.device, dtype=x.dtype)
        if mirror_mode:
            correction = self._mirror_weight(A @ B, mirror_mode)
            return out + F.linear(x, correction)
        return out + F.linear(F.linear(x, B), A)
