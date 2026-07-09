"""Lightweight Koszul and affine-toric persistence diagnostics.

The training-time path here is intentionally parameter-free.  It turns local
reasoning windows into a small family of row-stochastic operators that play the
role of chart actions on a persistence module, then penalizes the first Koszul
syzygies among those actions.  This is not a full commutative-algebra engine;
it is a tractable audit/loss that makes the affine-toric and multiparameter
persistence claims observable during Parameter-Golf training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class KoszulPersistenceConfig:
    max_points: int = 24
    max_windows: int = 4
    window_size: int = 32
    step_stride: int = 8
    num_parameters: int = 3
    temperature: float = 0.12
    chart_exponents: int = 12
    theta: float = 0.6180339887498948
    beta: float = 1.4142135623730951
    rank_temperature: float = 0.05


def _zero_like(hidden: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = hidden.float().sum() * 0.0
    return {
        "koszul_persistence_loss": zero,
        "koszul_exactness_residual": zero.detach(),
        "koszul_syzygy_residual": zero.detach(),
        "koszul_fitting_rank_residual": zero.detach(),
        "koszul_buchsbaum_eisenbud_rank_residual": zero.detach(),
        "koszul_buchsbaum_eisenbud_multiplier_residual": zero.detach(),
        "koszul_multigraded_betti_mass": zero.detach(),
        "koszul_toric_affine_chart_entropy": zero.detach(),
        "koszul_toric_affine_chart_coverage": zero.detach(),
        "koszul_chart_transition_resolution_shift": zero.detach(),
        "koszul_windows": zero.detach(),
    }


def _window_starts(length: int, cfg: KoszulPersistenceConfig) -> list[int]:
    if length < 4:
        return []
    window = max(4, min(int(cfg.window_size), length))
    stride = max(1, int(cfg.step_stride))
    starts = list(range(0, max(1, length - window + 1), stride))
    final_start = max(0, length - window)
    if final_start not in starts:
        starts.append(final_start)
    if len(starts) > max(1, int(cfg.max_windows)):
        idx = np.linspace(0, len(starts) - 1, num=max(1, int(cfg.max_windows))).round().astype(int)
        starts = [starts[int(i)] for i in idx]
    return starts


def _sample_window(window: torch.Tensor, positions: torch.Tensor | None, max_points: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    if window.shape[0] <= max_points:
        return window, positions
    index = torch.linspace(0, window.shape[0] - 1, steps=max_points, device=window.device).round().long()
    sampled_positions = positions.index_select(0, index) if positions is not None else None
    return window.index_select(0, index), sampled_positions


def _fixed_chart_directions(width: int, count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Deterministic toric-character directions with no serialized weights."""

    count = max(2, int(count))
    coord = torch.arange(width, device=device, dtype=dtype) + 1.0
    freqs = torch.arange(1, count + 1, device=device, dtype=dtype)[:, None]
    angles = 2.0 * math.pi * freqs * coord[None, :] / float(max(2, width + count))
    dirs = torch.sin(angles) + 0.5 * torch.cos((freqs + 1.0) * angles / (freqs + 0.5))
    return F.normalize(dirs, dim=-1)


def _soft_rank(matrix: torch.Tensor, temperature: float) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix.new_zeros(())
    device_type = matrix.device.type if matrix.device.type in {"cpu", "cuda"} else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        safe = torch.nan_to_num(matrix.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if safe.shape[0] <= safe.shape[1]:
            gram = safe @ safe.transpose(0, 1)
        else:
            gram = safe.transpose(0, 1) @ safe
        gram = torch.nan_to_num(gram, nan=0.0, posinf=0.0, neginf=0.0)
        dim = gram.shape[0]
        eye = torch.eye(dim, device=gram.device, dtype=torch.float32)
        scale = gram.detach().diagonal().abs().mean().clamp_min(1e-6)
        ridge = max(float(temperature), 1e-5) * scale
        regularized = gram + ridge * eye
        try:
            response = torch.linalg.solve(regularized, gram)
        except RuntimeError:
            response = gram @ torch.linalg.pinv(regularized)
        rank = torch.trace(torch.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0))
        return rank.clamp_min(0.0).clamp_max(float(dim))


def _koszul_blocks(actions: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Build first Koszul differentials for two or three local actions."""

    r = len(actions)
    n = actions[0].shape[0]
    if r == 2:
        a0, a1 = actions
        d1 = torch.cat([a0, a1], dim=1)
        d2 = torch.cat([-a1, a0], dim=0)
        return d1, d2, None
    a0, a1, a2 = actions[:3]
    z = torch.zeros_like(a0)
    d1 = torch.cat([a0, a1, a2], dim=1)
    # Columns are (0,1), (0,2), (1,2); rows are e0,e1,e2.
    d2 = torch.cat(
        [
            torch.cat([-a1, -a2, z], dim=1),
            torch.cat([a0, z, -a2], dim=1),
            torch.cat([z, a0, a1], dim=1),
        ],
        dim=0,
    )
    d3 = torch.cat([a2, -a1, a0], dim=0)
    return d1, d2, d3


def _local_actions(
    points: torch.Tensor,
    positions: torch.Tensor | None,
    cfg: KoszulPersistenceConfig,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    n, width = points.shape
    temperature = max(float(cfg.temperature), 1e-4)
    eye = torch.eye(n, device=points.device, dtype=points.dtype)
    dist = torch.cdist(points, points, p=2)
    positive = dist[dist > 1e-6]
    if positive.numel() > 0:
        dist = dist / positive.detach().median().clamp_min(1e-4)
    masked_dist = dist + eye * 1.0e4
    radius_action = F.softmax(-masked_dist / temperature, dim=-1)
    radius_action = 0.85 * radius_action + 0.15 * eye

    time_action = torch.zeros_like(radius_action)
    idx = torch.arange(n, device=points.device)
    time_action[idx, idx] = 0.35
    time_action[idx[:-1], idx[1:]] = 0.65
    time_action[idx[-1], idx[-1]] = 1.0

    directions = _fixed_chart_directions(width, int(cfg.chart_exponents), points.device, points.dtype)
    chart_logits = points @ directions.transpose(0, 1)
    chart_probs = F.softmax(chart_logits / temperature, dim=-1)
    chart_action = chart_probs @ chart_probs.transpose(0, 1)
    chart_action = chart_action / chart_action.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    if positions is not None:
        pos = positions.float()
        angle1 = 2.0 * math.pi * float(cfg.theta) * pos
        angle2 = 2.0 * math.pi * float(cfg.beta) * pos
        phase = torch.stack([torch.cos(angle1), torch.sin(angle1), torch.cos(angle2), torch.sin(angle2)], dim=-1)
        phase_dist = torch.cdist(phase, phase, p=2) + eye * 1.0e4
        phase_action = F.softmax(-phase_dist / temperature, dim=-1)
        phase_action = 0.80 * phase_action + 0.20 * chart_action
    else:
        phase_action = chart_action

    actions = [time_action, radius_action, phase_action]
    return actions[: max(2, min(3, int(cfg.num_parameters)))], chart_probs, chart_logits, dist


def koszul_persistence_loss(
    hidden: torch.Tensor,
    positions: torch.Tensor | None = None,
    *,
    config: KoszulPersistenceConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Return a differentiable Koszul residual and detached algebra metrics."""

    cfg = config or KoszulPersistenceConfig()
    if hidden.ndim != 3 or hidden.shape[1] < 4:
        return _zero_like(hidden)
    x_all = torch.nan_to_num(hidden.float(), nan=0.0, posinf=0.0, neginf=0.0)
    starts = _window_starts(int(x_all.shape[1]), cfg)
    if not starts:
        return _zero_like(hidden)

    terms: dict[str, list[torch.Tensor]] = {
        "exactness": [],
        "syzygy": [],
        "fitting": [],
        "be_rank": [],
        "be_multiplier": [],
        "betti": [],
        "entropy": [],
        "coverage": [],
        "shift": [],
    }
    previous_chart: torch.Tensor | None = None
    for batch_index in range(x_all.shape[0]):
        sequence = x_all[batch_index]
        pos_sequence = positions[batch_index] if positions is not None else None
        for start in starts:
            window = sequence[start : start + max(4, min(int(cfg.window_size), sequence.shape[0]))]
            pos_window = pos_sequence[start : start + window.shape[0]] if pos_sequence is not None else None
            points, pos_points = _sample_window(window, pos_window, int(cfg.max_points))
            if points.shape[0] < 4:
                continue
            points = torch.nan_to_num(F.normalize(points, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
            actions, chart_probs, _chart_logits, _dist = _local_actions(points, pos_points, cfg)
            d1, d2, d3 = _koszul_blocks(actions)
            exactness = (d1 @ d2).pow(2).mean()
            if d3 is not None:
                exactness = exactness + (d2 @ d3).pow(2).mean()
            commutators = []
            for i, left in enumerate(actions):
                for right in actions[i + 1 :]:
                    commutators.append((left @ right - right @ left).pow(2).mean())
            syzygy = torch.stack(commutators).mean() if commutators else exactness.new_zeros(())

            r = len(actions)
            n = int(points.shape[0])
            sr_d1 = _soft_rank(d1, float(cfg.rank_temperature))
            sr_d2 = _soft_rank(d2, float(cfg.rank_temperature))
            if d3 is not None:
                sr_d3 = _soft_rank(d3, float(cfg.rank_temperature))
                mid_dim = float((r * (r - 1) // 2) * n)
                be_rank = (
                    (sr_d1 + sr_d2 - float(r * n)).abs() / max(1.0, float(r * n))
                    + (sr_d2 + sr_d3 - mid_dim).abs() / max(1.0, mid_dim)
                )
                fitting = (
                    F.relu(sr_d1.new_tensor(float(n)) - sr_d1) / max(1.0, float(n))
                    + F.relu(sr_d1.new_tensor(float((r - 1) * n)) - sr_d2) / max(1.0, float((r - 1) * n))
                    + F.relu(sr_d1.new_tensor(float(n)) - sr_d3) / max(1.0, float(n))
                )
                betti = (
                    F.relu(sr_d1.new_tensor(float(r * n)) - sr_d1 - sr_d2) / max(1.0, float(r * n))
                    + F.relu(sr_d1.new_tensor(mid_dim) - sr_d2 - sr_d3) / max(1.0, mid_dim)
                )
            else:
                be_rank = (sr_d1 + sr_d2 - float(r * n)).abs() / max(1.0, float(r * n))
                fitting = (
                    F.relu(sr_d1.new_tensor(float(n)) - sr_d1) / max(1.0, float(n))
                    + F.relu(sr_d1.new_tensor(float(n)) - sr_d2) / max(1.0, float(n))
                )
                betti = F.relu(sr_d1.new_tensor(float(r * n)) - sr_d1 - sr_d2) / max(1.0, float(r * n))
            # In the differentiable proxy, Buchsbaum-Eisenbud multiplier failure
            # is represented by exactness plus rank-condition mismatch; exact
            # complementary-minor checks are performed in the NumPy audit path.
            be_multiplier = 0.5 * exactness + 0.5 * be_rank
            chart_mass = chart_probs.mean(dim=0)
            entropy = -(chart_mass * (chart_mass + 1e-8).log()).sum() / math.log(max(2, chart_mass.numel()))
            coverage = (chart_mass > (1.0 / max(2, chart_mass.numel())) * 0.25).to(points.dtype).mean()
            chart_id = chart_probs.argmax(dim=-1).float()
            if previous_chart is None or previous_chart.numel() != chart_id.numel():
                shift = points.new_zeros(())
            else:
                shift = (chart_id - previous_chart.to(chart_id.device)).abs().mean() / max(1.0, float(chart_mass.numel()))
            previous_chart = chart_id.detach()
            terms["exactness"].append(exactness)
            terms["syzygy"].append(syzygy)
            terms["fitting"].append(fitting)
            terms["be_rank"].append(be_rank)
            terms["be_multiplier"].append(be_multiplier)
            terms["betti"].append(betti)
            terms["entropy"].append(entropy.detach())
            terms["coverage"].append(coverage.detach())
            terms["shift"].append(shift.detach())

    if not terms["exactness"]:
        return _zero_like(hidden)

    def mean(name: str) -> torch.Tensor:
        return torch.stack(terms[name]).mean()

    exactness_residual = mean("exactness")
    syzygy_residual = mean("syzygy")
    fitting_residual = mean("fitting")
    be_rank_residual = mean("be_rank")
    be_multiplier_residual = mean("be_multiplier")
    betti_mass = mean("betti")
    loss = (
        exactness_residual
        + 0.35 * syzygy_residual
        + 0.10 * fitting_residual
        + 0.10 * be_rank_residual
        + 0.04 * be_multiplier_residual
        + 0.02 * betti_mass
    )
    loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0e4, neginf=0.0).clamp_min(0.0)
    return {
        "koszul_persistence_loss": loss,
        "koszul_exactness_residual": torch.nan_to_num(exactness_residual.detach(), nan=0.0, posinf=1.0e4, neginf=0.0),
        "koszul_syzygy_residual": torch.nan_to_num(syzygy_residual.detach(), nan=0.0, posinf=1.0e4, neginf=0.0),
        "koszul_fitting_rank_residual": torch.nan_to_num(fitting_residual.detach(), nan=0.0, posinf=1.0e4, neginf=0.0),
        "koszul_buchsbaum_eisenbud_rank_residual": torch.nan_to_num(be_rank_residual.detach(), nan=0.0, posinf=1.0e4, neginf=0.0),
        "koszul_buchsbaum_eisenbud_multiplier_residual": torch.nan_to_num(be_multiplier_residual.detach(), nan=0.0, posinf=1.0e4, neginf=0.0),
        "koszul_multigraded_betti_mass": torch.nan_to_num(betti_mass.detach(), nan=0.0, posinf=1.0e4, neginf=0.0),
        "koszul_toric_affine_chart_entropy": torch.nan_to_num(mean("entropy").detach(), nan=0.0, posinf=1.0, neginf=0.0),
        "koszul_toric_affine_chart_coverage": torch.nan_to_num(mean("coverage").detach(), nan=0.0, posinf=1.0, neginf=0.0),
        "koszul_chart_transition_resolution_shift": torch.nan_to_num(mean("shift").detach(), nan=0.0, posinf=1.0e4, neginf=0.0),
        "koszul_windows": hidden.new_tensor(float(len(terms["exactness"]))).detach(),
    }
