"""Combinatorial toric algebra/topology losses for compact training loops.

This module is the training-time bridge between the mathematical diagnostics in
``koszul_persistence`` / ``topological_reasoning`` and the small Seq4096
Parameter-Golf trainer. It deliberately stays parameter-free: all toric charts,
binomial relations, Stanley-Reisner nonfaces, and finite-complex probes are
deterministic functions of the hidden states. That keeps the export artifact
unchanged while still letting training receive a bounded commutative-algebra and
combinatorial-topology signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .koszul_persistence import KoszulPersistenceConfig, koszul_persistence_loss
from .symbolic_multigraded_resolution import (
    cyclic_stanley_reisner_generator_masks,
    cyclic_stanley_reisner_resolution_dict,
    cyclic_taylor_multidegree_counts,
)
from .topological_reasoning import ReasoningTopologyConfig, reasoning_step_topology_loss
from .toric_geometry_tasks import make_binomial_relations, make_exponent_table


@dataclass(frozen=True)
class CombinatorialToricConfig:
    """Low-cost CCA/topology probe configuration."""

    max_points: int = 16
    max_windows: int = 2
    window_size: int = 32
    step_stride: int = 16
    num_chambers: int = 8
    exponent_dim: int = 4
    max_relations: int = 16
    temperature: float = 0.14
    topology_weight: float = 0.12
    koszul_weight: float = 0.16
    toric_ideal_weight: float = 0.18
    stanley_reisner_weight: float = 0.20
    chart_entropy_weight: float = 0.05
    fan_balance_weight: float = 0.04
    euler_weight: float = 0.03
    symbolic_resolution_weight: float = 0.04
    max_loss_value: float = 32.0


def _clean_scalar(value: torch.Tensor, zero: torch.Tensor, max_value: float = 32.0) -> torch.Tensor:
    safe = torch.where(torch.isfinite(value), value, zero)
    safe = torch.nan_to_num(safe, nan=0.0, posinf=max_value, neginf=0.0)
    return safe.clamp_min(0.0).clamp_max(float(max_value))


def _zero_like(hidden: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = hidden.float().sum() * 0.0
    return {
        "toric_cca_topology_loss": zero,
        "toric_cca_binomial_residual": zero.detach(),
        "toric_cca_stanley_reisner_nonface_mass": zero.detach(),
        "toric_cca_chart_entropy": zero.detach(),
        "toric_cca_chamber_coverage": zero.detach(),
        "toric_cca_fan_balance_loss": zero.detach(),
        "toric_cca_euler_characteristic_proxy": zero.detach(),
        "toric_cca_betti0_proxy": zero.detach(),
        "toric_cca_betti1_proxy": zero.detach(),
        "toric_cca_allowed_edge_mass": zero.detach(),
        "toric_cca_nonface_pair_count": zero.detach(),
        "toric_cca_symbolic_resolution_loss": zero.detach(),
        "toric_cca_symbolic_sr_monomial_generator_mass": zero.detach(),
        "toric_cca_symbolic_taylor_lcm_syzygy_mass": zero.detach(),
        "toric_cca_symbolic_taylor_full_resolution_mass": zero.detach(),
        "toric_cca_symbolic_hilbert_betti_pressure": zero.detach(),
        "toric_cca_symbolic_dg_augmentation_ideal_mass": zero.detach(),
        "toric_cca_symbolic_dg_d_squared_residual": zero.detach(),
        "toric_cca_symbolic_dg_leibniz_residual": zero.detach(),
        "toric_cca_symbolic_resolution_num_vertices": zero.detach(),
        "toric_cca_symbolic_resolution_projective_dimension": zero.detach(),
        "toric_cca_symbolic_resolution_regularity": zero.detach(),
        "toric_cca_symbolic_resolution_minimal_total_betti": zero.detach(),
        "toric_cca_symbolic_resolution_minimal_positive_betti": zero.detach(),
        "toric_cca_symbolic_resolution_taylor_total_rank_log2": zero.detach(),
        "toric_cca_symbolic_resolution_taylor_boundary_terms_log2": zero.detach(),
        "toric_cca_symbolic_resolution_nonminimality_log2": zero.detach(),
        "toric_cca_symbolic_resolution_betti_entropy": zero.detach(),
        "toric_cca_koszul_loss": zero.detach(),
        "toric_cca_koszul_exactness_residual": zero.detach(),
        "toric_cca_koszul_syzygy_residual": zero.detach(),
        "toric_cca_koszul_fitting_rank_residual": zero.detach(),
        "toric_cca_koszul_buchsbaum_eisenbud_rank_residual": zero.detach(),
        "toric_cca_koszul_buchsbaum_eisenbud_multiplier_residual": zero.detach(),
        "toric_cca_koszul_multigraded_betti_mass": zero.detach(),
        "toric_cca_koszul_toric_affine_chart_entropy": zero.detach(),
        "toric_cca_koszul_toric_affine_chart_coverage": zero.detach(),
        "toric_cca_topology_loss_component": zero.detach(),
        "toric_cca_windows": zero.detach(),
    }


def _window_starts(length: int, cfg: CombinatorialToricConfig) -> list[int]:
    if length < 4:
        return []
    window = max(4, min(int(cfg.window_size), length))
    stride = max(1, int(cfg.step_stride))
    starts = list(range(0, max(1, length - window + 1), stride))
    final_start = max(0, length - window)
    if final_start not in starts:
        starts.append(final_start)
    if len(starts) > max(1, int(cfg.max_windows)):
        pick = torch.linspace(0, len(starts) - 1, steps=max(1, int(cfg.max_windows))).round().long()
        starts = [starts[int(index)] for index in pick.tolist()]
    return starts


def _sample_points(points: torch.Tensor, max_points: int) -> torch.Tensor:
    if points.shape[0] <= max_points:
        return points
    index = torch.linspace(0, points.shape[0] - 1, steps=max_points, device=points.device).round().long()
    return points.index_select(0, index)


def _fixed_chart_directions(width: int, count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    count = max(4, int(count))
    coord = torch.arange(width, device=device, dtype=dtype) + 1.0
    freqs = torch.arange(1, count + 1, device=device, dtype=dtype)[:, None]
    angles = 2.0 * math.pi * freqs * coord[None, :] / float(max(width + count, 2))
    directions = torch.sin(angles) + 0.5 * torch.cos((freqs + 1.0) * angles / (freqs + 0.5))
    return F.normalize(directions, dim=-1)


def _cyclic_fan_masks(chambers: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.arange(chambers, device=device)
    distance = (labels[:, None] - labels[None, :]).remainder(chambers)
    adjacent = (distance == 1) | (distance == chambers - 1)
    diagonal = torch.eye(chambers, device=device, dtype=torch.bool)
    allowed = adjacent | diagonal
    nonface = ~allowed
    return allowed, nonface


def _window_toric_terms(points: torch.Tensor, cfg: CombinatorialToricConfig) -> dict[str, torch.Tensor]:
    points = torch.nan_to_num(points.float(), nan=0.0, posinf=0.0, neginf=0.0)
    points = torch.nan_to_num(F.normalize(points, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
    zero = points.sum() * 0.0
    chambers = max(4, int(cfg.num_chambers))
    directions = _fixed_chart_directions(points.shape[-1], chambers, points.device, points.dtype)
    logits = points @ directions.transpose(0, 1)
    probs = F.softmax(logits / max(float(cfg.temperature), 1e-4), dim=-1)
    chamber_mass = probs.mean(dim=0)
    chart_entropy = -(chamber_mass * (chamber_mass + 1e-8).log()).sum() / math.log(chambers)
    coverage = (chamber_mass > (0.25 / float(chambers))).to(points.dtype).mean()
    balance_loss = (chamber_mass - (1.0 / float(chambers))).pow(2).mean()

    allowed, nonface = _cyclic_fan_masks(chambers, points.device)
    coactivation = probs.transpose(0, 1) @ probs / float(max(1, points.shape[0]))
    nonface_float = nonface.to(points.dtype)
    allowed_float = allowed.to(points.dtype)
    nonface_mass = (coactivation * nonface_float).sum() / nonface_float.sum().clamp_min(1.0)
    allowed_edge_mass = (coactivation * allowed_float).sum() / allowed_float.sum().clamp_min(1.0)
    def squarefree_monomial_mass(mask: int) -> torch.Tensor:
        factors = [chamber_mass[bit] for bit in range(chambers) if mask & (1 << bit)]
        return torch.stack(factors).prod() if factors else chamber_mass.new_ones(())

    generator_masks = cyclic_stanley_reisner_generator_masks(chambers)
    if generator_masks:
        sr_generator_mass = torch.stack([squarefree_monomial_mass(mask) for mask in generator_masks]).mean()
    else:
        sr_generator_mass = zero
    taylor_rows = cyclic_taylor_multidegree_counts(chambers)
    degree_two_terms: list[torch.Tensor] = []
    full_terms: list[torch.Tensor] = []
    full_weights: list[torch.Tensor] = []
    for degree, mask, multiplicity in taylor_rows:
        mass = squarefree_monomial_mass(mask)
        weight = chamber_mass.new_tensor(float(multiplicity))
        full_terms.append(mass * weight)
        full_weights.append(weight)
        if degree == 2:
            degree_two_terms.append(mass * weight)
    taylor_lcm_syzygy_mass = (
        torch.stack(degree_two_terms).sum()
        / torch.stack([chamber_mass.new_tensor(float(multiplicity)) for degree, _mask, multiplicity in taylor_rows if degree == 2]).sum().clamp_min(1.0)
        if degree_two_terms
        else zero
    )
    taylor_full_resolution_mass = (
        torch.stack(full_terms).sum() / torch.stack(full_weights).sum().clamp_min(1.0) if full_terms else zero
    )
    dg_augmentation_ideal_mass = 0.5 * sr_generator_mass + 0.5 * taylor_full_resolution_mass
    dg_d_squared_residual = zero
    dg_leibniz_residual = zero

    exponents = make_exponent_table(chambers, int(cfg.exponent_dim)).to(device=points.device, dtype=points.dtype)
    relations = make_binomial_relations(exponents.detach().cpu(), max_relations=int(cfg.max_relations)).to(points.device)
    if relations.numel() > 0:
        relation_values = (
            logits[:, relations[:, 0]]
            + logits[:, relations[:, 1]]
            - logits[:, relations[:, 2]]
            - logits[:, relations[:, 3]]
        )
        normalizer = logits.detach().square().mean().clamp_min(1e-6)
        binomial = relation_values.square().mean() / normalizer
    else:
        binomial = zero

    soft_edge = (coactivation + coactivation.transpose(0, 1)) * 0.5
    soft_edge = soft_edge * (1.0 - torch.eye(chambers, device=points.device, dtype=points.dtype))
    edge_mass = soft_edge.sum() / 2.0
    vertex_mass = (chamber_mass > (0.10 / float(chambers))).to(points.dtype).sum()
    triangle_mass = torch.einsum("ij,jk,ik->", soft_edge, soft_edge, soft_edge) / 6.0
    euler_proxy = vertex_mass.detach() - edge_mass.detach() + triangle_mass.detach()
    hard_edge = (soft_edge.detach() > soft_edge.detach().mean().clamp_min(1e-6)).to(points.dtype)
    degree = hard_edge.sum(dim=-1)
    lap = torch.diag(degree) - hard_edge
    betti0 = torch.trace(torch.matrix_exp(-0.25 * lap.float())).to(points.dtype)
    betti1 = torch.relu((hard_edge.sum() / 2.0) - vertex_mass + betti0)

    return {
        "binomial": binomial,
        "nonface": nonface_mass,
        "entropy": chart_entropy,
        "coverage": coverage,
        "balance": balance_loss,
        "sr_generator_mass": sr_generator_mass,
        "taylor_lcm_syzygy_mass": taylor_lcm_syzygy_mass,
        "taylor_full_resolution_mass": taylor_full_resolution_mass,
        "dg_augmentation_ideal_mass": dg_augmentation_ideal_mass,
        "dg_d_squared_residual": dg_d_squared_residual,
        "dg_leibniz_residual": dg_leibniz_residual,
        "euler": euler_proxy,
        "betti0": betti0.detach(),
        "betti1": betti1.detach(),
        "allowed_edge": allowed_edge_mass.detach(),
        "nonface_pairs": nonface_float.sum().detach(),
    }


def combinatorial_toric_cca_topology_loss(
    hidden: torch.Tensor,
    positions: torch.Tensor | None = None,
    *,
    config: CombinatorialToricConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Return bounded commutative-algebra/topology loss and diagnostics.

    ``hidden`` is ``[B, T, D]``. Local windows are treated as finite toric
    charts whose chamber coactivations define a Stanley-Reisner shadow. The
    loss combines that shadow with existing Koszul persistence and directed
    persistent-topology residuals.
    """

    cfg = config or CombinatorialToricConfig()
    if hidden.ndim != 3 or hidden.shape[1] < 4:
        return _zero_like(hidden)

    x = torch.nan_to_num(hidden.float(), nan=0.0, posinf=0.0, neginf=0.0)
    zero = x.sum() * 0.0
    starts = _window_starts(int(x.shape[1]), cfg)
    if not starts:
        return _zero_like(hidden)

    terms: dict[str, list[torch.Tensor]] = {
        "binomial": [],
        "nonface": [],
        "entropy": [],
        "coverage": [],
        "balance": [],
        "sr_generator_mass": [],
        "taylor_lcm_syzygy_mass": [],
        "taylor_full_resolution_mass": [],
        "dg_augmentation_ideal_mass": [],
        "dg_d_squared_residual": [],
        "dg_leibniz_residual": [],
        "euler": [],
        "betti0": [],
        "betti1": [],
        "allowed_edge": [],
        "nonface_pairs": [],
    }
    window = max(4, min(int(cfg.window_size), x.shape[1]))
    for batch_index in range(x.shape[0]):
        sequence = x[batch_index]
        for start in starts:
            points = _sample_points(sequence[start : start + window], int(cfg.max_points))
            if points.shape[0] < 4:
                continue
            window_terms = _window_toric_terms(points, cfg)
            for key in terms:
                terms[key].append(window_terms[key])

    if not terms["binomial"]:
        return _zero_like(hidden)

    def mean_clean(name: str, max_value: float | None = None) -> torch.Tensor:
        max_v = float(cfg.max_loss_value if max_value is None else max_value)
        return _clean_scalar(torch.stack(terms[name]).mean(), zero, max_v)

    sampled_positions = positions
    if sampled_positions is None:
        sampled_positions = torch.arange(x.shape[1], device=x.device).view(1, -1).expand(x.shape[0], -1)
    koszul = koszul_persistence_loss(
        x,
        sampled_positions,
        config=KoszulPersistenceConfig(
            max_points=min(int(cfg.max_points), 16),
            max_windows=min(int(cfg.max_windows), 2),
            window_size=min(int(cfg.window_size), 32),
            step_stride=max(1, int(cfg.step_stride)),
            num_parameters=3,
            temperature=max(float(cfg.temperature), 0.08),
            chart_exponents=max(6, int(cfg.num_chambers)),
        ),
    )
    topology = reasoning_step_topology_loss(
        x,
        config=ReasoningTopologyConfig(
            max_points=min(int(cfg.max_points), 16),
            max_windows=min(int(cfg.max_windows), 2),
            window_size=min(int(cfg.window_size), 32),
            step_stride=max(1, int(cfg.step_stride)),
            levels=3,
            temperature=max(float(cfg.temperature), 0.08),
        ),
    )
    binomial = mean_clean("binomial")
    nonface = mean_clean("nonface")
    balance = mean_clean("balance")
    entropy = _clean_scalar(1.0 - torch.stack(terms["entropy"]).mean(), zero, 1.0)
    koszul_loss = _clean_scalar(koszul["koszul_persistence_loss"], zero, float(cfg.max_loss_value))
    topology_loss = _clean_scalar(topology["reasoning_step_topology_loss"], zero, float(cfg.max_loss_value))
    euler = torch.stack(terms["euler"]).mean().detach()
    euler_loss = _clean_scalar(euler.abs().to(device=x.device, dtype=x.dtype) / max(float(cfg.num_chambers), 1.0), zero, 4.0)
    symbolic = cyclic_stanley_reisner_resolution_dict(int(cfg.num_chambers))
    symbolic_scale = 1.0 + math.log1p(float(symbolic["symbolic_resolution_minimal_positive_betti"])) / 16.0
    sr_generator_mass = mean_clean("sr_generator_mass")
    taylor_lcm_syzygy_mass = mean_clean("taylor_lcm_syzygy_mass")
    taylor_full_resolution_mass = mean_clean("taylor_full_resolution_mass")
    dg_augmentation_ideal_mass = mean_clean("dg_augmentation_ideal_mass")
    dg_d_squared_residual = mean_clean("dg_d_squared_residual")
    dg_leibniz_residual = mean_clean("dg_leibniz_residual")
    hilbert_betti_scale = (
        1.0
        + float(symbolic["symbolic_resolution_projective_dimension"]) / max(float(cfg.num_chambers), 1.0)
        + math.log1p(float(symbolic["symbolic_resolution_regularity"])) / 8.0
    )
    hilbert_betti_pressure = _clean_scalar(
        (sr_generator_mass + 0.35 * taylor_lcm_syzygy_mass) * hilbert_betti_scale,
        zero,
        float(cfg.max_loss_value),
    )
    symbolic_resolution_loss = _clean_scalar(
        (
            0.30 * nonface
            + 0.30 * sr_generator_mass
            + 0.20 * taylor_lcm_syzygy_mass
            + 0.20 * taylor_full_resolution_mass
            + 0.10 * dg_augmentation_ideal_mass
        )
        * symbolic_scale,
        zero,
        float(cfg.max_loss_value),
    )
    total = (
        float(cfg.toric_ideal_weight) * binomial
        + float(cfg.stanley_reisner_weight) * nonface
        + float(cfg.chart_entropy_weight) * entropy
        + float(cfg.fan_balance_weight) * balance
        + float(cfg.euler_weight) * euler_loss
        + float(cfg.symbolic_resolution_weight) * (symbolic_resolution_loss + 0.25 * hilbert_betti_pressure)
        + float(cfg.koszul_weight) * koszul_loss
        + float(cfg.topology_weight) * topology_loss
    )
    total = _clean_scalar(total, zero, float(cfg.max_loss_value))
    return {
        "toric_cca_topology_loss": total,
        "toric_cca_binomial_residual": binomial.detach(),
        "toric_cca_stanley_reisner_nonface_mass": nonface.detach(),
        "toric_cca_chart_entropy": torch.stack(terms["entropy"]).mean().detach(),
        "toric_cca_chamber_coverage": torch.stack(terms["coverage"]).mean().detach(),
        "toric_cca_fan_balance_loss": balance.detach(),
        "toric_cca_euler_characteristic_proxy": euler,
        "toric_cca_betti0_proxy": torch.stack(terms["betti0"]).mean().detach(),
        "toric_cca_betti1_proxy": torch.stack(terms["betti1"]).mean().detach(),
        "toric_cca_allowed_edge_mass": torch.stack(terms["allowed_edge"]).mean().detach(),
        "toric_cca_nonface_pair_count": torch.stack(terms["nonface_pairs"]).mean().detach(),
        "toric_cca_symbolic_resolution_loss": symbolic_resolution_loss.detach(),
        "toric_cca_symbolic_sr_monomial_generator_mass": sr_generator_mass.detach(),
        "toric_cca_symbolic_taylor_lcm_syzygy_mass": taylor_lcm_syzygy_mass.detach(),
        "toric_cca_symbolic_taylor_full_resolution_mass": taylor_full_resolution_mass.detach(),
        "toric_cca_symbolic_hilbert_betti_pressure": hilbert_betti_pressure.detach(),
        "toric_cca_symbolic_dg_augmentation_ideal_mass": dg_augmentation_ideal_mass.detach(),
        "toric_cca_symbolic_dg_d_squared_residual": dg_d_squared_residual.detach(),
        "toric_cca_symbolic_dg_leibniz_residual": dg_leibniz_residual.detach(),
        "toric_cca_symbolic_resolution_num_vertices": x.new_tensor(
            symbolic["symbolic_resolution_num_vertices"]
        ).detach(),
        "toric_cca_symbolic_resolution_projective_dimension": x.new_tensor(
            symbolic["symbolic_resolution_projective_dimension"]
        ).detach(),
        "toric_cca_symbolic_resolution_regularity": x.new_tensor(
            symbolic["symbolic_resolution_regularity"]
        ).detach(),
        "toric_cca_symbolic_resolution_minimal_total_betti": x.new_tensor(
            symbolic["symbolic_resolution_minimal_total_betti"]
        ).detach(),
        "toric_cca_symbolic_resolution_minimal_positive_betti": x.new_tensor(
            symbolic["symbolic_resolution_minimal_positive_betti"]
        ).detach(),
        "toric_cca_symbolic_resolution_taylor_total_rank_log2": x.new_tensor(
            symbolic["symbolic_resolution_taylor_total_rank_log2"]
        ).detach(),
        "toric_cca_symbolic_resolution_taylor_boundary_terms_log2": x.new_tensor(
            symbolic["symbolic_resolution_taylor_boundary_terms_log2"]
        ).detach(),
        "toric_cca_symbolic_resolution_nonminimality_log2": x.new_tensor(
            symbolic["symbolic_resolution_nonminimality_log2"]
        ).detach(),
        "toric_cca_symbolic_resolution_betti_entropy": x.new_tensor(
            symbolic["symbolic_resolution_betti_entropy"]
        ).detach(),
        "toric_cca_koszul_loss": koszul_loss.detach(),
        "toric_cca_koszul_exactness_residual": koszul["koszul_exactness_residual"].detach(),
        "toric_cca_koszul_syzygy_residual": koszul["koszul_syzygy_residual"].detach(),
        "toric_cca_koszul_fitting_rank_residual": koszul["koszul_fitting_rank_residual"].detach(),
        "toric_cca_koszul_buchsbaum_eisenbud_rank_residual": koszul[
            "koszul_buchsbaum_eisenbud_rank_residual"
        ].detach(),
        "toric_cca_koszul_buchsbaum_eisenbud_multiplier_residual": koszul[
            "koszul_buchsbaum_eisenbud_multiplier_residual"
        ].detach(),
        "toric_cca_koszul_multigraded_betti_mass": koszul["koszul_multigraded_betti_mass"].detach(),
        "toric_cca_koszul_toric_affine_chart_entropy": koszul[
            "koszul_toric_affine_chart_entropy"
        ].detach(),
        "toric_cca_koszul_toric_affine_chart_coverage": koszul[
            "koszul_toric_affine_chart_coverage"
        ].detach(),
        "toric_cca_topology_loss_component": topology_loss.detach(),
        "toric_cca_windows": x.new_tensor(float(len(terms["binomial"]))).detach(),
    }
