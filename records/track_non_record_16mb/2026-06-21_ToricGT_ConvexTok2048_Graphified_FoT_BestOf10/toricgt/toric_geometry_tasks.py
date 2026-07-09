"""Toric-geometric training signals for the compact ToricGT adapter.

The module implements small synthetic toric tasks that can be attached to a
byte-model checkpoint without changing the main architecture.  The losses are
deliberately low-rank and low-weight: they make hidden states expose active
faces, moments, bends, binomial relations, affine-Coxeter wall structure, and
noncommutative phase leaves while leaving BPB as the primary objective.
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .cas_backed_losses import relation_tensor_from_certificate


@dataclass(frozen=True)
class ToricGeometryConfig:
    """Configuration for compact toric synthetic probes."""

    enabled: bool = False
    num_exponents: int = 16
    exponent_dim: int = 4
    probe_rank: int = 12
    quant_bits: int = 6
    theta: float = 0.6180339887498948
    beta: float = 1.4142135623730951
    teacher_temperature: float = 0.55
    margin: float = 0.18
    fan_weight: float = 1.0
    bend_weight: float = 0.3
    binom_weight: float = 0.3
    moment_weight: float = 0.5
    coxeter_weight: float = 0.2
    braid_weight: float = 0.1
    leaf_weight: float = 0.25
    max_positions: int = 256
    cas_toric_ideal_certificate_path: str = ""


def _fake_quantize_st(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric straight-through fake quantization for tiny probe matrices."""

    if bits <= 0 or bits >= 16:
        return x
    qmax = float(2 ** (bits - 1) - 1)
    scale = x.detach().abs().amax().clamp_min(1e-8) / qmax
    quantized = torch.round(x / scale).clamp(-qmax, qmax) * scale
    return x + (quantized - x).detach()


def make_exponent_table(num_exponents: int, exponent_dim: int) -> torch.Tensor:
    """Deterministic small Newton-polytope exponent table."""

    num_exponents = max(4, int(num_exponents))
    exponent_dim = max(2, int(exponent_dim))
    rows: list[list[float]] = []
    golden = 0.6180339887498948
    for idx in range(num_exponents):
        row = []
        for dim in range(exponent_dim):
            value = math.sin((idx + 1) * (dim + 2) * golden * math.pi)
            value += 0.5 * math.cos((idx + 3) * (dim + 1) * math.sqrt(2.0))
            row.append(value)
        rows.append(row)
    table = torch.tensor(rows, dtype=torch.float32)
    table = table - table.mean(dim=0, keepdim=True)
    table = table / table.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return table


def make_binomial_relations(exponents: torch.Tensor, max_relations: int = 24) -> torch.Tensor:
    """Find small approximate binomial relations a_i+a_j=a_k+a_l."""

    sums: dict[tuple[int, ...], tuple[int, int]] = {}
    relations: list[tuple[int, int, int, int]] = []
    rounded = torch.round(exponents * 3.0).to(torch.int64)
    n = int(rounded.shape[0])
    for i in range(n):
        for j in range(i, n):
            key = tuple((rounded[i] + rounded[j]).tolist())
            if key in sums:
                k, l = sums[key]
                if len({i, j, k, l}) >= 3:
                    relations.append((i, j, k, l))
                    if len(relations) >= max_relations:
                        return torch.tensor(relations, dtype=torch.long)
            else:
                sums[key] = (i, j)
    if not relations:
        relations = [(0, 1, 2, 3), (0, 2, 1, 3)]
    return torch.tensor(relations[:max_relations], dtype=torch.long)


def _phase_features(
    positions: torch.Tensor,
    exponent_dim: int,
    theta: float,
    beta: float,
) -> torch.Tensor:
    pos = positions.to(torch.float32)
    features = [
        torch.sin(2 * math.pi * theta * pos),
        torch.cos(2 * math.pi * theta * pos),
        torch.sin(2 * math.pi * beta * pos),
        torch.cos(2 * math.pi * beta * pos),
    ]
    power = 2
    while len(features) < exponent_dim:
        scale = theta * (power + 1) + beta / (power + 2)
        features.append(torch.sin(2 * math.pi * scale * pos + 0.17 * power))
        power += 1
    return torch.stack(features[:exponent_dim], dim=-1)


def reflect_affine(x: torch.Tensor, root: torch.Tensor, offset: torch.Tensor | float = 0.0) -> torch.Tensor:
    """Reflect points in the affine hyperplane <root,x>=offset."""

    root = root.to(device=x.device, dtype=x.dtype)
    offset_t = torch.as_tensor(offset, device=x.device, dtype=x.dtype)
    denom = root.pow(2).sum().clamp_min(1e-8)
    signed = (x * root).sum(dim=-1, keepdim=True) - offset_t
    return x - 2.0 * signed / denom * root


class LowRankToricGeometryProbe(nn.Module):
    """Low-rank quantized probe for toric synthetic objectives."""

    def __init__(self, d_model: int, config: ToricGeometryConfig) -> None:
        super().__init__()
        self.config = config
        rank = max(2, int(config.probe_rank))
        num_exponents = max(4, int(config.num_exponents))
        exponent_dim = max(2, int(config.exponent_dim))
        self.hidden_to_rank = nn.Linear(d_model, rank, bias=False)
        self.rank_to_face = nn.Linear(rank, num_exponents, bias=True)
        self.moment_projection = nn.Linear(exponent_dim, 2, bias=False)
        exponents = make_exponent_table(num_exponents, exponent_dim)
        relations, relation_source_exact = self._load_binomial_relations(
            str(config.cas_toric_ideal_certificate_path or ""),
            exponents,
        )
        roots = []
        for idx in range(min(3, exponent_dim - 1)):
            root = torch.zeros(exponent_dim, dtype=torch.float32)
            root[idx] = 1.0
            root[idx + 1] = -1.0
            roots.append(root)
        if not roots:
            roots.append(torch.tensor([1.0, -1.0], dtype=torch.float32))
        self.register_buffer("exponents", exponents)
        self.register_buffer("binomial_relations", relations)
        self.register_buffer("binomial_relation_source_exact", torch.tensor(float(relation_source_exact)))
        self.register_buffer("simple_roots", torch.stack(roots, dim=0))
        nn.init.xavier_uniform_(self.hidden_to_rank.weight)
        nn.init.xavier_uniform_(self.rank_to_face.weight)
        nn.init.zeros_(self.rank_to_face.bias)
        nn.init.xavier_uniform_(self.moment_projection.weight)

    def _probe_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        weight_rank = _fake_quantize_st(self.hidden_to_rank.weight, int(self.config.quant_bits))
        rank = F.linear(hidden.float(), weight_rank)
        weight_face = _fake_quantize_st(self.rank_to_face.weight, int(self.config.quant_bits))
        bias_face = _fake_quantize_st(self.rank_to_face.bias, int(self.config.quant_bits))
        return F.linear(rank, weight_face, bias_face)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not self.config.enabled:
            zero = hidden.float().sum() * 0.0
            return {"toric_geometry_loss": zero}
        if hidden.ndim != 3 or positions.ndim != 2:
            raise ValueError("hidden must be [B,L,D] and positions must be [B,L]")
        max_positions = max(8, int(self.config.max_positions))
        if hidden.shape[1] > max_positions:
            idx = torch.linspace(0, hidden.shape[1] - 1, steps=max_positions, device=hidden.device).long()
            hidden = hidden.index_select(1, idx)
            positions = positions.index_select(1, idx)
            if tokens is not None:
                tokens = tokens.index_select(1, idx)

        exponents = self.exponents.to(device=hidden.device, dtype=torch.float32)
        logits = self._probe_logits(hidden)
        teacher_features = _phase_features(
            positions,
            exponent_dim=exponents.shape[-1],
            theta=float(self.config.theta),
            beta=float(self.config.beta),
        ).to(device=hidden.device)
        teacher_logits = torch.einsum("ble,re->blr", teacher_features, exponents)
        teacher_bias = 0.13 * torch.sin(torch.arange(exponents.shape[0], device=hidden.device, dtype=torch.float32))
        teacher_logits = teacher_logits + teacher_bias.view(1, 1, -1)
        target_face = teacher_logits.argmax(dim=-1)

        face_ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_face.reshape(-1))
        target_logit = logits.gather(-1, target_face.unsqueeze(-1)).squeeze(-1)
        other_logits = logits.masked_fill(F.one_hot(target_face, num_classes=logits.shape[-1]).bool(), -1.0e4)
        margin = target_logit - other_logits.amax(dim=-1)
        fan_margin_loss = torch.relu(float(self.config.margin) - margin).mean()
        fan_loss = face_ce + fan_margin_loss

        temperature = max(float(self.config.teacher_temperature), 1e-4)
        pred_probs = torch.softmax(logits.float() / temperature, dim=-1)
        teacher_probs = torch.softmax(teacher_logits.float() / temperature, dim=-1)
        pred_moment = torch.einsum("blr,re->ble", pred_probs, exponents)
        teacher_moment = torch.einsum("blr,re->ble", teacher_probs, exponents)
        moment_loss = (pred_moment - teacher_moment).pow(2).mean()

        if pred_moment.shape[1] > 2:
            pred_bend = pred_moment[:, 2:] - 2.0 * pred_moment[:, 1:-1] + pred_moment[:, :-2]
            teacher_bend = teacher_moment[:, 2:] - 2.0 * teacher_moment[:, 1:-1] + teacher_moment[:, :-2]
            bend_loss = (pred_bend - teacher_bend).pow(2).mean()
            bend_magnitude = pred_bend.norm(dim=-1).mean()
        else:
            bend_loss = hidden.float().sum() * 0.0
            bend_magnitude = bend_loss.detach()

        relations = self.binomial_relations.to(device=hidden.device)
        relation_source_exact = bool(float(self.binomial_relation_source_exact.detach().cpu()))
        if relations.numel() > 0:
            if relations.dtype == torch.long and relations.ndim == 2 and relations.shape[-1] == 4:
                max_col = int(relations.max().item())
                if max_col >= logits.shape[-1]:
                    raise ValueError(f"binomial relation index {max_col} exceeds logits width {logits.shape[-1]}")
                pred_relation = (
                    logits[..., relations[:, 0]]
                    + logits[..., relations[:, 1]]
                    - logits[..., relations[:, 2]]
                    - logits[..., relations[:, 3]]
                )
                if relation_source_exact:
                    teacher_relation = torch.zeros_like(pred_relation)
                else:
                    teacher_relation = (
                        teacher_logits[..., relations[:, 0]]
                        + teacher_logits[..., relations[:, 1]]
                        - teacher_logits[..., relations[:, 2]]
                        - teacher_logits[..., relations[:, 3]]
                    )
            elif relations.ndim == 2:
                if relations.shape[-1] > logits.shape[-1]:
                    raise ValueError(f"binomial relation width {relations.shape[-1]} exceeds logits width {logits.shape[-1]}")
                rel_float = relations.to(device=hidden.device, dtype=logits.dtype)
                pred_relation = torch.matmul(logits[..., : relations.shape[-1]], rel_float.transpose(0, 1))
                if relation_source_exact:
                    teacher_relation = torch.zeros_like(pred_relation)
                else:
                    teacher_relation = torch.matmul(
                        teacher_logits[..., : relations.shape[-1]],
                        rel_float.transpose(0, 1),
                    )
            else:
                raise ValueError("binomial relation tensor must be [R,4] or [R,C]")
            binom_loss = (pred_relation - teacher_relation).pow(2).mean()
            binom_residual = pred_relation.abs().mean()
        else:
            binom_loss = hidden.float().sum() * 0.0
            binom_residual = binom_loss.detach()

        roots = self.simple_roots.to(device=hidden.device, dtype=torch.float32)
        wall_values = torch.einsum("ble,ae->bla", pred_moment, roots)
        nearest_wall_distance = wall_values.abs().amin(dim=-1).mean()
        coxeter_terms = []
        braid_terms = []
        for root in roots:
            reflected_pred = reflect_affine(pred_moment, root)
            reflected_teacher = reflect_affine(teacher_moment, root)
            coxeter_terms.append((reflected_pred - reflected_teacher).pow(2).mean())
        coxeter_loss = torch.stack(coxeter_terms).mean() if coxeter_terms else moment_loss * 0.0
        if roots.shape[0] >= 2:
            s1 = roots[0]
            s2 = roots[1]
            left_pred = reflect_affine(reflect_affine(reflect_affine(pred_moment, s1), s2), s1)
            right_pred = reflect_affine(reflect_affine(reflect_affine(pred_moment, s2), s1), s2)
            left_teacher = reflect_affine(reflect_affine(reflect_affine(teacher_moment, s1), s2), s1)
            right_teacher = reflect_affine(reflect_affine(reflect_affine(teacher_moment, s2), s1), s2)
            braid_terms.append((left_pred - left_teacher).pow(2).mean())
            braid_terms.append((right_pred - right_teacher).pow(2).mean())
            braid_terms.append((left_pred - right_pred).pow(2).mean())
        braid_loss = torch.stack(braid_terms).mean() if braid_terms else coxeter_loss * 0.0

        phase2 = self.moment_projection(pred_moment).float()
        if phase2.shape[1] > 1:
            delta_pos = (positions[:, 1:].float() - positions[:, :-1].float()).unsqueeze(-1)
            expected = 2 * math.pi * delta_pos * torch.tensor(
                [float(self.config.theta), float(self.config.beta)],
                device=hidden.device,
                dtype=torch.float32,
            )
            phase_delta = phase2[:, 1:, :] - phase2[:, :-1, :] - expected
            leaf_residual = (1.0 - torch.cos(phase_delta)).mean()
        else:
            leaf_residual = moment_loss * 0.0

        total = (
            float(self.config.fan_weight) * fan_loss
            + float(self.config.bend_weight) * bend_loss
            + float(self.config.binom_weight) * binom_loss
            + float(self.config.moment_weight) * moment_loss
            + float(self.config.coxeter_weight) * coxeter_loss
            + float(self.config.braid_weight) * braid_loss
            + float(self.config.leaf_weight) * leaf_residual
        )
        coarse = float(self.config.fan_weight) * fan_loss + float(self.config.moment_weight) * moment_loss
        intermediate = (
            coarse
            + float(self.config.bend_weight) * bend_loss
            + float(self.config.coxeter_weight) * coxeter_loss
            + float(self.config.leaf_weight) * leaf_residual
        )
        full = total
        active_entropy = -(pred_probs.clamp_min(1e-8) * pred_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        active_entropy = active_entropy / math.log(max(2, logits.shape[-1]))
        return {
            "toric_geometry_loss": total,
            "toric_geometry_loss_coarse": coarse,
            "toric_geometry_loss_intermediate": intermediate,
            "toric_geometry_loss_full": full,
            "toric_fan_loss": fan_loss.detach(),
            "toric_active_face_ce": face_ce.detach(),
            "toric_active_face_margin": margin.mean().detach(),
            "toric_active_face_entropy": active_entropy.detach(),
            "toric_bend_loss": bend_loss.detach(),
            "toric_bend_magnitude": bend_magnitude.detach(),
            "toric_binomial_loss": binom_loss.detach(),
            "toric_binomial_residual": binom_residual.detach(),
            "toric_binomial_relation_source_exact": torch.as_tensor(float(relation_source_exact), device=hidden.device),
            "toric_moment_loss": moment_loss.detach(),
            "toric_coxeter_loss": coxeter_loss.detach(),
            "toric_affine_wall_distance": nearest_wall_distance.detach(),
            "toric_braid_loss": braid_loss.detach(),
            "toric_leaf_residual": leaf_residual.detach(),
            "toric_probe_rank": torch.as_tensor(float(self.config.probe_rank), device=hidden.device),
            "toric_probe_quant_bits": torch.as_tensor(float(self.config.quant_bits), device=hidden.device),
        }

    @staticmethod
    def _load_binomial_relations(path: str, exponents: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if not path:
            return make_binomial_relations(exponents), False
        certificate_path = Path(path)
        if not certificate_path.exists():
            raise FileNotFoundError(
                f"CAS toric-ideal certificate not found: {certificate_path}. "
                "Run scripts/build_toric_tropical_certificates.py --all-exact-cas first."
            )
        payload = json.loads(certificate_path.read_text(encoding="utf-8"))
        relations = relation_tensor_from_certificate(payload)
        if relations.ndim != 2:
            raise ValueError(f"CAS toric-ideal relation tensor must be rank 2, got {tuple(relations.shape)}")
        return relations.detach().cpu(), True


@torch.no_grad()
def empirical_toric_shadow_stats_np(hidden: torch.Tensor | list | "np.ndarray", max_points: int = 256) -> dict[str, object]:
    """Compute lightweight empirical fan-cell and bend statistics from hidden states.

    This standalone audit intentionally avoids depending on the probe weights.
    It uses deterministic random Fourier-like directions as pseudo-exponents,
    then reports occupied active cells, margins, fitted local slopes, and bend
    magnitudes for plotting/checkpoint comparison.
    """

    import numpy as np

    points = np.asarray(hidden, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2:
        return {
            "active_faces": [],
            "margins": [],
            "bend_magnitudes": [],
            "pseudo_exponents": [],
            "occupied_fan_cells": 0.0,
            "fan_cell_entropy": 0.0,
            "mean_margin": 0.0,
            "min_margin": 0.0,
            "mean_bend": 0.0,
            "slope_residual": 0.0,
        }
    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, num=max_points).round().astype(int)
        points = points[idx]
    points = points - points.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.maximum(norm, 1e-8)
    d = points.shape[1]
    r = min(16, max(4, d // 8))
    dirs = []
    for i in range(r):
        row = np.sin((np.arange(d, dtype=np.float32) + 1) * (i + 2) * 0.173)
        row += 0.5 * np.cos((np.arange(d, dtype=np.float32) + 3) * (i + 1) * 0.097)
        row = row / max(float(np.linalg.norm(row)), 1e-8)
        dirs.append(row)
    dirs_np = np.stack(dirs, axis=0)
    scores = points @ dirs_np.T
    active = scores.argmax(axis=-1)
    top2 = np.sort(scores, axis=-1)[:, -2:]
    margins = top2[:, 1] - top2[:, 0]
    counts = np.bincount(active, minlength=r).astype(np.float64)
    probs = counts[counts > 0] / max(1.0, counts.sum())
    entropy = float(-(probs * np.log(probs)).sum() / np.log(max(2, r))) if probs.size else 0.0
    slopes = dirs_np[active]
    if slopes.shape[0] > 2:
        bends = slopes[2:] - 2.0 * slopes[1:-1] + slopes[:-2]
        bend_magnitudes = np.linalg.norm(bends, axis=-1)
    else:
        bend_magnitudes = np.zeros(1, dtype=np.float32)
    residuals = []
    for face in np.unique(active):
        mask = active == face
        if int(mask.sum()) < 2:
            continue
        y = scores[mask, int(face)]
        x = points[mask]
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        residuals.append(float(np.mean((x @ coef - y) ** 2)))
    return {
        "active_faces": active.astype(int).tolist(),
        "margins": margins.astype(float).tolist(),
        "bend_magnitudes": bend_magnitudes.astype(float).tolist(),
        "pseudo_exponents": dirs_np.astype(float).tolist(),
        "occupied_fan_cells": float(np.count_nonzero(counts)),
        "fan_cell_entropy": entropy,
        "mean_margin": float(np.mean(margins)),
        "min_margin": float(np.min(margins)),
        "mean_bend": float(np.mean(bend_magnitudes)),
        "slope_residual": float(np.mean(residuals)) if residuals else 0.0,
    }
