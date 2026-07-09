"""Finite toric vector-bundle and sheaf probes for ToricGT training.

This module implements the trainable part of the Klyachko/sheaf story in a
bounded form.  A toric vector bundle on a toric variety is represented by a
shared fiber together with compatible filtrations indexed by fan
one-dimensional cones.  The
probe below uses an exact finite certificate for those filtrations and trains
hidden states to expose:

* boundary one-dimensional-cone labels for tropical/toric compactification strata,
* membership in Klyachko filtration subspaces,
* local splitting over affine toric charts, and
* Cech-style gluing of local sheaf sections across chart overlaps.

The certificate is deliberately small and deterministic.  It is a training and
metric layer, not a replacement for Sage/Macaulay2 computations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ToricVectorBundleConfig:
    """Configuration for the finite Klyachko/sheaf probe."""

    rank: int = 8
    num_one_dimensional_cones: int = 8
    num_rays: int | None = None
    num_cones: int = 8
    filtration_levels: int = 3
    max_positions: int = 128
    temperature: float = 0.25
    one_dimensional_cone_weight: float = 0.20
    ray_weight: float | None = None
    filtration_weight: float = 1.00
    splitting_weight: float = 0.35
    cech_weight: float = 0.25
    cocycle_weight: float = 0.05

    @property
    def effective_num_one_dimensional_cones(self) -> int:
        return int(self.num_one_dimensional_cones if self.num_rays is None else self.num_rays)

    @property
    def effective_one_dimensional_cone_weight(self) -> float:
        return float(self.one_dimensional_cone_weight if self.ray_weight is None else self.ray_weight)


@dataclass(frozen=True)
class KlyachkoBundleCertificate:
    """Finite toric vector-bundle certificate.

    ``filtration_masks[rho, l]`` is the basis mask for the level ``l`` subspace
    of the decreasing filtration attached to the one-dimensional cone indexed
    by ``rho``.  ``chart_frames`` are orthogonal bases for affine charts.
    Transition matrices derived from these frames satisfy the Cech cocycle
    identity exactly up to floating-point round off.
    """

    one_dimensional_cones: torch.Tensor
    cone_one_dimensional_cone_mask: torch.Tensor
    filtration_masks: torch.Tensor
    chart_frames: torch.Tensor
    cone_adjacency: torch.Tensor

    @property
    def rays(self) -> torch.Tensor:
        """Deprecated alias for older tests/scripts."""

        return self.one_dimensional_cones

    @property
    def cone_ray_mask(self) -> torch.Tensor:
        """Deprecated alias for older tests/scripts."""

        return self.cone_one_dimensional_cone_mask

    @property
    def rank(self) -> int:
        return int(self.filtration_masks.shape[-1])

    @property
    def num_one_dimensional_cones(self) -> int:
        return int(self.filtration_masks.shape[0])

    @property
    def num_rays(self) -> int:
        """Deprecated alias for older tests/scripts."""

        return self.num_one_dimensional_cones

    @property
    def num_cones(self) -> int:
        return int(self.cone_one_dimensional_cone_mask.shape[0])

    @property
    def filtration_levels(self) -> int:
        return int(self.filtration_masks.shape[1])


def default_klyachko_certificate(
    *,
    rank: int = 8,
    num_one_dimensional_cones: int = 8,
    num_rays: int | None = None,
    num_cones: int | None = None,
    filtration_levels: int = 3,
    device: torch.device | None = None,
) -> KlyachkoBundleCertificate:
    """Construct a small exact Klyachko-style certificate.

    The fan is the cyclic two-dimensional fan whose maximal cones are spanned
    by adjacent one-dimensional cones.  The filtration masks are nested
    coordinate subspaces; this makes the certificate itself exactly compatible
    while still giving hidden states a nontrivial membership and
    chart-splitting target.
    """

    rank = max(2, int(rank))
    num_one_dimensional_cones = max(
        2,
        int(num_one_dimensional_cones if num_rays is None else num_rays),
    )
    num_cones = max(1, int(num_one_dimensional_cones if num_cones is None else num_cones))
    filtration_levels = max(2, int(filtration_levels))
    device = device or torch.device("cpu")

    angles = torch.arange(num_one_dimensional_cones, device=device, dtype=torch.float32) * (
        2.0 * math.pi / float(num_one_dimensional_cones)
    )
    one_dimensional_cones = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

    cone_one_dimensional_cone_mask = torch.zeros(
        num_cones,
        num_one_dimensional_cones,
        dtype=torch.bool,
        device=device,
    )
    for cone in range(num_cones):
        left = cone % num_one_dimensional_cones
        right = (cone + 1) % num_one_dimensional_cones
        cone_one_dimensional_cone_mask[cone, left] = True
        cone_one_dimensional_cone_mask[cone, right] = True

    coords = torch.arange(rank, device=device)
    filtration_masks = torch.zeros(
        num_one_dimensional_cones,
        filtration_levels,
        rank,
        dtype=torch.float32,
        device=device,
    )
    for one_dimensional_cone in range(num_one_dimensional_cones):
        classes = torch.remainder(coords + one_dimensional_cone, filtration_levels)
        for level in range(filtration_levels):
            filtration_masks[one_dimensional_cone, level] = (classes >= level).to(torch.float32)

    eye = torch.eye(rank, device=device)
    frames = []
    for cone in range(num_cones):
        frame = torch.roll(eye, shifts=cone % rank, dims=0)
        signs = torch.where(
            torch.remainder(torch.arange(rank, device=device) + cone, 2) == 0,
            torch.ones(rank, device=device),
            -torch.ones(rank, device=device),
        )
        frames.append(frame * signs[:, None])
    chart_frames = torch.stack(frames, dim=0)

    cone_adjacency = torch.zeros(num_cones, num_cones, dtype=torch.bool, device=device)
    for cone in range(num_cones):
        cone_adjacency[cone, (cone - 1) % num_cones] = True
        cone_adjacency[cone, (cone + 1) % num_cones] = True
    cone_adjacency.fill_diagonal_(False)

    return KlyachkoBundleCertificate(
        one_dimensional_cones=one_dimensional_cones,
        cone_one_dimensional_cone_mask=cone_one_dimensional_cone_mask,
        filtration_masks=filtration_masks,
        chart_frames=chart_frames,
        cone_adjacency=cone_adjacency,
    )


def klyachko_nesting_residual(filtration_masks: torch.Tensor) -> torch.Tensor:
    """Return exact finite residual for decreasing filtrations."""

    if filtration_masks.ndim != 3 or filtration_masks.shape[1] < 2:
        return filtration_masks.float().sum() * 0.0
    coarse = filtration_masks[:, :-1, :].float()
    fine = filtration_masks[:, 1:, :].float()
    return F.relu(fine - coarse).pow(2).mean()


def cech_cocycle_residual(chart_frames: torch.Tensor, cone_adjacency: torch.Tensor) -> torch.Tensor:
    """Check the chart transition cocycle ``T_ac = T_bc T_ab``."""

    if chart_frames.ndim != 3 or chart_frames.shape[0] < 3:
        return chart_frames.float().sum() * 0.0
    frames = chart_frames.float()
    adjacency = cone_adjacency.to(device=frames.device)
    terms = []
    count = frames.shape[0]
    for a in range(count):
        for b in range(count):
            if not bool(adjacency[a, b]):
                continue
            for c in range(count):
                if bool(adjacency[b, c]) and bool(adjacency[a, c]):
                    t_ab = frames[b] @ frames[a].transpose(0, 1)
                    t_bc = frames[c] @ frames[b].transpose(0, 1)
                    t_ac = frames[c] @ frames[a].transpose(0, 1)
                    terms.append((t_bc @ t_ab - t_ac).pow(2).mean())
    if not terms:
        return frames.sum() * 0.0
    return torch.stack(terms).mean()


class ToricVectorBundleProbe(nn.Module):
    """Training-only Klyachko vector-bundle/sheaf probe."""

    def __init__(self, d_model: int, config: ToricVectorBundleConfig | None = None) -> None:
        super().__init__()
        self.config = config or ToricVectorBundleConfig()
        rank = max(2, int(self.config.rank))
        num_one_dimensional_cones = max(2, int(self.config.effective_num_one_dimensional_cones))
        levels = max(2, int(self.config.filtration_levels))
        self.fiber = nn.Linear(d_model, rank)
        self.one_dimensional_cone_head = nn.Linear(rank, num_one_dimensional_cones)
        self.level_head = nn.Linear(rank, levels)
        cert = default_klyachko_certificate(
            rank=rank,
            num_one_dimensional_cones=num_one_dimensional_cones,
            num_cones=int(self.config.num_cones),
            filtration_levels=levels,
        )
        self.register_buffer("one_dimensional_cones", cert.one_dimensional_cones, persistent=False)
        self.register_buffer(
            "cone_one_dimensional_cone_mask",
            cert.cone_one_dimensional_cone_mask,
            persistent=False,
        )
        self.register_buffer("filtration_masks", cert.filtration_masks, persistent=False)
        self.register_buffer("chart_frames", cert.chart_frames, persistent=False)
        self.register_buffer("cone_adjacency", cert.cone_adjacency, persistent=False)

    def forward(
        self,
        hidden: torch.Tensor,
        target_positions: torch.Tensor | None = None,
        target_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if hidden.ndim != 3 or hidden.shape[1] < 2:
            zero = hidden.float().sum() * 0.0
            return self._zero_like(zero)
        max_positions = max(2, int(self.config.max_positions))
        x = hidden.float()
        if x.shape[1] > max_positions:
            index = torch.linspace(0, x.shape[1] - 1, steps=max_positions, device=x.device).round().long()
            x = x.index_select(1, index)
            if target_positions is not None:
                target_positions = target_positions.index_select(1, index)
            if target_tokens is not None:
                target_tokens = target_tokens.index_select(1, index)

        fiber = torch.tanh(self.fiber(x))
        flat_fiber = fiber.reshape(-1, fiber.shape[-1])
        one_dimensional_cone_logits = self.one_dimensional_cone_head(flat_fiber)
        level_logits = self.level_head(flat_fiber)
        one_dimensional_cone_probs = F.softmax(
            one_dimensional_cone_logits / max(float(self.config.temperature), 1e-4),
            dim=-1,
        )
        level_probs = F.softmax(level_logits / max(float(self.config.temperature), 1e-4), dim=-1)

        one_dimensional_cone_labels = self._one_dimensional_cone_labels(
            one_dimensional_cone_probs,
            target_positions,
        )
        level_labels = self._level_labels(level_probs, target_positions, target_tokens)
        one_dimensional_cone_ce = F.cross_entropy(one_dimensional_cone_logits, one_dimensional_cone_labels)
        level_ce = F.cross_entropy(level_logits, level_labels)
        filtration = self._filtration_membership(flat_fiber, one_dimensional_cone_labels, level_labels)
        one_dimensional_cone_probs_view = one_dimensional_cone_probs.view(fiber.shape[0], fiber.shape[1], -1)
        splitting = self._cone_splitting_residual(fiber, one_dimensional_cone_probs_view)
        gluing = self._cech_gluing_residual(fiber, one_dimensional_cone_probs_view)
        nesting = klyachko_nesting_residual(self.filtration_masks)
        cocycle = cech_cocycle_residual(self.chart_frames, self.cone_adjacency)
        total = (
            float(self.config.effective_one_dimensional_cone_weight) * (
                one_dimensional_cone_ce + 0.25 * level_ce
            )
            + float(self.config.filtration_weight) * filtration
            + float(self.config.splitting_weight) * splitting
            + float(self.config.cech_weight) * gluing
            + float(self.config.cocycle_weight) * (nesting + cocycle)
        )
        one_dimensional_cone_entropy = -(
            one_dimensional_cone_probs.clamp_min(1e-8) * one_dimensional_cone_probs.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        one_dimensional_cone_entropy = one_dimensional_cone_entropy / math.log(
            max(2, int(self.one_dimensional_cones.shape[0]))
        )
        active_one_dimensional_cone_mass = one_dimensional_cone_probs.max(dim=-1).values.mean()
        return {
            "toric_vector_bundle_1d_cone_ce_loss": total,
            "toric_vector_bundle_1d_cone_ce": one_dimensional_cone_ce.detach(),
            "toric_vector_bundle_filtration_level_ce": level_ce.detach(),
            "toric_vector_bundle_filtration_residual": filtration.detach(),
            "toric_vector_bundle_1d_cone_klyachko_nesting_residual": nesting.detach(),
            "toric_vector_bundle_cone_splitting_residual": splitting.detach(),
            "toric_vector_bundle_cech_gluing_residual": gluing.detach(),
            "toric_sheaf_chart_gluing_residual": gluing.detach(),
            "toric_sheaf_cocycle_residual": cocycle.detach(),
            "toric_vector_bundle_1d_cone_entropy": one_dimensional_cone_entropy.detach(),
            "toric_vector_bundle_active_1d_cone_mass": active_one_dimensional_cone_mass.detach(),
            "toric_vector_bundle_rank": torch.as_tensor(float(self.filtration_masks.shape[-1]), device=hidden.device),
            "toric_vector_bundle_num_1d_cones": torch.as_tensor(
                float(self.one_dimensional_cones.shape[0]),
                device=hidden.device,
            ),
            "toric_vector_bundle_filtration_levels": torch.as_tensor(
                float(self.filtration_masks.shape[1]),
                device=hidden.device,
            ),
        }

    @property
    def rays(self) -> torch.Tensor:
        """Deprecated module alias for older scripts."""

        return self.one_dimensional_cones

    @property
    def cone_ray_mask(self) -> torch.Tensor:
        """Deprecated module alias for older scripts."""

        return self.cone_one_dimensional_cone_mask

    def _one_dimensional_cone_labels(
        self,
        probs: torch.Tensor,
        target_positions: torch.Tensor | None,
    ) -> torch.Tensor:
        num_one_dimensional_cones = int(self.one_dimensional_cones.shape[0])
        if target_positions is not None:
            return torch.remainder(
                target_positions.reshape(-1).to(device=probs.device, dtype=torch.long),
                num_one_dimensional_cones,
            )
        return probs.detach().argmax(dim=-1)

    def _level_labels(
        self,
        probs: torch.Tensor,
        target_positions: torch.Tensor | None,
        target_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        levels = int(self.filtration_masks.shape[1])
        if target_tokens is not None:
            return torch.remainder(target_tokens.reshape(-1).to(device=probs.device, dtype=torch.long), levels)
        if target_positions is not None:
            return torch.remainder(target_positions.reshape(-1).to(device=probs.device, dtype=torch.long), levels)
        return probs.detach().argmax(dim=-1)

    def _filtration_membership(
        self,
        flat_fiber: torch.Tensor,
        one_dimensional_cone_labels: torch.Tensor,
        level_labels: torch.Tensor,
    ) -> torch.Tensor:
        masks = self.filtration_masks.to(device=flat_fiber.device, dtype=flat_fiber.dtype)
        selected = masks[one_dimensional_cone_labels, level_labels]
        residual = flat_fiber * (1.0 - selected)
        denom = flat_fiber.pow(2).mean(dim=-1).clamp_min(1e-6)
        return (residual.pow(2).mean(dim=-1) / denom).mean()

    def _cone_weights(self, one_dimensional_cone_probs: torch.Tensor) -> torch.Tensor:
        cone_mask = self.cone_one_dimensional_cone_mask.to(
            device=one_dimensional_cone_probs.device,
            dtype=one_dimensional_cone_probs.dtype,
        )
        weights = one_dimensional_cone_probs @ cone_mask.transpose(0, 1)
        return weights / cone_mask.sum(dim=-1).clamp_min(1.0)

    def _cone_splitting_residual(
        self,
        fiber: torch.Tensor,
        one_dimensional_cone_probs: torch.Tensor,
    ) -> torch.Tensor:
        cone_weights = self._cone_weights(one_dimensional_cone_probs)
        frames = self.chart_frames.to(device=fiber.device, dtype=fiber.dtype)
        terms = []
        flat = fiber.reshape(-1, fiber.shape[-1])
        flat_weights = cone_weights.reshape(-1, cone_weights.shape[-1])
        for cone in range(frames.shape[0]):
            weights = flat_weights[:, cone]
            if weights.detach().sum() <= 1e-8:
                continue
            local = flat @ frames[cone].transpose(0, 1)
            centered = local - (weights[:, None] * local).sum(dim=0, keepdim=True) / weights.sum().clamp_min(1e-6)
            cov = (centered * weights[:, None]).transpose(0, 1) @ centered / weights.sum().clamp_min(1e-6)
            off_diag = cov - torch.diag(torch.diagonal(cov))
            terms.append(off_diag.pow(2).mean())
        if not terms:
            return fiber.float().sum() * 0.0
        return torch.stack(terms).mean()

    def _cech_gluing_residual(
        self,
        fiber: torch.Tensor,
        one_dimensional_cone_probs: torch.Tensor,
    ) -> torch.Tensor:
        cone_weights = self._cone_weights(one_dimensional_cone_probs)
        frames = self.chart_frames.to(device=fiber.device, dtype=fiber.dtype)
        adjacency = self.cone_adjacency.to(device=fiber.device)
        flat = fiber.reshape(-1, fiber.shape[-1])
        flat_weights = cone_weights.reshape(-1, cone_weights.shape[-1])
        local_means = []
        valid = []
        for cone in range(frames.shape[0]):
            weights = flat_weights[:, cone]
            mass = weights.sum()
            if mass.detach() <= 1e-8:
                local_means.append(torch.zeros(frames.shape[-1], device=fiber.device, dtype=fiber.dtype))
                valid.append(False)
                continue
            local = flat @ frames[cone].transpose(0, 1)
            local_means.append((weights[:, None] * local).sum(dim=0) / mass.clamp_min(1e-6))
            valid.append(True)
        terms = []
        for left in range(frames.shape[0]):
            if not valid[left]:
                continue
            for right in range(frames.shape[0]):
                if not valid[right] or not bool(adjacency[left, right]):
                    continue
                transition = frames[right] @ frames[left].transpose(0, 1)
                transported = local_means[left] @ transition.transpose(0, 1)
                terms.append((transported - local_means[right]).pow(2).mean())
        if not terms:
            return fiber.float().sum() * 0.0
        return torch.stack(terms).mean()

    @staticmethod
    def _zero_like(zero: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "toric_vector_bundle_1d_cone_ce_loss": zero,
            "toric_vector_bundle_1d_cone_ce": zero.detach(),
            "toric_vector_bundle_filtration_level_ce": zero.detach(),
            "toric_vector_bundle_filtration_residual": zero.detach(),
            "toric_vector_bundle_1d_cone_klyachko_nesting_residual": zero.detach(),
            "toric_vector_bundle_cone_splitting_residual": zero.detach(),
            "toric_vector_bundle_cech_gluing_residual": zero.detach(),
            "toric_sheaf_chart_gluing_residual": zero.detach(),
            "toric_sheaf_cocycle_residual": zero.detach(),
            "toric_vector_bundle_1d_cone_entropy": zero.detach(),
            "toric_vector_bundle_active_1d_cone_mass": zero.detach(),
            "toric_vector_bundle_rank": zero.detach(),
            "toric_vector_bundle_num_1d_cones": zero.detach(),
            "toric_vector_bundle_filtration_levels": zero.detach(),
        }
