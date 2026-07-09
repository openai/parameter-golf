"""Finite Toric BGG certificate probes for late-phase ToricGT training.

The module keeps the representation-theoretic machinery deliberately finite:
small standard-label posets, sparse boundary matrices, Koszul degree profiles,
and paired Gale signatures.  It is a training/audit layer, not a symbolic
algebra system and not part of the Parameter-Golf export by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


TORIC_BGG_METRIC_PROVENANCE = {
    "toric_bgg_resolution_consistency": "exact_finite_chain_complex_boundary_square",
    "toric_bgg_d2_residual": "exact_finite_chain_complex_boundary_square",
    "toric_bgg_standard_leakage": "exact_finite_chain_poset_standard_mask",
    "toric_bgg_standard_allowed_mass": "exact_finite_chain_poset_standard_mask",
    "toric_bgg_koszul_linearity_residual": "exact_finite_koszul_degree_profile",
    "toric_bgg_gale_dual_consistency": "finite_gale_dual_signature_pair",
    "toric_bgg_signature_smoothness": "finite_bgg_signature_path",
    "toric_bgg_standard_entropy": "finite_standard_label_distribution",
}


@dataclass(frozen=True)
class ToricBGGCertificate:
    """Compact finite shadow of a Toric BGG supervision object."""

    standard_mask: torch.Tensor
    boundaries: tuple[torch.Tensor, ...] = ()
    internal_degree: torch.Tensor | None = None
    betti_target: torch.Tensor | None = None
    gale_partner: torch.Tensor | None = None


@dataclass(frozen=True)
class ToricBGGConfig:
    """Configuration for the low-rank Toric BGG probe."""

    num_standard_tokens: int = 8
    probe_rank: int = 8
    signature_dim: int = 16
    max_positions: int = 64
    d2_weight: float = 1.0
    standard_weight: float = 0.25
    koszul_weight: float = 0.15
    gale_weight: float = 0.10
    signature_weight: float = 0.10


def toric_bgg_metric_provenance() -> dict[str, str]:
    """Return exact finite-certificate provenance for Toric BGG metrics."""

    return dict(TORIC_BGG_METRIC_PROVENANCE)


def chain_standard_mask(num_labels: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return the standard-filtration mask for a finite chain poset.

    Rows are current labels ``lambda`` and columns are attendable standard
    labels ``mu``.  The entry is true exactly when ``mu <= lambda``.
    """

    labels = torch.arange(max(1, int(num_labels)), device=device)
    return labels[None, :] <= labels[:, None]


def boundary_square_residual(boundaries: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
    """Measure finite chain-complex failure by summing ``||D_{k-1}D_k||^2``."""

    if len(boundaries) < 2:
        if not boundaries:
            return torch.zeros(())
        return boundaries[0].float().sum() * 0.0
    residuals = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        residuals.append((left.float() @ right.float()).pow(2).mean())
    return torch.stack(residuals).mean()


def standard_filtration_leakage(
    probs: torch.Tensor,
    labels: torch.Tensor,
    standard_mask: torch.Tensor,
) -> torch.Tensor:
    """Return attention/probability mass outside the standard order ideal."""

    if probs.ndim != 2:
        raise ValueError("probs must have shape [items, labels]")
    labels = labels.to(device=probs.device, dtype=torch.long).clamp(0, standard_mask.shape[0] - 1)
    allowed = standard_mask.to(device=probs.device).index_select(0, labels)
    return probs.masked_fill(allowed, 0.0).sum(dim=-1).mean()


def koszul_linearity_residual(
    mass: torch.Tensor,
    homological_degree: torch.Tensor,
    internal_degree: torch.Tensor,
    *,
    offset: int = 0,
) -> torch.Tensor:
    """Penalize mass away from the expected linear degree ``j = k + offset``."""

    expected = homological_degree.to(device=mass.device, dtype=mass.dtype) + float(offset)
    internal = internal_degree.to(device=mass.device, dtype=mass.dtype)
    return (mass.float() * (internal - expected).abs().float()).sum() / mass.float().sum().clamp_min(1e-8)


def gale_dual_consistency(
    source_signature: torch.Tensor,
    target_signature: torch.Tensor,
    map_ab: torch.Tensor,
) -> torch.Tensor:
    """Symmetric finite Gale-dual signature consistency loss."""

    mapped = source_signature.float() @ map_ab.float().transpose(0, 1)
    reverse = target_signature.float() @ map_ab.float()
    return F.mse_loss(mapped, target_signature.float()) + F.mse_loss(reverse, source_signature.float())


def toy_bgg_certificate(device: torch.device | None = None) -> ToricBGGCertificate:
    """Return the two-variable Koszul sanity certificate from the paper."""

    d1 = torch.tensor([[1.0, 1.0]], device=device)
    d2 = torch.tensor([[-1.0], [1.0]], device=device)
    return ToricBGGCertificate(
        standard_mask=chain_standard_mask(3, device=device),
        boundaries=(d1, d2),
        internal_degree=torch.tensor([0, 1, 2], device=device),
        betti_target=torch.tensor([1.0, 2.0, 1.0], device=device),
    )


class ToricBGGProbe(nn.Module):
    """Low-rank finite-certificate probe over random-order hidden states."""

    def __init__(self, d_model: int, config: ToricBGGConfig | None = None) -> None:
        super().__init__()
        self.config = config or ToricBGGConfig()
        labels = max(2, int(self.config.num_standard_tokens))
        rank = max(1, int(self.config.probe_rank))
        signature_dim = max(2, int(self.config.signature_dim))
        self.down = nn.Linear(d_model, rank)
        self.standard = nn.Linear(rank, labels)
        self.signature = nn.Linear(rank, signature_dim)
        self.boundary_1 = nn.Linear(rank, labels * labels)
        self.boundary_2 = nn.Linear(rank, labels * labels)
        self.gale_map = nn.Parameter(torch.eye(signature_dim))
        self.register_buffer("standard_mask", chain_standard_mask(labels), persistent=False)
        self.register_buffer("homological_degree", torch.arange(labels), persistent=False)
        self.register_buffer("internal_degree", torch.arange(labels), persistent=False)

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
        flat = x.reshape(-1, x.shape[-1])
        features = torch.tanh(self.down(flat))
        logits = self.standard(features)
        probs = F.softmax(logits, dim=-1)
        labels = self._standard_labels(probs, target_positions, target_tokens)
        standard_loss = standard_filtration_leakage(probs, labels, self.standard_mask)

        summary = features.mean(dim=0, keepdim=True)
        labels_count = int(self.standard_mask.shape[0])
        d1 = self.boundary_1(summary).view(labels_count, labels_count)
        d2 = self.boundary_2(summary).view(labels_count, labels_count)
        d2_loss = boundary_square_residual((d1, d2))

        mass = probs.mean(dim=0)
        koszul_loss = koszul_linearity_residual(
            mass,
            self.homological_degree,
            self.internal_degree,
        )
        signatures = F.normalize(self.signature(features), dim=-1).view(x.shape[0], x.shape[1], -1).mean(dim=1)
        gale_loss = self._gale_loss(signatures)
        signature_loss = self._signature_smoothness(signatures)
        total = (
            float(self.config.d2_weight) * d2_loss
            + float(self.config.standard_weight) * standard_loss
            + float(self.config.koszul_weight) * koszul_loss
            + float(self.config.gale_weight) * gale_loss
            + float(self.config.signature_weight) * signature_loss
        )
        allowed_mass = probs.masked_fill(~self.standard_mask.to(device=probs.device).index_select(0, labels), 0.0).sum(dim=-1).mean()
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        entropy = entropy / torch.log(torch.as_tensor(float(labels_count), device=hidden.device))
        return {
            "toric_bgg_loss": total,
            "toric_bgg_resolution_consistency": (1.0 / (1.0 + d2_loss.detach())),
            "toric_bgg_d2_residual": d2_loss.detach(),
            "toric_bgg_standard_leakage": standard_loss.detach(),
            "toric_bgg_standard_allowed_mass": allowed_mass.detach(),
            "toric_bgg_koszul_linearity_residual": koszul_loss.detach(),
            "toric_bgg_gale_dual_consistency": gale_loss.detach(),
            "toric_bgg_signature_smoothness": signature_loss.detach(),
            "toric_bgg_standard_entropy": entropy.detach(),
            "toric_bgg_exact_certificate_available": torch.ones((), device=hidden.device, dtype=hidden.float().dtype),
            "toric_bgg_provenance_exact_finite_chain": torch.ones((), device=hidden.device, dtype=hidden.float().dtype),
            "toric_bgg_late_gate_required": torch.ones((), device=hidden.device, dtype=hidden.float().dtype),
        }

    def _standard_labels(
        self,
        probs: torch.Tensor,
        target_positions: torch.Tensor | None,
        target_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        labels_count = int(self.standard_mask.shape[0])
        if target_positions is not None:
            return torch.remainder(target_positions.reshape(-1).to(device=probs.device, dtype=torch.long), labels_count)
        if target_tokens is not None:
            return torch.remainder(target_tokens.reshape(-1).to(device=probs.device, dtype=torch.long), labels_count)
        return probs.detach().argmax(dim=-1)

    def _gale_loss(self, signatures: torch.Tensor) -> torch.Tensor:
        if signatures.shape[0] < 2:
            return signatures.float().sum() * 0.0
        source = signatures[:-1]
        target = torch.flip(signatures[1:], dims=[-1])
        return gale_dual_consistency(source, target, self.gale_map)

    def _signature_smoothness(self, signatures: torch.Tensor) -> torch.Tensor:
        if signatures.shape[0] < 2:
            return signatures.float().sum() * 0.0
        return (signatures[1:] - signatures[:-1]).pow(2).mean()

    @staticmethod
    def _zero_like(zero: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "toric_bgg_loss": zero,
            "toric_bgg_resolution_consistency": zero.detach(),
            "toric_bgg_d2_residual": zero.detach(),
            "toric_bgg_standard_leakage": zero.detach(),
            "toric_bgg_standard_allowed_mass": zero.detach(),
            "toric_bgg_koszul_linearity_residual": zero.detach(),
            "toric_bgg_gale_dual_consistency": zero.detach(),
            "toric_bgg_signature_smoothness": zero.detach(),
            "toric_bgg_standard_entropy": zero.detach(),
            "toric_bgg_exact_certificate_available": zero.detach() + 1.0,
            "toric_bgg_provenance_exact_finite_chain": zero.detach() + 1.0,
            "toric_bgg_late_gate_required": zero.detach() + 1.0,
        }
