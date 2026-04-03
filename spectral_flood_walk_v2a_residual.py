from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def parse_orders_csv(spec: str) -> tuple[int, ...]:
    if not spec.strip():
        return ()
    return tuple(int(part.strip()) for part in spec.split(",") if part.strip())


def build_eval_score_mask(batch_starts: list[int], seq_len: int, stride: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((len(batch_starts), seq_len), dtype=torch.bool, device=device)
    tail = min(stride, seq_len)
    for row, start in enumerate(batch_starts):
        if start == 0:
            mask[row, :] = True
        else:
            mask[row, seq_len - tail :] = True
    return mask


class ResidualBasis(nn.Module):
    def __init__(self, basis: torch.Tensor, num_orders: int) -> None:
        super().__init__()
        if basis.ndim != 2:
            raise ValueError(f"basis must be rank-2, got shape {tuple(basis.shape)}")
        self.basis = nn.Parameter(basis.to(dtype=torch.float32), requires_grad=False)
        self.order_scale_logits = nn.Parameter(torch.zeros(num_orders, dtype=torch.float32), requires_grad=False)

    @classmethod
    def from_embedding_svd(cls, weight: torch.Tensor, rank: int, num_orders: int) -> "ResidualBasis":
        matrix = weight.detach().float()
        u, _, _ = torch.linalg.svd(matrix, full_matrices=False)
        basis = u[:, :rank].contiguous()
        return cls(basis=basis, num_orders=num_orders)

    def order_scales(self) -> torch.Tensor:
        return torch.sigmoid(self.order_scale_logits)

    def project(self, coeffs: torch.Tensor) -> torch.Tensor:
        return coeffs @ self.basis.transpose(0, 1)

    def correction(self, order_coeffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scales = self.order_scales().to(dtype=order_coeffs.dtype)
        combined = (order_coeffs * scales.view(1, 1, -1, 1)).sum(dim=2)
        return self.project(combined), combined

    def projected_direction(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        flat_probs = probs.reshape(-1, probs.size(-1))
        expected = flat_probs @ self.basis
        target_basis = self.basis[targets.reshape(-1)]
        return (target_basis - expected).reshape(*targets.shape, -1)


class ResidualRouter:
    def __init__(self, orders: tuple[int, ...], table_size: int, device: torch.device) -> None:
        self.orders = orders
        self.table_size = table_size
        self.device = device
        self.offset_mul = torch.tensor(
            [6361, 10007, 16931, 26699, 39367, 58757, 75941, 100151],
            dtype=torch.int64,
            device=device,
        )
        self.mix_prime = 1_000_003

    def context_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq = input_ids.shape
        if not self.orders:
            return torch.empty((batch, seq, 0), dtype=torch.long, device=input_ids.device)
        token_ids = input_ids.to(torch.int64) + 1
        mod = max(self.table_size - 1, 1)
        outputs: list[torch.Tensor] = []
        for order in self.orders:
            ids = torch.zeros((batch, seq), dtype=torch.long, device=input_ids.device)
            if order <= seq:
                length = seq - order + 1
                acc = torch.zeros((batch, length), dtype=torch.int64, device=input_ids.device)
                for offset in range(order):
                    segment = token_ids[:, offset : offset + length]
                    acc = (acc * self.mix_prime + segment * self.offset_mul[offset]) % mod
                ids[:, order - 1 :] = (acc + 1).to(torch.long)
            outputs.append(ids)
        return torch.stack(outputs, dim=2)


@dataclass
class ResidualTableStats:
    resident_mb: float
    compact_mb_estimate: float
    non_empty_fraction: float
    mean_non_empty_slots: float
    max_non_empty_slots: float
    mean_coeff_norm: float
    mean_hits: float


class ResidualTables:
    def __init__(
        self,
        *,
        orders: tuple[int, ...],
        table_size: int,
        rank: int,
        device: torch.device,
        decay: float,
    ) -> None:
        self.orders = orders
        self.table_size = table_size
        self.rank = rank
        self.device = device
        self.decay = float(decay)
        self.coeffs = torch.zeros((len(orders), table_size, rank), dtype=torch.float32, device=device)
        self.counts = torch.zeros((len(orders), table_size), dtype=torch.float32, device=device)
        self.hits = torch.zeros((len(orders), table_size), dtype=torch.float32, device=device)

    def clone(self) -> "ResidualTables":
        cloned = ResidualTables(
            orders=self.orders,
            table_size=self.table_size,
            rank=self.rank,
            device=self.device,
            decay=self.decay,
        )
        cloned.coeffs = self.coeffs.clone()
        cloned.counts = self.counts.clone()
        cloned.hits = self.hits.clone()
        return cloned

    def resident_bytes(self) -> int:
        total = 0
        for tensor in (self.coeffs, self.counts, self.hits):
            total += int(tensor.numel() * tensor.element_size())
        return total

    def compact_bytes_estimate(self) -> int:
        coeff_bytes = int(self.coeffs.numel())
        scale_bytes = int(self.coeffs.size(0) * self.coeffs.size(1) * 2)
        stat_bytes = int((self.counts.numel() + self.hits.numel()) * 2)
        return coeff_bytes + scale_bytes + stat_bytes

    def lookup(self, context_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if context_ids.numel() == 0:
            empty = torch.zeros((*context_ids.shape[:2], 0, self.rank), dtype=torch.float32, device=self.device)
            return empty, {"active_slots": 0.0, "mean_coeff_norm": 0.0}
        gathered = []
        active_slots = 0.0
        coeff_norms: list[float] = []
        for order_idx in range(context_ids.size(2)):
            ids = context_ids[:, :, order_idx]
            coeff = self.coeffs[order_idx, ids]
            gathered.append(coeff)
            if ids.numel() > 0:
                active_slots += float((self.counts[order_idx, ids] > 0).float().mean().item())
                coeff_norms.append(float(coeff.norm(dim=-1).mean().item()))
        stats = {
            "active_slots": active_slots / max(context_ids.size(2), 1),
            "mean_coeff_norm": sum(coeff_norms) / max(len(coeff_norms), 1),
        }
        return torch.stack(gathered, dim=2), stats

    @torch.no_grad()
    def note_reads(self, context_ids: torch.Tensor) -> None:
        if context_ids.numel() == 0:
            return
        for order_idx in range(context_ids.size(2)):
            ids = context_ids[:, :, order_idx].reshape(-1)
            self.hits[order_idx].scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

    @torch.no_grad()
    def update(
        self,
        context_ids: torch.Tensor,
        projected_direction: torch.Tensor,
        order_scales: torch.Tensor,
        step_size: float,
        score_mask: torch.Tensor | None = None,
    ) -> None:
        if context_ids.numel() == 0:
            return
        scale_values = order_scales.to(device=self.device, dtype=torch.float32)
        direction = projected_direction.to(device=self.device, dtype=torch.float32)
        weights = None
        if score_mask is not None:
            weights = score_mask.to(device=self.device, dtype=torch.float32).reshape(-1)
        for order_idx in range(context_ids.size(2)):
            ids = context_ids[:, :, order_idx].reshape(-1)
            order_delta = (direction.reshape(-1, self.rank) * scale_values[order_idx]) * step_size
            if weights is not None:
                order_delta = order_delta * weights.unsqueeze(-1)
            self.coeffs[order_idx].mul_(self.decay)
            self.coeffs[order_idx].scatter_add_(0, ids.unsqueeze(-1).expand(-1, self.rank), order_delta)
            count_values = torch.ones_like(ids, dtype=torch.float32)
            if weights is not None:
                count_values = weights
            self.counts[order_idx].scatter_add_(0, ids, count_values)

    def stats(self) -> ResidualTableStats:
        non_empty = (self.counts > 0).float()
        per_order_non_empty = non_empty.sum(dim=1)
        coeff_norm = self.coeffs.norm(dim=-1)
        return ResidualTableStats(
            resident_mb=float(self.resident_bytes() / (1024.0 * 1024.0)),
            compact_mb_estimate=float(self.compact_bytes_estimate() / (1024.0 * 1024.0)),
            non_empty_fraction=float(non_empty.mean().item()),
            mean_non_empty_slots=float(per_order_non_empty.mean().item()),
            max_non_empty_slots=float(per_order_non_empty.max().item()),
            mean_coeff_norm=float(coeff_norm.mean().item()),
            mean_hits=float(self.hits.mean().item()),
        )

    def export_npz(self, basis: ResidualBasis, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        basis_arr = basis.basis.detach().cpu().numpy()
        basis_scale = max(float(np.max(np.abs(basis_arr))) / 127.0, 1e-8) if basis_arr.size else 1e-8
        basis_q = np.clip(np.round(basis_arr / basis_scale), -127, 127).astype(np.int8, copy=False)
        coeff_arr = self.coeffs.detach().cpu().numpy()
        coeff_scale = np.maximum(np.max(np.abs(coeff_arr), axis=-1, keepdims=True) / 127.0, 1e-8).astype(np.float32)
        coeff_q = np.clip(np.round(coeff_arr / coeff_scale), -127, 127).astype(np.int8, copy=False)
        np.savez_compressed(
            path,
            basis_q=basis_q,
            basis_scale=np.array([basis_scale], dtype=np.float32),
            coeff_q=coeff_q,
            coeff_scale=coeff_scale.astype(np.float16),
            order_scales=basis.order_scales().detach().cpu().numpy().astype(np.float16),
            counts=self.counts.detach().cpu().numpy().astype(np.float16),
        )
        return int(path.stat().st_size)


def probability_sum_deviation(logits: torch.Tensor, delta_logits: torch.Tensor | None = None) -> float:
    corrected = logits if delta_logits is None else logits + delta_logits.to(dtype=logits.dtype)
    probs = torch.softmax(corrected.float(), dim=-1)
    return float((probs.sum(dim=-1) - 1.0).abs().max().item())
