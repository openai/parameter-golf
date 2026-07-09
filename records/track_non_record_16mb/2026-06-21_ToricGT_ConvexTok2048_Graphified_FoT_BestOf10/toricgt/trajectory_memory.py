"""Reasoning-trajectory memory and anticipative retrieval heads.

The memory layer is deliberately small.  It is not a retrieval-augmented
language model bolted onto ToricGT.  It stores compact graph-of-thought
trajectory summaries and trains a head to predict which prior trajectory would
be useful to consult.  Offline indices can later feed long-context ring
attention, while the in-batch head gives a cheap training signal today.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .derived_category_metrics import chain_complex_from_edges_np, derived_category_feature_summary
from .gudhi_persistence import (
    GudhiPersistenceConfig,
    torch_persistence_image,
    torch_persistence_landscape,
    vectorized_point_cloud_signature,
)
from .got_trajectory import got_dag_metrics, got_dag_summary_np


@dataclass(frozen=True)
class TrajectoryMemoryConfig:
    projection_dim: int = 128
    max_summary_points: int = 96
    teacher_temperature: float = 0.20
    retrieval_temperature: float = 0.20
    distill_weight: float = 0.25
    quality_weight: float = 0.10
    topology_weight: float = 0.20
    graphcg_weight: float = 0.30
    toric_weight: float = 0.20
    dag_weight: float = 0.20
    derived_weight: float = 0.20
    persistence_weight: float = 0.20
    persistence_max_points: int = 48
    persistence_landscape_layers: int = 3
    persistence_landscape_resolution: int = 24
    persistence_image_resolution: int = 12
    sheaf_gate_min: float = 0.15
    sheaf_gate_threshold: float = 0.12
    sheaf_gate_softness: float = 0.18
    sheaf_ce_weight: float = 1.0


@dataclass
class TrajectoryMemoryRecord:
    record_id: str
    key: list[float]
    value: dict[str, Any]
    dataset: str = ""
    task_family: str = ""
    quality: float = 0.0
    helper_k: float = 0.0
    topology: dict[str, float] | None = None
    toric: dict[str, float] | None = None
    trajectory_graph: dict[str, Any] | None = None
    derived_category: dict[str, Any] | None = None


def _safe_normalize_np(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(denom, 1e-8)


def _persistence_vector_np(
    points: np.ndarray,
    *,
    layers: int = 2,
    landscape_resolution: int = 16,
    image_resolution: int = 8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Exact GUDHI PH signature for offline trajectory-memory records.

    Online training uses differentiable Torch vectorizers over supplied
    birth/death tensors.  Offline memory records should be audit-grade, so this
    path goes through GUDHI simplex trees and GUDHI vectorizers directly.
    """

    if points.shape[0] < 2:
        vector = np.zeros((layers * landscape_resolution + image_resolution * image_resolution + 4,), dtype=np.float32)
        return vector, {
            "backend_gudhi": 1.0,
            "total_persistence": 0.0,
            "max_persistence": 0.0,
            "persistence_entropy": 0.0,
            "entropy": 0.0,
            "vector_norm": 0.0,
        }
    cfg = GudhiPersistenceConfig(
        max_points=int(points.shape[0]),
        max_dimension=2,
        radius_quantile=0.75,
        num_radii=4,
        num_levels=4,
        landscape_resolution=int(landscape_resolution),
        landscape_layers=int(layers),
        image_resolution=int(image_resolution),
        macaulay2_resolutions=False,
    )
    vector, metrics = vectorized_point_cloud_signature(points, cfg)
    entropy = float(metrics.get("persistence_entropy", 0.0))
    metrics["entropy"] = entropy
    return vector, {
        "backend_gudhi": float(metrics.get("backend_gudhi", 1.0)),
        "total_persistence": float(metrics.get("total_persistence", 0.0)),
        "max_persistence": float(metrics.get("max_persistence", 0.0)),
        "entropy": entropy,
        "persistence_entropy": entropy,
        "vector_norm": float(metrics.get("vector_norm", np.linalg.norm(vector))),
        "h0_total_persistence": float(metrics.get("h0_total_persistence", 0.0)),
        "h1_total_persistence": float(metrics.get("h1_total_persistence", 0.0)),
        "h2_total_persistence": float(metrics.get("h2_total_persistence", 0.0)),
        "h0_landscape_norm": float(metrics.get("h0_landscape_norm", 0.0)),
        "h1_landscape_norm": float(metrics.get("h1_landscape_norm", 0.0)),
        "h2_landscape_norm": float(metrics.get("h2_landscape_norm", 0.0)),
        "h0_betti": float(metrics.get("h0_betti", 0.0)),
        "h1_betti": float(metrics.get("h1_betti", 0.0)),
        "h2_betti": float(metrics.get("h2_betti", 0.0)),
        "d_squared_residual": float(metrics.get("d_squared_residual", 0.0)),
    }


def summarize_trajectory_np(
    hidden: np.ndarray,
    *,
    positions: np.ndarray | None = None,
    losses: np.ndarray | None = None,
    edges: np.ndarray | None = None,
    theta: float = 0.6180339887498948,
    beta: float = 1.4142135623730951,
    max_points: int = 96,
) -> dict[str, Any]:
    """Return a compact CPU summary for a reasoning trajectory."""

    points = np.asarray(hidden, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0:
        return {
            "key": [],
            "quality": 0.0,
            "topology": {},
            "toric": {},
        }
    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, num=max_points).round().astype(int)
        points = points[idx]
        if positions is not None:
            positions = np.asarray(positions)[idx]
        if losses is not None:
            losses = np.asarray(losses)[idx]
        if edges is not None:
            edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
            old_to_new = {int(old): int(new) for new, old in enumerate(idx.tolist())}
            remapped: list[tuple[int, int]] = []
            for src, dst in edge_array:
                if int(src) in old_to_new and int(dst) in old_to_new:
                    remapped.append((old_to_new[int(src)], old_to_new[int(dst)]))
            edges = np.asarray(remapped, dtype=np.int64).reshape(-1, 2) if remapped else None
    centered = points - points.mean(axis=0, keepdims=True)
    unit = _safe_normalize_np(centered)
    pooled = unit.mean(axis=0)
    endpoint = unit[-1] - unit[0] if unit.shape[0] > 1 else np.zeros_like(pooled)
    velocity = np.diff(unit, axis=0) if unit.shape[0] > 1 else np.zeros((1, unit.shape[1]), dtype=np.float32)
    speed = np.linalg.norm(velocity, axis=-1)
    if unit.shape[0] > 2:
        accel = np.diff(velocity, axis=0)
        curvature = np.linalg.norm(accel, axis=-1).mean()
    else:
        curvature = 0.0
    if unit.shape[0] >= 3:
        dists = np.linalg.norm(unit[:, None, :] - unit[None, :, :], axis=-1)
        finite = dists[np.triu_indices(unit.shape[0], k=1)]
        radius = float(np.quantile(finite, 0.18)) if finite.size else 0.0
        adjacency = (dists <= radius).astype(np.float32)
        np.fill_diagonal(adjacency, 0.0)
        edge_density = float(adjacency.sum() / max(1, unit.shape[0] * (unit.shape[0] - 1)))
    else:
        radius = 0.0
        edge_density = 0.0
    pos = np.arange(unit.shape[0], dtype=np.float32) if positions is None else np.asarray(positions, dtype=np.float32)
    phase_u = np.stack([np.sin(2.0 * math.pi * theta * pos), np.cos(2.0 * math.pi * theta * pos)], axis=-1).mean(axis=0)
    phase_v = np.stack([np.sin(2.0 * math.pi * beta * pos), np.cos(2.0 * math.pi * beta * pos)], axis=-1).mean(axis=0)
    loss_arr = np.asarray(losses, dtype=np.float32) if losses is not None else np.zeros((unit.shape[0],), dtype=np.float32)
    quality = float(-np.mean(loss_arr)) if loss_arr.size else 0.0
    scalars = np.asarray(
        [
            float(speed.mean()) if speed.size else 0.0,
            float(speed.std()) if speed.size else 0.0,
            float(curvature),
            float(edge_density),
            float(radius),
            float(phase_u[0]),
            float(phase_u[1]),
            float(phase_v[0]),
            float(phase_v[1]),
            quality,
        ],
        dtype=np.float32,
    )
    dag = got_dag_summary_np(unit.shape[0], edges=edges)
    chain = chain_complex_from_edges_np(unit.shape[0], edges=edges, max_vertices=min(max_points, 8))
    persistence_vector, persistence = _persistence_vector_np(unit)
    dag_scalars = np.asarray(
        [
            float(dag.get("branch_count", 0.0)),
            float(dag.get("merge_count", 0.0)),
            float(dag.get("edge_count", 0.0)),
            float(dag.get("back_edge_fraction", 0.0)),
            float(dag.get("simplex_edge_density", 0.0)),
            float(dag.get("max_out_degree", 0.0)),
            float(dag.get("max_in_degree", 0.0)),
        ],
        dtype=np.float32,
    )
    key = np.concatenate([pooled, endpoint, scalars, dag_scalars, persistence_vector], axis=0)
    return {
        "key": key.astype(float).tolist(),
        "quality": quality,
        "topology": {
            "speed_mean": float(scalars[0]),
            "speed_std": float(scalars[1]),
            "curvature": float(scalars[2]),
            "edge_density": float(scalars[3]),
            "radius": float(scalars[4]),
            "got_dag_branch_count": float(dag_scalars[0]),
            "got_dag_merge_count": float(dag_scalars[1]),
            "got_dag_edge_count": float(dag_scalars[2]),
            "got_dag_back_edge_fraction": float(dag_scalars[3]),
            "got_dag_simplex_edge_density": float(dag_scalars[4]),
            "got_dag_max_out_degree": float(dag_scalars[5]),
            "got_dag_max_in_degree": float(dag_scalars[6]),
            "persistence_total": float(persistence["total_persistence"]),
            "persistence_max": float(persistence["max_persistence"]),
            "persistence_entropy": float(persistence["entropy"]),
            "persistence_vector_norm": float(persistence["vector_norm"]),
            "persistence_backend_gudhi": float(persistence["backend_gudhi"]),
            "persistence_h0_total": float(persistence.get("h0_total_persistence", 0.0)),
            "persistence_h1_total": float(persistence.get("h1_total_persistence", 0.0)),
            "persistence_h2_total": float(persistence.get("h2_total_persistence", 0.0)),
            "persistence_h0_landscape_norm": float(persistence.get("h0_landscape_norm", 0.0)),
            "persistence_h1_landscape_norm": float(persistence.get("h1_landscape_norm", 0.0)),
            "persistence_h2_landscape_norm": float(persistence.get("h2_landscape_norm", 0.0)),
            "persistence_h0_betti": float(persistence.get("h0_betti", 0.0)),
            "persistence_h1_betti": float(persistence.get("h1_betti", 0.0)),
            "persistence_h2_betti": float(persistence.get("h2_betti", 0.0)),
            "persistence_d_squared_residual": float(persistence.get("d_squared_residual", 0.0)),
        },
        "toric": {
            "phase_u_sin": float(phase_u[0]),
            "phase_u_cos": float(phase_u[1]),
            "phase_v_sin": float(phase_v[0]),
            "phase_v_cos": float(phase_v[1]),
        },
        "trajectory_graph": {
            "kind": "branch_merge_dag",
            "edges": np.asarray(edges, dtype=np.int64).reshape(-1, 2).astype(int).tolist()
            if edges is not None
            else [],
            **dag,
        },
        "derived_category": {
            "kind": "got_trajectory_chain_complex_summary",
            "chain_complex": chain,
            "persistence_signature": {
                "backend": "gudhi",
                "vectorization": "landscape+persistence_image+silhouette+entropy_vector",
                "homology_dimensions": [0, 1, 2],
            },
        },
    }


class TrajectoryMemoryIndex:
    """A compact cosine-search index for stored reasoning trajectories."""

    def __init__(self) -> None:
        self.records: list[TrajectoryMemoryRecord] = []
        self._keys: np.ndarray | None = None

    def add(self, record: TrajectoryMemoryRecord) -> None:
        if not record.key:
            return
        self.records.append(record)
        self._keys = None

    def build(self) -> None:
        if not self.records:
            self._keys = np.zeros((0, 0), dtype=np.float32)
            return
        keys = np.asarray([record.key for record in self.records], dtype=np.float32)
        self._keys = _safe_normalize_np(keys)

    def search(
        self,
        query_key: list[float] | np.ndarray,
        *,
        top_k: int = 8,
        dataset: str | None = None,
        task_family: str | None = None,
    ) -> list[tuple[TrajectoryMemoryRecord, float]]:
        if self._keys is None:
            self.build()
        if self._keys is None or self._keys.shape[0] == 0:
            return []
        query = np.asarray(query_key, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self._keys.shape[1]:
            return []
        scores = self._keys @ _safe_normalize_np(query[None, :]).reshape(-1)
        order = np.argsort(-scores)
        out: list[tuple[TrajectoryMemoryRecord, float]] = []
        for idx in order.tolist():
            record = self.records[int(idx)]
            if dataset and record.dataset != dataset:
                continue
            if task_family and record.task_family != task_family:
                continue
            out.append((record, float(scores[int(idx)])))
            if len(out) >= top_k:
                break
        return out

    def save_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "TrajectoryMemoryIndex":
        index = cls()
        path = Path(path)
        if not path.exists():
            return index
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                index.add(TrajectoryMemoryRecord(**payload))
        index.build()
        return index


class TrajectoryRetrievalHead(nn.Module):
    """Train an anticipative retrieval score over trajectory summaries.

    The head uses in-batch candidates as a teacher-free approximation to the
    later offline memory task.  The teacher distribution favors trajectories
    with similar GraphCG chart coordinates, similar toric phase summaries,
    similar local topology, and better local NLL quality.
    """

    def __init__(self, d_model: int, config: TrajectoryMemoryConfig | None = None) -> None:
        super().__init__()
        self.config = config or TrajectoryMemoryConfig()
        projection_dim = int(self.config.projection_dim)
        self.query = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, projection_dim), nn.GELU(), nn.Linear(projection_dim, projection_dim))
        self.key = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, projection_dim), nn.GELU(), nn.Linear(projection_dim, projection_dim))
        self.quality_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    @staticmethod
    def _h0_mst_deaths_torch(points: torch.Tensor) -> torch.Tensor:
        n = int(points.shape[0])
        if n < 2:
            return points.new_zeros((0,))
        dist = torch.cdist(points.float(), points.float(), p=2)
        src_idx, dst_idx = torch.triu_indices(n, n, offset=1, device=points.device)
        weights = dist[src_idx, dst_idx]
        order = torch.argsort(weights, stable=True)
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        deaths: list[torch.Tensor] = []
        for edge_id_tensor in order.detach().cpu().tolist():
            edge_id = int(edge_id_tensor)
            src = int(src_idx[edge_id].detach().cpu().item())
            dst = int(dst_idx[edge_id].detach().cpu().item())
            src_root = find(src)
            dst_root = find(dst)
            if src_root == dst_root:
                continue
            if rank[src_root] < rank[dst_root]:
                src_root, dst_root = dst_root, src_root
            parent[dst_root] = src_root
            if rank[src_root] == rank[dst_root]:
                rank[src_root] += 1
            deaths.append(weights[edge_id].detach())
            if len(deaths) == n - 1:
                break
        return torch.stack(deaths) if deaths else points.new_zeros((0,))

    def _persistence_signature(
        self,
        hidden: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, _ = hidden.shape
        max_points = max(2, int(self.config.persistence_max_points))
        layers = max(1, int(self.config.persistence_landscape_layers))
        landscape_resolution = max(4, int(self.config.persistence_landscape_resolution))
        image_resolution = max(4, int(self.config.persistence_image_resolution))
        signature_dim = layers * landscape_resolution + image_resolution * image_resolution + 4
        signatures: list[torch.Tensor] = []
        stats: list[torch.Tensor] = []
        with torch.no_grad():
            for bidx in range(batch):
                points = hidden[bidx, node_mask[bidx]].detach().float()
                if points.shape[0] > max_points:
                    index = torch.linspace(0, points.shape[0] - 1, steps=max_points, device=points.device).round().long()
                    points = points.index_select(0, torch.unique_consecutive(index))
                if points.shape[0] < 2:
                    signatures.append(hidden.new_zeros((signature_dim,), dtype=torch.float32))
                    stats.append(hidden.new_zeros((4,), dtype=torch.float32))
                    continue
                points = points - points.mean(dim=0, keepdim=True)
                points = F.normalize(points, dim=-1)
                deaths = self._h0_mst_deaths_torch(points)
                positive = deaths[deaths > 1e-8]
                scale = positive.median().clamp_min(1e-6) if positive.numel() else deaths.new_tensor(1.0)
                deaths = (deaths / scale).clamp_min(0.0)
                radius_max = torch.maximum(deaths.max() * 1.05, deaths.new_tensor(2.0)) if deaths.numel() else deaths.new_tensor(2.0)
                grid = torch.linspace(
                    0.0,
                    float(radius_max.detach().cpu().item()),
                    steps=landscape_resolution,
                    device=hidden.device,
                    dtype=torch.float32,
                )
                diagram = torch.stack([torch.zeros_like(deaths), deaths], dim=-1) if deaths.numel() else hidden.new_zeros((0, 2), dtype=torch.float32)
                if diagram.shape[0] == 0:
                    landscape = hidden.new_zeros((layers, landscape_resolution), dtype=torch.float32)
                    image = hidden.new_zeros((image_resolution, image_resolution), dtype=torch.float32)
                else:
                    landscape = torch_persistence_landscape(diagram, grid, layers=layers)
                    image_grid = torch.linspace(
                        0.0,
                        float(radius_max.detach().cpu().item()),
                        steps=image_resolution,
                        device=hidden.device,
                        dtype=torch.float32,
                    )
                    image = torch_persistence_image(
                        diagram,
                        image_grid,
                        image_grid,
                        sigma=max(float(radius_max.detach().cpu().item()) / float(image_resolution), 1e-3),
                    )
                total = deaths.sum() if deaths.numel() else hidden.new_zeros((), dtype=torch.float32)
                probs = deaths / total.clamp_min(1e-8) if deaths.numel() else deaths
                entropy = (
                    -(probs * probs.clamp_min(1e-8).log()).sum() / math.log(max(2, int(probs.numel())))
                    if probs.numel()
                    else hidden.new_zeros((), dtype=torch.float32)
                )
                stat = torch.stack(
                    [
                        total.to(device=hidden.device, dtype=torch.float32),
                        (deaths.max() if deaths.numel() else hidden.new_zeros((), dtype=torch.float32)).to(hidden.device),
                        entropy.to(hidden.device, dtype=torch.float32),
                        landscape.norm(p=2).to(hidden.device, dtype=torch.float32),
                    ]
                )
                vector = torch.cat([landscape.reshape(-1), image.reshape(-1), stat], dim=0)
                signatures.append(vector.to(device=hidden.device, dtype=torch.float32))
                stats.append(stat)
        return torch.stack(signatures, dim=0), torch.stack(stats, dim=0)

    def _summary_features(
        self,
        hidden: torch.Tensor,
        target_positions: torch.Tensor | None,
        graphcg_basis: torch.Tensor | None,
        trajectory_node_mask: torch.Tensor | None,
        trajectory_edge_index: torch.Tensor | None,
        trajectory_edge_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        h = hidden.float()
        if trajectory_node_mask is None:
            node_mask = torch.ones(h.shape[:2], device=h.device, dtype=torch.bool)
        else:
            node_mask = trajectory_node_mask.to(device=h.device, dtype=torch.bool)
        denom = node_mask.sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
        pooled = (h * node_mask.unsqueeze(-1).to(h.dtype)).sum(dim=1) / denom
        last_index = node_mask.long().sum(dim=1).clamp_min(1) - 1
        first = h[:, 0, :]
        last = h[torch.arange(h.shape[0], device=h.device), last_index, :]
        endpoint = last - first if h.shape[1] > 1 else torch.zeros_like(pooled)
        summary = pooled + 0.25 * endpoint
        if h.shape[1] > 1:
            velocity = h[:, 1:, :] - h[:, :-1, :]
            speed = velocity.pow(2).mean(dim=-1).sqrt()
            speed_mean = speed.mean(dim=1)
            speed_std = speed.std(dim=1, unbiased=False)
        else:
            speed_mean = h.new_zeros((h.shape[0],))
            speed_std = h.new_zeros((h.shape[0],))
        if graphcg_basis is not None:
            basis = F.normalize(graphcg_basis.float(), dim=-1)
            chart = torch.matmul(F.normalize(pooled, dim=-1), basis.transpose(0, 1))
            chart_probs = torch.softmax(chart, dim=-1)
        else:
            chart_probs = F.normalize(pooled[:, : min(8, pooled.shape[-1])], dim=-1)
        if target_positions is None:
            pos = torch.arange(h.shape[1], device=h.device, dtype=torch.float32)[None, :].expand(h.shape[0], -1)
        else:
            pos = target_positions.float()
        theta = 0.6180339887498948
        beta = 1.4142135623730951
        toric = torch.stack(
            [
                torch.sin(2.0 * math.pi * theta * pos).mean(dim=1),
                torch.cos(2.0 * math.pi * theta * pos).mean(dim=1),
                torch.sin(2.0 * math.pi * beta * pos).mean(dim=1),
                torch.cos(2.0 * math.pi * beta * pos).mean(dim=1),
            ],
            dim=-1,
        )
        dag = got_dag_metrics(
            h,
            node_mask=node_mask,
            edge_index=trajectory_edge_index,
            edge_mask=trajectory_edge_mask,
        )
        topology = torch.stack(
            [
                speed_mean,
                speed_std,
                dag["got_dag_branch_count_batch"],
                dag["got_dag_merge_count_batch"],
                dag["got_dag_simplex_edge_density_batch"],
                dag["got_dag_triangle_density_batch"],
            ],
            dim=-1,
        )
        dag_feature = torch.stack(
            [
                dag["got_dag_branch_count_batch"],
                dag["got_dag_merge_count_batch"],
                dag["got_dag_back_edge_fraction_batch"],
                dag["got_dag_branch_diversity_batch"],
                dag["got_dag_merge_scatter_batch"],
                dag["got_dag_balance_residual_batch"],
            ],
            dim=-1,
        )
        derived_feature = derived_category_feature_summary(
            h,
            node_mask=node_mask,
            edge_index=trajectory_edge_index,
            edge_mask=trajectory_edge_mask,
        )
        persistence, persistence_stats = self._persistence_signature(h, node_mask)
        return {
            "summary": summary,
            "chart": chart_probs,
            "toric": toric,
            "topology": topology,
            "dag": dag_feature,
            "derived": derived_feature,
            "persistence": persistence,
            "persistence_stats": persistence_stats,
        }

    @staticmethod
    def _json_float(value: torch.Tensor | float | int) -> float:
        if torch.is_tensor(value):
            return float(torch.nan_to_num(value.detach().float(), nan=0.0, posinf=1e6, neginf=-1e6).cpu().item())
        if not math.isfinite(float(value)):
            return 0.0
        return float(value)

    @staticmethod
    def _json_vector(values: torch.Tensor) -> list[float]:
        safe = torch.nan_to_num(values.detach().float(), nan=0.0, posinf=1e6, neginf=-1e6).cpu()
        return [float(item) for item in safe.reshape(-1).tolist()]

    @torch.no_grad()
    def trace(
        self,
        hidden: torch.Tensor,
        target_positions: torch.Tensor | None = None,
        per_token_nll: torch.Tensor | None = None,
        *,
        graphcg_basis: torch.Tensor | None = None,
        trajectory_node_mask: torch.Tensor | None = None,
        trajectory_edge_index: torch.Tensor | None = None,
        trajectory_edge_mask: torch.Tensor | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Return a JSON-safe analogical memory retrieval trace.

        The trace mirrors the in-batch teacher used during training.  It is an
        explanation object: each query trajectory reports which other
        trajectories would be retrieved as analogical memories, how much the
        model assigned to them, and which chart/toric/topological/DAG/derived
        components supported that retrieval.
        """

        batch = int(hidden.shape[0])
        top_k = max(0, int(top_k))
        if per_token_nll is None or per_token_nll.shape[0] != batch:
            per_token_nll = hidden.new_zeros(hidden.shape[:2])
        elif per_token_nll.ndim == 1:
            per_token_nll = per_token_nll[:, None]
        if batch < 2 or top_k < 1:
            return {
                "kind": "trajectory_memory_analogical_retrieval_trace",
                "enabled": True,
                "batch_size": batch,
                "top_k": top_k,
                "queries": [],
                "reason": "need_at_least_two_trajectories",
            }

        features = self._summary_features(
            hidden,
            target_positions,
            graphcg_basis,
            trajectory_node_mask,
            trajectory_edge_index,
            trajectory_edge_mask,
        )
        query = F.normalize(self.query(features["summary"].to(hidden.dtype)), dim=-1)
        key = F.normalize(self.key(features["summary"].to(hidden.dtype)), dim=-1)
        retrieval_temperature = max(float(self.config.retrieval_temperature), 1e-4)
        persistence_feature = F.normalize(features["persistence"].float(), dim=-1)
        persistence_sim = persistence_feature @ persistence_feature.transpose(0, 1)
        logits = torch.matmul(query, key.transpose(0, 1)) / retrieval_temperature
        logits = logits + float(self.config.persistence_weight) * persistence_sim.to(logits.dtype) / retrieval_temperature
        diag = torch.eye(batch, device=hidden.device, dtype=torch.bool)
        logits = logits.masked_fill(diag, -1e4)
        probs = torch.softmax(logits, dim=-1)

        quality = -per_token_nll.detach().float().mean(dim=1)
        quality_z = (quality - quality.mean()) / (quality.std(unbiased=False) + 1e-6)
        chart = F.normalize(features["chart"].float(), dim=-1)
        chart_sim = chart @ chart.transpose(0, 1)
        toric = F.normalize(features["toric"].float(), dim=-1)
        toric_sim = toric @ toric.transpose(0, 1)
        topo = features["topology"].float()
        topo_dist = torch.cdist(topo, topo, p=2)
        topo_sim = -topo_dist / (topo_dist.mean() + 1e-6)
        dag_feature = F.normalize(features["dag"].float(), dim=-1)
        dag_sim = dag_feature @ dag_feature.transpose(0, 1)
        derived_feature = F.normalize(features["derived"].float(), dim=-1)
        derived_sim = derived_feature @ derived_feature.transpose(0, 1)
        teacher_raw = (
            float(self.config.graphcg_weight) * chart_sim
            + float(self.config.toric_weight) * toric_sim
            + float(self.config.topology_weight) * topo_sim
            + float(self.config.persistence_weight) * persistence_sim
            + float(self.config.dag_weight) * dag_sim
            + float(self.config.derived_weight) * derived_sim
            + quality_z[None, :]
        )
        teacher_logits = teacher_raw.masked_fill(diag, -1e4) / max(float(self.config.teacher_temperature), 1e-4)
        teacher_probs = torch.softmax(teacher_logits, dim=-1)
        labels = teacher_logits.argmax(dim=-1)

        k = min(top_k, max(1, batch - 1))
        query_rows: list[dict[str, Any]] = []
        dag_fields = [
            "branch_count",
            "merge_count",
            "back_edge_fraction",
            "branch_diversity",
            "merge_scatter",
            "balance_residual",
        ]
        derived_fields = [
            "branch_count",
            "merge_count",
            "back_edge_fraction",
            "simplex_edge_density",
            "triangle_density",
            "branch_diversity",
            "merge_scatter",
            "projective_dimension_norm",
            "regularity_norm",
            "total_betti_log_norm",
            "betti_entropy",
        ]
        for query_index in range(batch):
            candidate_probs, candidate_indices = torch.topk(probs[query_index], k=k)
            candidates: list[dict[str, Any]] = []
            for rank, (prob, candidate_index_tensor) in enumerate(zip(candidate_probs, candidate_indices, strict=True), start=1):
                candidate_index = int(candidate_index_tensor.detach().cpu().item())
                candidates.append(
                    {
                        "rank": int(rank),
                        "candidate_index": candidate_index,
                        "model_logit": self._json_float(logits[query_index, candidate_index]),
                        "retrieval_probability": self._json_float(prob),
                        "teacher_probability": self._json_float(teacher_probs[query_index, candidate_index]),
                        "teacher_raw_score": self._json_float(teacher_raw[query_index, candidate_index]),
                        "is_teacher_argmax": bool(candidate_index == int(labels[query_index].detach().cpu().item())),
                        "components": {
                            "graphcg_chart_similarity": self._json_float(chart_sim[query_index, candidate_index]),
                            "toric_phase_similarity": self._json_float(toric_sim[query_index, candidate_index]),
                            "topology_similarity": self._json_float(topo_sim[query_index, candidate_index]),
                            "persistence_landscape_similarity": self._json_float(
                                persistence_sim[query_index, candidate_index]
                            ),
                            "dag_similarity": self._json_float(dag_sim[query_index, candidate_index]),
                            "derived_category_similarity": self._json_float(derived_sim[query_index, candidate_index]),
                            "candidate_quality_z": self._json_float(quality_z[candidate_index]),
                            "candidate_persistence_total": self._json_float(
                                features["persistence_stats"][candidate_index, 0]
                            ),
                            "candidate_persistence_entropy": self._json_float(
                                features["persistence_stats"][candidate_index, 2]
                            ),
                        },
                    }
                )
            query_rows.append(
                {
                    "query_index": int(query_index),
                    "teacher_argmax_index": int(labels[query_index].detach().cpu().item()),
                    "retrieval_entropy": self._json_float(
                        -(probs[query_index] * probs[query_index].clamp_min(1e-8).log()).sum()
                    ),
                    "quality_z": self._json_float(quality_z[query_index]),
                    "dag_features": dict(zip(dag_fields, self._json_vector(features["dag"][query_index]), strict=True)),
                    "derived_category_features": dict(
                        zip(derived_fields, self._json_vector(features["derived"][query_index]), strict=True)
                    ),
                    "persistence_features": {
                        "total_persistence": self._json_float(features["persistence_stats"][query_index, 0]),
                        "max_persistence": self._json_float(features["persistence_stats"][query_index, 1]),
                        "entropy": self._json_float(features["persistence_stats"][query_index, 2]),
                        "landscape_norm": self._json_float(features["persistence_stats"][query_index, 3]),
                    },
                    "top_candidates": candidates,
                }
            )

        return {
            "kind": "trajectory_memory_analogical_retrieval_trace",
            "enabled": True,
            "memory_source": "in_batch_branch_merge_got_trajectories",
            "analogy_teacher": "weighted_graphcg_toric_topology_persistence_dag_derived_quality",
            "batch_size": batch,
            "top_k": k,
            "weights": {
                "graphcg": float(self.config.graphcg_weight),
                "toric": float(self.config.toric_weight),
                "topology": float(self.config.topology_weight),
                "persistence": float(self.config.persistence_weight),
                "dag": float(self.config.dag_weight),
                "derived_category": float(self.config.derived_weight),
                "quality": 1.0,
            },
            "dag_feature_fields": dag_fields,
            "derived_category_feature_fields": derived_fields,
            "queries": query_rows,
        }

    def forward(
        self,
        hidden: torch.Tensor,
        target_positions: torch.Tensor | None,
        per_token_nll: torch.Tensor,
        *,
        graphcg_basis: torch.Tensor | None = None,
        trajectory_node_mask: torch.Tensor | None = None,
        trajectory_edge_index: torch.Tensor | None = None,
        trajectory_edge_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = hidden.shape[0]
        zero = hidden.new_zeros(())
        if batch < 2:
            return {
                "trajectory_memory_loss": zero,
                "trajectory_memory_ce": zero,
                "trajectory_memory_distill_loss": zero,
                "trajectory_memory_quality_loss": zero,
                "trajectory_memory_recall1": zero,
                "trajectory_memory_entropy": zero,
                "trajectory_memory_teacher_diag_prob": zero,
                "trajectory_memory_score_gap": zero,
                "trajectory_memory_dag_similarity": zero,
                "trajectory_memory_dag_branch_count": zero,
                "trajectory_memory_dag_merge_count": zero,
                "trajectory_memory_derived_similarity": zero,
                "trajectory_memory_derived_projective_dimension": zero,
                "trajectory_memory_derived_regularity": zero,
                "trajectory_memory_persistence_similarity": zero,
                "trajectory_memory_persistence_norm": zero,
                "trajectory_memory_persistence_entropy": zero,
                "trajectory_memory_persistence_total": zero,
                "trajectory_memory_persistence_weight": zero,
                "trajectory_memory_sheaf_gluing_score": zero,
                "trajectory_memory_sheaf_gate": zero,
                "trajectory_memory_selected_chart_similarity": zero,
                "trajectory_memory_selected_toric_similarity": zero,
                "trajectory_memory_selected_topology_similarity": zero,
                "trajectory_memory_selected_dag_similarity": zero,
                "trajectory_memory_selected_derived_similarity": zero,
                "trajectory_memory_selected_persistence_similarity": zero,
            }
        features = self._summary_features(
            hidden,
            target_positions,
            graphcg_basis,
            trajectory_node_mask,
            trajectory_edge_index,
            trajectory_edge_mask,
        )
        query = F.normalize(self.query(features["summary"].to(hidden.dtype)), dim=-1)
        key = F.normalize(self.key(features["summary"].to(hidden.dtype)), dim=-1)
        retrieval_temperature = max(float(self.config.retrieval_temperature), 1e-4)
        with torch.no_grad():
            persistence_feature = F.normalize(features["persistence"].float(), dim=-1)
            persistence_sim = persistence_feature @ persistence_feature.transpose(0, 1)
        logits = torch.matmul(query, key.transpose(0, 1)) / retrieval_temperature
        logits = logits + float(self.config.persistence_weight) * persistence_sim.to(logits.dtype) / retrieval_temperature
        diag = torch.eye(batch, device=hidden.device, dtype=torch.bool)
        logits = logits.masked_fill(diag, -1e4)

        quality = -per_token_nll.detach().float().mean(dim=1)
        quality_z = (quality - quality.mean()) / (quality.std(unbiased=False) + 1e-6)
        with torch.no_grad():
            chart = F.normalize(features["chart"].float(), dim=-1)
            chart_sim = chart @ chart.transpose(0, 1)
            toric = F.normalize(features["toric"].float(), dim=-1)
            toric_sim = toric @ toric.transpose(0, 1)
            topo = features["topology"].float()
            topo_dist = torch.cdist(topo, topo, p=2)
            topo_sim = -topo_dist / (topo_dist.mean() + 1e-6)
            dag_feature = F.normalize(features["dag"].float(), dim=-1)
            dag_sim = dag_feature @ dag_feature.transpose(0, 1)
            derived_feature = F.normalize(features["derived"].float(), dim=-1)
            derived_sim = derived_feature @ derived_feature.transpose(0, 1)
            teacher = (
                float(self.config.graphcg_weight) * chart_sim
                + float(self.config.toric_weight) * toric_sim
                + float(self.config.topology_weight) * topo_sim
                + float(self.config.persistence_weight) * persistence_sim
                + float(self.config.dag_weight) * dag_sim
                + float(self.config.derived_weight) * derived_sim
                + quality_z[None, :]
            )
            teacher = teacher.masked_fill(diag, -1e4) / max(float(self.config.teacher_temperature), 1e-4)
            labels = teacher.argmax(dim=-1)
            teacher_probs = torch.softmax(teacher, dim=-1)
            gather_index = labels.unsqueeze(1)
            selected_chart = chart_sim.gather(1, gather_index).mean()
            selected_toric = toric_sim.gather(1, gather_index).mean()
            selected_topology = topo_sim.gather(1, gather_index).mean()
            selected_dag = dag_sim.gather(1, gather_index).mean()
            selected_derived = derived_sim.gather(1, gather_index).mean()
            selected_persistence = persistence_sim.gather(1, gather_index).mean()
            topology_unit = torch.tanh(selected_topology)
            sheaf_gluing_score = torch.stack(
                [
                    selected_chart,
                    selected_toric,
                    topology_unit,
                    selected_dag,
                    selected_derived,
                    selected_persistence,
                ]
            ).mean()
            sheaf_gate = float(self.config.sheaf_gate_min) + (1.0 - float(self.config.sheaf_gate_min)) * torch.sigmoid(
                (sheaf_gluing_score - float(self.config.sheaf_gate_threshold))
                / max(float(self.config.sheaf_gate_softness), 1e-6)
            )
        ce = F.cross_entropy(logits, labels)
        distill = F.kl_div(torch.log_softmax(logits, dim=-1), teacher_probs, reduction="batchmean")
        quality_pred = self.quality_head(features["summary"].to(hidden.dtype)).squeeze(-1).float()
        quality_loss = F.mse_loss(quality_pred, quality_z)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean() / math.log(max(2, batch - 1))
        pred = logits.argmax(dim=-1)
        top2 = torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1).values
        gap = (top2[:, 0] - top2[:, -1]).mean() if top2.shape[-1] > 1 else zero
        loss = (
            float(self.config.sheaf_ce_weight) * sheaf_gate.to(dtype=ce.dtype) * ce
            + float(self.config.distill_weight) * distill
            + float(self.config.quality_weight) * quality_loss
        )
        return {
            "trajectory_memory_loss": loss,
            "trajectory_memory_ce": ce.detach(),
            "trajectory_memory_distill_loss": distill.detach(),
            "trajectory_memory_quality_loss": quality_loss.detach(),
            "trajectory_memory_recall1": (pred == labels).float().mean().detach(),
            "trajectory_memory_entropy": entropy.detach(),
            "trajectory_memory_teacher_diag_prob": teacher_probs.diagonal().mean().detach(),
            "trajectory_memory_score_gap": gap.detach(),
            "trajectory_memory_dag_similarity": dag_sim.masked_fill(diag, 0.0).mean().detach(),
            "trajectory_memory_dag_branch_count": features["dag"][:, 0].mean().detach(),
            "trajectory_memory_dag_merge_count": features["dag"][:, 1].mean().detach(),
            "trajectory_memory_derived_similarity": derived_sim.masked_fill(diag, 0.0).mean().detach(),
            "trajectory_memory_derived_projective_dimension": features["derived"][:, 7].mean().detach(),
            "trajectory_memory_derived_regularity": features["derived"][:, 8].mean().detach(),
            "trajectory_memory_persistence_similarity": persistence_sim.masked_fill(diag, 0.0).mean().detach(),
            "trajectory_memory_persistence_norm": features["persistence"].float().norm(dim=-1).mean().detach(),
            "trajectory_memory_persistence_entropy": features["persistence_stats"][:, 2].mean().detach(),
            "trajectory_memory_persistence_total": features["persistence_stats"][:, 0].mean().detach(),
            "trajectory_memory_persistence_weight": hidden.new_tensor(float(self.config.persistence_weight)).detach(),
            "trajectory_memory_sheaf_gluing_score": sheaf_gluing_score.detach(),
            "trajectory_memory_sheaf_gate": sheaf_gate.detach(),
            "trajectory_memory_selected_chart_similarity": selected_chart.detach(),
            "trajectory_memory_selected_toric_similarity": selected_toric.detach(),
            "trajectory_memory_selected_topology_similarity": selected_topology.detach(),
            "trajectory_memory_selected_dag_similarity": selected_dag.detach(),
            "trajectory_memory_selected_derived_similarity": selected_derived.detach(),
            "trajectory_memory_selected_persistence_similarity": selected_persistence.detach(),
        }
