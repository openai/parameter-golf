"""Derived-category trajectory metrics for branch/merge reasoning DAGs.

The train-time loss compares graph-of-thought trajectories as finite chain
complexes.  The audit path emits actual combinatorial objects: chain complexes
and symbolic multigraded Stanley-Reisner/Taylor resolution certificates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .got_trajectory import default_branch_merge_edges, got_dag_metrics
from .symbolic_multigraded_resolution import (
    cyclic_stanley_reisner_resolution_certificate,
    cyclic_stanley_reisner_resolution_metrics,
)


@dataclass(frozen=True)
class DerivedCategoryConfig:
    max_vertices: int = 8
    max_edges: int = 64
    transport_temperature: float = 0.25
    chain_weight: float = 0.45
    cone_weight: float = 0.25
    betti_weight: float = 0.10
    resolution_weight: float = 0.20
    eps: float = 1e-6


def _rank_over_q(matrix: np.ndarray) -> int:
    rows = [[Fraction(int(value)) for value in row] for row in np.asarray(matrix, dtype=np.int64).tolist()]
    if not rows:
        return 0
    num_rows = len(rows)
    num_cols = len(rows[0]) if rows[0] else 0
    rank = 0
    pivot_col = 0
    while rank < num_rows and pivot_col < num_cols:
        pivot = None
        for row in range(rank, num_rows):
            if rows[row][pivot_col] != 0:
                pivot = row
                break
        if pivot is None:
            pivot_col += 1
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_value = rows[rank][pivot_col]
        rows[rank] = [value / pivot_value for value in rows[rank]]
        for row in range(num_rows):
            if row == rank:
                continue
            factor = rows[row][pivot_col]
            if factor == 0:
                continue
            rows[row] = [value - factor * pivot_entry for value, pivot_entry in zip(rows[row], rows[rank])]
        rank += 1
        pivot_col += 1
    return rank


def _valid_edges_np(num_vertices: int, edges: np.ndarray | None, max_edges: int) -> list[tuple[int, int]]:
    n = max(0, int(num_vertices))
    if edges is None:
        edge_array = default_branch_merge_edges(n).cpu().numpy()
    else:
        edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for raw_src, raw_dst in edge_array[: max(0, int(max_edges))]:
        src = int(raw_src)
        dst = int(raw_dst)
        edge = (src, dst)
        if src < 0 or dst < 0 or src >= n or dst >= n or src == dst or edge in seen:
            continue
        seen.add(edge)
        out.append(edge)
    return out


def _triangles_from_edges(num_vertices: int, edges: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    undirected = {tuple(sorted(edge)) for edge in edges}
    triangles: list[tuple[int, int, int]] = []
    for simplex in combinations(range(num_vertices), 3):
        if all(tuple(sorted(pair)) in undirected for pair in combinations(simplex, 2)):
            triangles.append(tuple(int(v) for v in simplex))
    return triangles


def chain_complex_from_edges_np(
    num_vertices: int,
    edges: np.ndarray | None = None,
    *,
    max_vertices: int = 8,
    max_edges: int = 64,
) -> dict[str, Any]:
    """Return the finite simplicial chain complex induced by a GoT DAG."""

    n = max(0, min(int(num_vertices), int(max_vertices)))
    valid_edges = _valid_edges_np(n, edges, max_edges)
    triangles = _triangles_from_edges(n, valid_edges)
    edge_to_col = {edge: index for index, edge in enumerate(valid_edges)}
    d1 = np.zeros((n, len(valid_edges)), dtype=np.int64)
    for col, (src, dst) in enumerate(valid_edges):
        d1[src, col] -= 1
        d1[dst, col] += 1

    d2 = np.zeros((len(valid_edges), len(triangles)), dtype=np.int64)
    for col, (a, b, c) in enumerate(triangles):
        for edge, coeff in [((b, c), 1), ((a, c), -1), ((a, b), 1)]:
            row = edge_to_col.get(edge)
            sign = 1
            if row is None:
                row = edge_to_col.get((edge[1], edge[0]))
                sign = -1
            if row is not None:
                d2[row, col] += coeff * sign

    rank_d1 = _rank_over_q(d1)
    rank_d2 = _rank_over_q(d2)
    betti0 = n - rank_d1
    betti1 = len(valid_edges) - rank_d1 - rank_d2
    betti2 = len(triangles) - rank_d2
    return {
        "kind": "got_simplicial_chain_complex",
        "field": "Q",
        "num_vertices": int(n),
        "vertices": list(range(n)),
        "edges": [[int(src), int(dst)] for src, dst in valid_edges],
        "triangles": [[int(a), int(b), int(c)] for a, b, c in triangles],
        "boundary_1": d1.astype(int).tolist(),
        "boundary_2": d2.astype(int).tolist(),
        "ranks": {
            "rank_boundary_1": int(rank_d1),
            "rank_boundary_2": int(rank_d2),
        },
        "betti": {
            "beta_0": int(betti0),
            "beta_1": int(betti1),
            "beta_2": int(betti2),
        },
        "euler_characteristic": int(n - len(valid_edges) + len(triangles)),
    }


def projective_resolution_certificate(num_vertices: int, *, max_vertices: int = 8) -> dict[str, Any]:
    """Return an exact symbolic object in the bounded derived category."""

    n = max(1, min(int(num_vertices), int(max_vertices)))
    certificate = cyclic_stanley_reisner_resolution_certificate(n)
    certificate.setdefault("fitting_entry_ideal_rows", certificate.get("fitting_entry_ideal_generators", []))
    certificate.setdefault("fitting_summary_rows", certificate.get("fitting_determinantal_summary", []))
    certificate.setdefault("dg_product_summary_rows", certificate.get("dg_product_summary_by_bidegree", []))
    return {
        "kind": "derived_category_projective_resolution_object",
        "ambient_category": "D^b(grmod-S)",
        "object": "S/I_Delta",
        "quasi_isomorphism_model": "Taylor DG algebra resolving the Stanley-Reisner quotient",
        "resolution": certificate,
    }


def derived_category_objects_from_batch(
    edge_index: torch.Tensor | np.ndarray,
    node_mask: torch.Tensor | np.ndarray | None = None,
    edge_mask: torch.Tensor | np.ndarray | None = None,
    *,
    max_vertices: int = 8,
    max_edges: int = 64,
) -> list[dict[str, Any]]:
    """Emit actual finite derived-category objects for a batch."""

    edge_array = edge_index.detach().cpu().numpy() if torch.is_tensor(edge_index) else np.asarray(edge_index)
    if edge_array.ndim == 2:
        edge_array = edge_array[None, ...]
    batch = int(edge_array.shape[0])
    node_array = node_mask.detach().cpu().numpy() if torch.is_tensor(node_mask) else node_mask
    edge_mask_array = edge_mask.detach().cpu().numpy() if torch.is_tensor(edge_mask) else edge_mask
    objects: list[dict[str, Any]] = []
    for index in range(batch):
        if node_array is None:
            num_vertices = int(max_vertices)
        else:
            num_vertices = int(np.asarray(node_array[index]).astype(bool).sum())
        edges = edge_array[index]
        if edge_mask_array is not None:
            edges = edges[np.asarray(edge_mask_array[index]).astype(bool)]
        chain = chain_complex_from_edges_np(
            num_vertices,
            edges,
            max_vertices=max_vertices,
            max_edges=max_edges,
        )
        resolution = projective_resolution_certificate(chain["num_vertices"] or 1, max_vertices=max_vertices)
        objects.append(
            {
                "kind": "got_trajectory_derived_category_sample",
                "sample_index": int(index),
                "chain_complex": chain,
                "projective_resolution": resolution,
            }
        )
    return objects


def _expand_edges(
    batch: int,
    num_nodes: int,
    edge_index: torch.Tensor | None,
    edge_mask: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_index is None:
        template = default_branch_merge_edges(num_nodes, device=device)
        edge_index = template.unsqueeze(0).expand(batch, -1, -1)
        edge_mask = torch.ones(edge_index.shape[:2], dtype=torch.bool, device=device)
    else:
        edge_index = edge_index.to(device=device, dtype=torch.long)
        if edge_index.ndim == 2:
            edge_index = edge_index.unsqueeze(0).expand(batch, -1, -1)
        if edge_mask is None:
            edge_mask = torch.ones(edge_index.shape[:2], dtype=torch.bool, device=device)
        else:
            edge_mask = edge_mask.to(device=device, dtype=torch.bool)
            if edge_mask.ndim == 1:
                edge_mask = edge_mask.unsqueeze(0).expand(batch, -1)
    return edge_index, edge_mask


def _chain_tensors_for_sample(
    hidden: torch.Tensor,
    node_mask: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    cfg: DerivedCategoryConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    n = min(int(node_mask.long().sum().item()), int(cfg.max_vertices), hidden.shape[0])
    h = hidden[:n]
    if n == 0:
        z = hidden.new_zeros((0, 0))
        return z, z, hidden.new_zeros((0, hidden.shape[-1])), hidden.new_zeros((0, hidden.shape[-1])), {"n": 0, "e": 0, "t": 0}
    edges: list[tuple[int, int]] = []
    for raw_edge, raw_valid in zip(edge_index[: int(cfg.max_edges)], edge_mask[: int(cfg.max_edges)]):
        if not bool(raw_valid.item()):
            continue
        src = int(raw_edge[0].item())
        dst = int(raw_edge[1].item())
        if src < 0 or dst < 0 or src >= n or dst >= n or src == dst:
            continue
        edge = (src, dst)
        if edge not in edges:
            edges.append(edge)
    if not edges:
        template = default_branch_merge_edges(n, device=hidden.device)
        edges = [(int(src), int(dst)) for src, dst in template[: int(cfg.max_edges)].detach().cpu().tolist()]
    d1 = hidden.new_zeros((n, len(edges)))
    edge_features: list[torch.Tensor] = []
    for col, (src, dst) in enumerate(edges):
        d1[src, col] -= 1.0
        d1[dst, col] += 1.0
        edge_features.append(h[dst] - h[src])
    edge_h = torch.stack(edge_features, dim=0) if edge_features else hidden.new_zeros((0, hidden.shape[-1]))

    triangles = _triangles_from_edges(n, edges)
    edge_to_col = {edge: index for index, edge in enumerate(edges)}
    d2 = hidden.new_zeros((len(edges), len(triangles)))
    tri_features: list[torch.Tensor] = []
    for col, (a, b, c) in enumerate(triangles):
        tri_features.append((h[a] + h[b] + h[c]) / 3.0)
        for edge, coeff in [((b, c), 1.0), ((a, c), -1.0), ((a, b), 1.0)]:
            row = edge_to_col.get(edge)
            sign = 1.0
            if row is None:
                row = edge_to_col.get((edge[1], edge[0]))
                sign = -1.0
            if row is not None:
                d2[row, col] += coeff * sign
    tri_h = torch.stack(tri_features, dim=0) if tri_features else hidden.new_zeros((0, hidden.shape[-1]))
    return d1, d2, edge_h, tri_h, {"n": n, "e": len(edges), "t": len(triangles)}


def _transport(target: torch.Tensor, source: torch.Tensor, temperature: float) -> torch.Tensor:
    if target.shape[0] == 0 or source.shape[0] == 0:
        return target.new_zeros((target.shape[0], source.shape[0]))
    sim = F.normalize(target.float(), dim=-1) @ F.normalize(source.float(), dim=-1).transpose(0, 1)
    return torch.softmax(sim / max(float(temperature), 1e-4), dim=0)


def analogical_derived_category_loss(
    hidden: torch.Tensor,
    *,
    node_mask: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
    edge_mask: torch.Tensor | None = None,
    config: DerivedCategoryConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Compare branch/merge trajectories by finite chain maps and resolutions."""

    cfg = config or DerivedCategoryConfig()
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [batch, nodes, dim]")
    h = torch.nan_to_num(hidden.float(), nan=0.0, posinf=30.0, neginf=-30.0)
    batch, num_nodes, _dim = h.shape
    device = h.device
    if node_mask is None:
        node_mask = torch.ones((batch, num_nodes), dtype=torch.bool, device=device)
    else:
        node_mask = node_mask.to(device=device, dtype=torch.bool)
    edge_index, edge_mask = _expand_edges(batch, num_nodes, edge_index, edge_mask, device)
    zero = h.new_zeros(())
    if batch < 2:
        return {
            "derived_category_loss": zero,
            "derived_category_chain_map_residual": zero,
            "derived_category_mapping_cone_residual": zero,
            "derived_category_boundary_2_residual": zero,
            "derived_category_betti_transport_residual": zero,
            "derived_category_projective_resolution_distance": zero,
            "derived_category_exact_projective_dimension": zero,
            "derived_category_exact_regularity": zero,
            "derived_category_exact_minimal_total_betti": zero,
            "derived_category_pairs": zero,
        }

    chain_terms: list[torch.Tensor] = []
    cone_terms: list[torch.Tensor] = []
    d2_terms: list[torch.Tensor] = []
    betti_terms: list[torch.Tensor] = []
    resolution_terms: list[torch.Tensor] = []
    exact_pd: list[float] = []
    exact_reg: list[float] = []
    exact_betti: list[float] = []
    pair_count = 0
    for src_index in range(batch):
        dst_index = (src_index + 1) % batch
        d1_s, d2_s, edge_h_s, tri_h_s, counts_s = _chain_tensors_for_sample(
            h[src_index], node_mask[src_index], edge_index[src_index], edge_mask[src_index], cfg
        )
        d1_t, d2_t, edge_h_t, tri_h_t, counts_t = _chain_tensors_for_sample(
            h[dst_index], node_mask[dst_index], edge_index[dst_index], edge_mask[dst_index], cfg
        )
        if counts_s["n"] == 0 or counts_t["n"] == 0 or counts_s["e"] == 0 or counts_t["e"] == 0:
            continue
        p0 = _transport(h[dst_index, : counts_t["n"]], h[src_index, : counts_s["n"]], cfg.transport_temperature)
        p1 = _transport(edge_h_t, edge_h_s, cfg.transport_temperature)
        residual_1 = d1_t @ p1 - p0 @ d1_s
        chain_terms.append(residual_1.pow(2).mean())

        if counts_s["t"] > 0 and counts_t["t"] > 0:
            p2 = _transport(tri_h_t, tri_h_s, cfg.transport_temperature)
            residual_2 = d2_t @ p2 - p1 @ d2_s
            d2_term = residual_2.pow(2).mean()
        else:
            d2_term = zero
        d2_terms.append(d2_term)
        cone_terms.append((residual_1.pow(2).mean() + d2_term).sqrt())

        chain_s = chain_complex_from_edges_np(
            counts_s["n"],
            edge_index[src_index, : int(cfg.max_edges)].detach().cpu().numpy(),
            max_vertices=cfg.max_vertices,
            max_edges=cfg.max_edges,
        )
        chain_t = chain_complex_from_edges_np(
            counts_t["n"],
            edge_index[dst_index, : int(cfg.max_edges)].detach().cpu().numpy(),
            max_vertices=cfg.max_vertices,
            max_edges=cfg.max_edges,
        )
        betti_s = chain_s["betti"]
        betti_t = chain_t["betti"]
        betti_delta = (
            abs(float(betti_s["beta_0"]) - float(betti_t["beta_0"]))
            + abs(float(betti_s["beta_1"]) - float(betti_t["beta_1"]))
            + abs(float(betti_s["beta_2"]) - float(betti_t["beta_2"]))
        ) / max(1.0, float(counts_s["n"] + counts_t["n"]))
        betti_terms.append(h.new_tensor(betti_delta))

        n_resolution = max(1, min(int(cfg.max_vertices), counts_s["n"]))
        resolution = cyclic_stanley_reisner_resolution_metrics(n_resolution)
        exact_pd.append(float(resolution.projective_dimension))
        exact_reg.append(float(resolution.regularity))
        exact_betti.append(float(resolution.minimal_total_betti))
        branch_merge = got_dag_metrics(
            h[src_index : src_index + 1],
            node_mask=node_mask[src_index : src_index + 1],
            edge_index=edge_index[src_index : src_index + 1],
            edge_mask=edge_mask[src_index : src_index + 1],
        )
        target_pd = float(resolution.projective_dimension) / max(1.0, float(n_resolution))
        target_reg = float(resolution.regularity) / max(1.0, float(n_resolution))
        resolution_energy = (
            (branch_merge["got_dag_simplex_edge_density_batch"][0] - target_pd) ** 2
            + (branch_merge["got_dag_triangle_density_batch"][0] - target_reg) ** 2
            + 0.01 * branch_merge["got_dag_back_edge_fraction_batch"][0].pow(2)
        )
        resolution_terms.append(resolution_energy)
        pair_count += 1

    if pair_count == 0:
        return {
            "derived_category_loss": zero,
            "derived_category_chain_map_residual": zero,
            "derived_category_mapping_cone_residual": zero,
            "derived_category_boundary_2_residual": zero,
            "derived_category_betti_transport_residual": zero,
            "derived_category_projective_resolution_distance": zero,
            "derived_category_exact_projective_dimension": zero,
            "derived_category_exact_regularity": zero,
            "derived_category_exact_minimal_total_betti": zero,
            "derived_category_pairs": zero,
        }

    chain_residual = torch.stack(chain_terms).mean()
    cone_residual = torch.stack(cone_terms).mean()
    boundary_2_residual = torch.stack(d2_terms).mean()
    betti_residual = torch.stack(betti_terms).mean()
    resolution_distance = torch.stack(resolution_terms).mean()
    loss = (
        float(cfg.chain_weight) * chain_residual
        + float(cfg.cone_weight) * cone_residual
        + float(cfg.betti_weight) * betti_residual
        + float(cfg.resolution_weight) * resolution_distance
    )
    return {
        "derived_category_loss": loss,
        "derived_category_chain_map_residual": chain_residual.detach(),
        "derived_category_mapping_cone_residual": cone_residual.detach(),
        "derived_category_boundary_2_residual": boundary_2_residual.detach(),
        "derived_category_betti_transport_residual": betti_residual.detach(),
        "derived_category_projective_resolution_distance": resolution_distance.detach(),
        "derived_category_exact_projective_dimension": h.new_tensor(float(np.mean(exact_pd)) if exact_pd else 0.0),
        "derived_category_exact_regularity": h.new_tensor(float(np.mean(exact_reg)) if exact_reg else 0.0),
        "derived_category_exact_minimal_total_betti": h.new_tensor(float(np.mean(exact_betti)) if exact_betti else 0.0),
        "derived_category_pairs": h.new_tensor(float(pair_count)),
    }


def derived_category_feature_summary(
    hidden: torch.Tensor,
    *,
    node_mask: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
    edge_mask: torch.Tensor | None = None,
    config: DerivedCategoryConfig | None = None,
) -> torch.Tensor:
    """Return bounded derived-category features for memory retrieval teachers."""

    cfg = config or DerivedCategoryConfig()
    metrics = got_dag_metrics(hidden, node_mask=node_mask, edge_index=edge_index, edge_mask=edge_mask)
    n = max(1, min(int(cfg.max_vertices), hidden.shape[1]))
    resolution = cyclic_stanley_reisner_resolution_metrics(n)
    exact = hidden.new_tensor(
        [
            float(resolution.projective_dimension) / max(1.0, float(n)),
            float(resolution.regularity) / max(1.0, float(n)),
            math.log1p(float(resolution.minimal_total_betti)) / max(1.0, math.log1p(2.0**n)),
            float(resolution.betti_entropy),
        ]
    )
    base = torch.stack(
        [
            metrics["got_dag_branch_count_batch"],
            metrics["got_dag_merge_count_batch"],
            metrics["got_dag_back_edge_fraction_batch"],
            metrics["got_dag_simplex_edge_density_batch"],
            metrics["got_dag_triangle_density_batch"],
            metrics["got_dag_branch_diversity_batch"],
            metrics["got_dag_merge_scatter_batch"],
        ],
        dim=-1,
    )
    exact_expand = exact.unsqueeze(0).expand(hidden.shape[0], -1)
    return torch.cat([base, exact_expand], dim=-1)
