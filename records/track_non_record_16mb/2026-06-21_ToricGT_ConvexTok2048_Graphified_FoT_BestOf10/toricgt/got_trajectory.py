"""Branching graph-of-thought trajectory utilities.

Graph-of-thought reasoning is represented as a directed acyclic graph of
reasoning-step vertices, not as a single chain.  The helpers in this module are
small enough for the training loop but explicit about branch fan-out, merge
fan-in, and the local simplex structure induced by DAG edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class GoTDAGConfig:
    branch_min_variance: float = 0.02
    merge_max_scatter: float = 0.20
    linear_chain_max_fraction: float = 0.35
    acyclicity_weight: float = 1.0
    branch_weight: float = 0.35
    merge_weight: float = 0.35
    linearity_weight: float = 0.25
    balance_weight: float = 0.10
    eps: float = 1e-6


def default_branch_merge_edges(num_nodes: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return a deterministic diamond-DAG template for sequence-only states.

    Each four-step window forms a small branch/merge cell
    ``i -> {i+1, i+2} -> i+3``.  Consecutive windows are linked through their
    join vertices.  This is used only when a dataset did not supply explicit
    graph-of-thought edges.
    """

    if num_nodes <= 1:
        return torch.zeros((0, 2), dtype=torch.long, device=device)
    edges: list[tuple[int, int]] = []
    for start in range(0, max(1, num_nodes - 1), 3):
        a = start
        b = min(start + 1, num_nodes - 1)
        c = min(start + 2, num_nodes - 1)
        d = min(start + 3, num_nodes - 1)
        if a < b:
            edges.append((a, b))
        if a < c and c != b:
            edges.append((a, c))
        if b < d and d != b:
            edges.append((b, d))
        if c < d and d != c:
            edges.append((c, d))
        if d + 1 < num_nodes:
            edges.append((d, d + 1))
    dedup = sorted(set(edge for edge in edges if edge[0] != edge[1]))
    return torch.tensor(dedup, dtype=torch.long, device=device)


def default_branch_merge_edges_np(num_nodes: int) -> np.ndarray:
    edges = default_branch_merge_edges(num_nodes, device=None)
    return edges.cpu().numpy().astype(np.int64)


def _expand_edges(
    batch_size: int,
    num_nodes: int,
    edge_index: torch.Tensor | None,
    edge_mask: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_index is None:
        template = default_branch_merge_edges(num_nodes, device=device)
        if template.numel() == 0:
            edge_index = torch.zeros((batch_size, 0, 2), dtype=torch.long, device=device)
            edge_mask = torch.zeros((batch_size, 0), dtype=torch.bool, device=device)
        else:
            edge_index = template.unsqueeze(0).expand(batch_size, -1, -1)
            edge_mask = torch.ones((batch_size, template.shape[0]), dtype=torch.bool, device=device)
    else:
        edge_index = edge_index.to(device=device, dtype=torch.long)
        if edge_index.ndim == 2:
            edge_index = edge_index.unsqueeze(0).expand(batch_size, -1, -1)
        if edge_mask is None:
            edge_mask = torch.ones(edge_index.shape[:2], dtype=torch.bool, device=device)
        else:
            edge_mask = edge_mask.to(device=device, dtype=torch.bool)
            if edge_mask.ndim == 1:
                edge_mask = edge_mask.unsqueeze(0).expand(batch_size, -1)
    if edge_index.shape[0] != batch_size:
        raise ValueError("edge_index batch dimension does not match hidden states")
    return edge_index, edge_mask


def got_dag_metrics(
    hidden: torch.Tensor,
    *,
    node_mask: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
    edge_mask: torch.Tensor | None = None,
    config: GoTDAGConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Compute branch/merge metrics and a finite structural training loss.

    ``hidden`` has shape ``[batch, nodes, dim]``.  Edges point from predecessor
    reasoning-step vertices to successor vertices.  When explicit edges are
    missing, a deterministic diamond-DAG template is used so sequence-only
    models still learn branch-and-merge reasoning rather than a single path.
    """

    cfg = config or GoTDAGConfig()
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [batch, nodes, dim]")
    h = torch.nan_to_num(hidden.float(), nan=0.0, posinf=30.0, neginf=-30.0)
    batch, num_nodes, _ = h.shape
    device = h.device
    if node_mask is None:
        node_mask = torch.ones((batch, num_nodes), dtype=torch.bool, device=device)
    else:
        node_mask = node_mask.to(device=device, dtype=torch.bool)
    edge_index, edge_mask = _expand_edges(batch, num_nodes, edge_index, edge_mask, device)
    if edge_index.numel() == 0:
        zero = h.new_zeros(())
        return {
            "got_dag_loss": zero,
            "got_dag_loss_batch": h.new_zeros((batch,)),
            "got_dag_branch_count_batch": h.new_zeros((batch,)),
            "got_dag_merge_count_batch": h.new_zeros((batch,)),
            "got_dag_back_edge_fraction_batch": h.new_zeros((batch,)),
            "got_dag_branch_diversity_batch": h.new_zeros((batch,)),
            "got_dag_merge_scatter_batch": h.new_zeros((batch,)),
            "got_dag_balance_residual_batch": h.new_zeros((batch,)),
            "got_dag_simplex_edge_density_batch": h.new_zeros((batch,)),
            "got_dag_triangle_density_batch": h.new_zeros((batch,)),
            "got_dag_linear_chain_fraction_batch": h.new_zeros((batch,)),
            "got_dag_branch_merge_edge_fraction_batch": h.new_zeros((batch,)),
            "got_dag_branch_count": zero,
            "got_dag_merge_count": zero,
            "got_dag_edge_count": zero,
            "got_dag_back_edge_fraction": zero,
            "got_dag_branch_diversity": zero,
            "got_dag_merge_scatter": zero,
            "got_dag_balance_residual": zero,
            "got_dag_max_out_degree": zero,
            "got_dag_max_in_degree": zero,
            "got_dag_simplex_edge_density": zero,
            "got_dag_triangle_density": zero,
            "got_dag_linear_chain_fraction": zero,
            "got_dag_branch_merge_edge_fraction": zero,
        }

    src = edge_index[..., 0].clamp(0, max(num_nodes - 1, 0))
    dst = edge_index[..., 1].clamp(0, max(num_nodes - 1, 0))
    valid = edge_mask & node_mask.gather(1, src) & node_mask.gather(1, dst) & (src != dst)
    valid_f = valid.float()
    out_degree = torch.zeros((batch, num_nodes), device=device, dtype=torch.float32)
    in_degree = torch.zeros((batch, num_nodes), device=device, dtype=torch.float32)
    out_degree.scatter_add_(1, src, valid_f)
    in_degree.scatter_add_(1, dst, valid_f)
    branch_soft = torch.sigmoid(3.0 * (out_degree - 1.0)) * node_mask.float()
    merge_soft = torch.sigmoid(3.0 * (in_degree - 1.0)) * node_mask.float()
    branch_count = branch_soft.sum(dim=1)
    merge_count = merge_soft.sum(dim=1)
    edge_count = valid_f.sum(dim=1).clamp_min(1.0)
    src_out_degree = out_degree.gather(1, src)
    dst_in_degree = in_degree.gather(1, dst)
    branch_incident = (src_out_degree > 1.0) & valid
    merge_incident = (dst_in_degree > 1.0) & valid
    branch_merge_incident = branch_incident | merge_incident
    linear_chain_edge = valid & ~branch_merge_incident
    linear_chain_fraction = linear_chain_edge.float().sum(dim=1) / edge_count
    branch_merge_edge_fraction = branch_merge_incident.float().sum(dim=1) / edge_count

    batch_index = torch.arange(batch, device=device).unsqueeze(1).expand_as(src)
    h_src = h[batch_index, src]
    h_dst = h[batch_index, dst]
    edge_vec = h_dst - h_src
    edge_len = edge_vec.pow(2).mean(dim=-1).sqrt()

    dst_sum = torch.zeros_like(h)
    dst_sum.scatter_add_(1, src.unsqueeze(-1).expand(-1, -1, h.shape[-1]), h_dst * valid_f.unsqueeze(-1))
    dst_mean = dst_sum / out_degree.clamp_min(1.0).unsqueeze(-1)
    branch_var_edges = (h_dst - dst_mean[batch_index, src]).pow(2).mean(dim=-1) * valid_f
    branch_var = torch.zeros((batch, num_nodes), device=device, dtype=torch.float32)
    branch_var.scatter_add_(1, src, branch_var_edges)
    branch_var = branch_var / out_degree.clamp_min(1.0)
    branch_diversity = (branch_var * branch_soft).sum(dim=1) / branch_soft.sum(dim=1).clamp_min(1.0)

    pred_sum = torch.zeros_like(h)
    pred_sum.scatter_add_(1, dst.unsqueeze(-1).expand(-1, -1, h.shape[-1]), h_src * valid_f.unsqueeze(-1))
    pred_mean = pred_sum / in_degree.clamp_min(1.0).unsqueeze(-1)
    merge_var_edges = (h_src - pred_mean[batch_index, dst]).pow(2).mean(dim=-1) * valid_f
    merge_var = torch.zeros((batch, num_nodes), device=device, dtype=torch.float32)
    merge_var.scatter_add_(1, dst, merge_var_edges)
    merge_var = merge_var / in_degree.clamp_min(1.0)
    merge_scatter = (merge_var * merge_soft).sum(dim=1) / merge_soft.sum(dim=1).clamp_min(1.0)

    back_edge_fraction = ((src >= dst).float() * valid_f).sum(dim=1) / edge_count
    balance = (branch_count - merge_count).abs() / (branch_count + merge_count + 1.0)
    possible_edges = node_mask.float().sum(dim=1).clamp_min(2.0)
    simplex_edge_density = edge_count / (possible_edges * (possible_edges - 1.0))

    adjacency = torch.zeros((batch, num_nodes, num_nodes), device=device, dtype=torch.float32)
    adjacency[batch_index, src, dst] = torch.maximum(adjacency[batch_index, src, dst], valid_f)
    undirected = torch.maximum(adjacency, adjacency.transpose(1, 2))
    tri = torch.einsum("bij,bjk,bki->b", undirected, undirected, undirected) / 6.0
    possible_tri = possible_edges * (possible_edges - 1.0) * (possible_edges - 2.0) / 6.0
    triangle_density = tri / possible_tri.clamp_min(1.0)

    branch_collapse = F.relu(float(cfg.branch_min_variance) - branch_diversity)
    merge_spread = F.relu(merge_scatter - float(cfg.merge_max_scatter))
    linearity_penalty = F.relu(linear_chain_fraction - float(cfg.linear_chain_max_fraction))
    loss_per_batch = (
        float(cfg.branch_weight) * branch_collapse
        + float(cfg.merge_weight) * merge_spread
        + float(cfg.acyclicity_weight) * back_edge_fraction
        + float(cfg.linearity_weight) * linearity_penalty
        + float(cfg.balance_weight) * balance
    )
    if torch.isfinite(edge_len).all():
        length_scale = edge_len.detach().masked_fill(~valid, 0.0).sum(dim=1) / edge_count
        loss_per_batch = loss_per_batch + 0.01 * torch.nan_to_num(length_scale, nan=0.0)

    return {
        "got_dag_loss": loss_per_batch.mean(),
        "got_dag_loss_batch": loss_per_batch,
        "got_dag_branch_count_batch": branch_count,
        "got_dag_merge_count_batch": merge_count,
        "got_dag_back_edge_fraction_batch": back_edge_fraction,
        "got_dag_branch_diversity_batch": branch_diversity,
        "got_dag_merge_scatter_batch": merge_scatter,
        "got_dag_balance_residual_batch": balance,
        "got_dag_simplex_edge_density_batch": simplex_edge_density,
        "got_dag_triangle_density_batch": triangle_density,
        "got_dag_linear_chain_fraction_batch": linear_chain_fraction,
        "got_dag_branch_merge_edge_fraction_batch": branch_merge_edge_fraction,
        "got_dag_branch_count": branch_count.mean().detach(),
        "got_dag_merge_count": merge_count.mean().detach(),
        "got_dag_edge_count": edge_count.mean().detach(),
        "got_dag_back_edge_fraction": back_edge_fraction.mean().detach(),
        "got_dag_branch_diversity": branch_diversity.mean().detach(),
        "got_dag_merge_scatter": merge_scatter.mean().detach(),
        "got_dag_balance_residual": balance.mean().detach(),
        "got_dag_max_out_degree": out_degree.max(dim=1).values.mean().detach(),
        "got_dag_max_in_degree": in_degree.max(dim=1).values.mean().detach(),
        "got_dag_simplex_edge_density": simplex_edge_density.mean().detach(),
        "got_dag_triangle_density": triangle_density.mean().detach(),
        "got_dag_linear_chain_fraction": linear_chain_fraction.mean().detach(),
        "got_dag_branch_merge_edge_fraction": branch_merge_edge_fraction.mean().detach(),
    }


def got_dag_summary_np(
    num_nodes: int,
    edges: np.ndarray | None = None,
) -> dict[str, float]:
    """Exact branch/merge counts for serialized trajectory-memory records."""

    if num_nodes <= 0:
        return {
            "branch_count": 0.0,
            "merge_count": 0.0,
            "edge_count": 0.0,
            "back_edge_fraction": 0.0,
            "simplex_edge_density": 0.0,
            "linear_chain_fraction": 0.0,
            "branch_merge_edge_fraction": 0.0,
        }
    if edges is None:
        edges = default_branch_merge_edges_np(num_nodes)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    valid = edges[(edges[:, 0] >= 0) & (edges[:, 1] >= 0) & (edges[:, 0] < num_nodes) & (edges[:, 1] < num_nodes) & (edges[:, 0] != edges[:, 1])]
    out_degree = np.zeros((num_nodes,), dtype=np.float64)
    in_degree = np.zeros((num_nodes,), dtype=np.float64)
    for src, dst in valid:
        out_degree[int(src)] += 1.0
        in_degree[int(dst)] += 1.0
    edge_count = float(len(valid))
    possible_edges = max(1.0, float(num_nodes * (num_nodes - 1)))
    linear_edges = 0
    branch_merge_edges = 0
    for src, dst in valid:
        if out_degree[int(src)] > 1.0 or in_degree[int(dst)] > 1.0:
            branch_merge_edges += 1
        else:
            linear_edges += 1
    return {
        "branch_count": float((out_degree > 1.0).sum()),
        "merge_count": float((in_degree > 1.0).sum()),
        "edge_count": edge_count,
        "back_edge_fraction": float((valid[:, 0] >= valid[:, 1]).mean()) if edge_count else 0.0,
        "simplex_edge_density": edge_count / possible_edges,
        "max_out_degree": float(out_degree.max(initial=0.0)),
        "max_in_degree": float(in_degree.max(initial=0.0)),
        "linear_chain_fraction": float(linear_edges / edge_count) if edge_count else 0.0,
        "branch_merge_edge_fraction": float(branch_merge_edges / edge_count) if edge_count else 0.0,
    }


def graph_json_has_branch_merge(payload: dict[str, Any]) -> bool:
    """Return true when a serialized trajectory graph has both split and join."""

    nodes = payload.get("nodes") or []
    edges = payload.get("edges") or []
    ids = {str(node.get("id", idx)): idx for idx, node in enumerate(nodes)}
    in_degree = [0 for _ in nodes]
    out_degree = [0 for _ in nodes]
    for edge in edges:
        src = ids.get(str(edge.get("source")))
        dst = ids.get(str(edge.get("target")))
        if src is None or dst is None or src == dst:
            continue
        out_degree[src] += 1
        in_degree[dst] += 1
    return any(value > 1 for value in out_degree) and any(value > 1 for value in in_degree)
